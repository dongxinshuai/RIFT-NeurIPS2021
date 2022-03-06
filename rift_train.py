# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter import E

import numpy as np
import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_data_iters

from attack.attack_algorithms.genetic_attack import genetic_attack
from attack.attack_algorithms.pwws_attack import pwws_attack

import opts
import models
import utils 

try:
    import cPickle as pickle
except ImportError:
    import pickle


def train(opt, train_iters_per_y, test_iter):

    # model
    model=models.setup(opt)
    if opt.resume != None:
        model = utils.set_params(model, opt.resume)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        model.device=device

    # optimizer
    from transformers import AdamW
    no_decay = ['plm','plm_teacher']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if opt.optimizer=='adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)
    else:
        raise NotImplementedError

    from utils import WarmupMultiStepLR
    scheduler = WarmupMultiStepLR(optimizer, (50, 80), 0.1, 1.0/10.0, 2, 'linear')

    tik = time.time()
    best_acc_under_ascc = 0
    best_acc_under_pwws = 0
    best_acc_under_genetic = 0
    best_save_dir = None

    for epoch in range(opt.training_epochs):

        sum_loss = 0
        sum_loss_adv = sum_loss_kl = sum_loss_clean  = 0
        sum_loss_mi_adv = sum_loss_mi_clean = sum_loss_mi_giveny_adv = sum_loss_mi_giveny_clean = 0
        sum_loss_params_l2 = 0
        total = 0

        # print hyperparameters
        print('w_adv: ', opt.weight_adv)
        print('w_clean: ', opt.weight_clean)
        print('w_kl: ', opt.weight_kl)
        print('w_mi_clean', opt.weight_mi_clean)
        print('w_mi_adv', opt.weight_mi_adv)
        print('w_mi_giveny_clean', opt.weight_mi_giveny_clean)
        print('w_mi_giveny_adv', opt.weight_mi_giveny_adv)
        print('w_params_l2', opt.weight_params_l2)
        print('mixout_p', opt.mixout_p)

        if opt.dataset == 'imdb':
            zipped_iters_per_y = zip(train_iters_per_y[0], train_iters_per_y[1])
        elif opt.dataset == "snli":
            zipped_iters_per_y = zip(train_iters_per_y[0], train_iters_per_y[1], train_iters_per_y[2])
        else:
            raise NotImplementedError 

        for iters, (batches_per_y) in enumerate(zipped_iters_per_y):
            
            if opt.dataset == "imdb":
                batch_y0, batch_y1 = batches_per_y[0], batches_per_y[1]
                text_x = torch.cat((batch_y0[0].to(device),batch_y1[0].to(device)), 0)
                y = torch.cat((batch_y0[1].to(device),batch_y1[1].to(device)), 0)
                text_subs= torch.cat((batch_y0[2].to(device),batch_y1[2].to(device)), 0)
                text_subs_mask= torch.cat((batch_y0[3].to(device),batch_y1[3].to(device)), 0)
                attention_mask= torch.cat((batch_y0[4].to(device),batch_y1[4].to(device)), 0)
            elif opt.dataset == "snli":
                batch_y0, batch_y1, batch_y2 = batches_per_y[0], batches_per_y[1], batches_per_y[2]
                text_x = torch.cat((batch_y0[0].to(device),batch_y1[0].to(device),batch_y2[0].to(device)), 0)
                y = torch.cat((batch_y0[1].to(device),batch_y1[1].to(device),batch_y2[1].to(device)), 0)
                text_subs= torch.cat((batch_y0[2].to(device),batch_y1[2].to(device),batch_y2[2].to(device)), 0)
                text_subs_mask= torch.cat((batch_y0[3].to(device),batch_y1[3].to(device),batch_y2[3].to(device)), 0)
                attention_mask= torch.cat((batch_y0[4].to(device),batch_y1[4].to(device),batch_y2[4].to(device)), 0)
            else:
                raise NotImplementedError 

            model.plm.train()
            model.cls_to_logit.train()

            # generate adv combination weight by ascc
            ascc_attack_info = {
                'num_steps': opt.ascc_train_attack_iters,
                'loss_func': 'kl',
                'ascc_w_optm_lr': opt.ascc_w_optm_lr,
                'sparse_weight': opt.ascc_train_attack_sparse_weight,
                'out_type': "adv_comb_w"
            }
            adv_comb_w = model.ascc_attack(text_x, attention_mask, y, text_subs, text_subs_mask, ascc_attack_info)

            optimizer.zero_grad()

            # get adv predition
            logit_adv = model.comb_w_to_logit(adv_comb_w, text_subs, attention_mask)

            # clean loss
            logit_clean = model(text_x, attention_mask)
            loss_clean= F.cross_entropy(logit_clean, y)

            # adv loss
            if opt.weight_adv == 0:
                loss_adv = torch.zeros(1).to(device)
            else:
                loss_adv = F.cross_entropy(logit_adv, y)

            # kl loss
            criterion_kl = nn.KLDivLoss(reduction="sum")
            if opt.weight_kl == 0:
                loss_kl = torch.zeros(1).to(device)
            else:
                loss_kl = (1.0 / len(logit_adv)) * criterion_kl(F.log_softmax(logit_adv, dim=1),
                                                            F.softmax(logit_clean, dim=1))

            # mutual information
            if opt.weight_mi_adv == 0:
                loss_mi_adv = torch.zeros(1).to(device)
            else:      
                loss_mi_adv = model.comb_w_to_mutual_info(text_x, adv_comb_w, text_subs, attention_mask, opt.infonce_sim_metric, y=None)
                
            if opt.weight_mi_clean == 0:
                loss_mi_clean = torch.zeros(1).to(device)
            else:
                loss_mi_clean = model.text_to_mutual_info(text_x, attention_mask, opt.infonce_sim_metric, y=None)

            if opt.weight_mi_giveny_adv == 0:
                loss_mi_giveny_adv = torch.zeros(1).to(device)
            else:
                bs_per_y = len(text_x)//opt.label_size
                loss_mi_giveny_adv = 0
                for c in range(opt.label_size):
                    loss_mi_giveny_adv += model.comb_w_to_mutual_info(text_x[c*bs_per_y:(c+1)*bs_per_y], \
                                            adv_comb_w[c*bs_per_y:(c+1)*bs_per_y], text_subs[c*bs_per_y:(c+1)*bs_per_y], \
                                            attention_mask[c*bs_per_y:(c+1)*bs_per_y], opt.infonce_sim_metric, y=c)
                loss_mi_giveny_adv = loss_mi_giveny_adv/opt.label_size

            if opt.weight_mi_giveny_clean == 0:
                loss_mi_giveny_clean = torch.zeros(1).to(device)
            else:
                bs_per_y = len(text_x)//opt.label_size
                loss_mi_giveny_clean = 0
                for c in range(opt.label_size):
                    loss_mi_giveny_clean += model.text_to_mutual_info(text_x[c*bs_per_y:(c+1)*bs_per_y], \
                    attention_mask[c*bs_per_y:(c+1)*bs_per_y], opt.infonce_sim_metric, y=c)

                loss_mi_giveny_clean = loss_mi_giveny_clean/opt.label_size

            loss_params_l2 = model.params_l2()  

            # optimization
            loss = opt.weight_kl*loss_kl + opt.weight_adv*loss_adv + opt.weight_clean*loss_clean \
            + loss_mi_clean*opt.weight_mi_clean + loss_mi_adv*opt.weight_mi_adv \
            + loss_mi_giveny_adv*opt.weight_mi_giveny_adv + loss_mi_giveny_clean*opt.weight_mi_giveny_clean \
            + loss_params_l2*opt.weight_params_l2
    
            loss.backward() 
            optimizer.step()

            # print losses
            total += 1
            sum_loss += loss.item()
            sum_loss_adv += loss_adv.item()
            sum_loss_clean += loss_clean.item()
            sum_loss_kl += loss_kl.item()
            sum_loss_mi_adv += loss_mi_adv.item()
            sum_loss_mi_clean += loss_mi_clean.item()
            sum_loss_mi_giveny_clean += loss_mi_giveny_clean.item()
            sum_loss_mi_giveny_adv += loss_mi_giveny_adv.item()
            sum_loss_params_l2 += loss_params_l2.item()
            _, idx = torch.max(logit_clean, 1) 
            precision=(idx==y).float().mean().item()
            _, idx_adv = torch.max(logit_adv, 1)
            precision_adv=(idx_adv==y).float().mean().item()
            
            out_log = "%d epoch, %d iters - loss: %.3f, loss_kl: %.3f, loss_adv: %.3f, loss_clean: %.3f, loss_mi_clean: %.3f, \
            loss_mi_adv: %.3f, loss_mi_giveny_clean: %.3f, loss_mi_giveny_adv: %.3f, loss_params_l2: %.3f \
            | acc: %.3f acc_adv: %.3f | in %.3f seconds" % \
            (epoch, iters, sum_loss/total, sum_loss_kl/total, sum_loss_adv/total, sum_loss_clean/total, \
            sum_loss_mi_clean/total, sum_loss_mi_adv/total, sum_loss_mi_giveny_clean/total, sum_loss_mi_giveny_adv/total, \
            sum_loss_params_l2/total, precision, precision_adv, time.time()-tik)
            print(out_log)
            tik = time.time()
                
        scheduler.step()

        # test
        if epoch>=10 and epoch%2==1: 
            
            current_model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
            state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
            torch.save(state, current_model_path)

            acc=utils.evaluation(opt, device, model, test_iter)
            print("%d epoch test acc%.4f" % (epoch,acc))

            print("begin ascc attack")
            acc_under_ascc=utils.evaluation_by_ascc(opt, device, model, test_iter)
            best_acc_under_ascc = max(best_acc_under_ascc, acc_under_ascc)
            print("%d epoch test acc under ascc attack  %.4f, best: %.4f" % (epoch, acc_under_ascc, best_acc_under_ascc))

            print("begin pwws attack")
            acc_under_pwws = pwws_attack(opt, model, opt.dataset, opt.pwws_test_num)
            best_acc_under_pwws = max(best_acc_under_pwws, acc_under_pwws)
            print("%d epoch test acc under pwws attack  %.4f, best: %.4f" % (epoch, acc_under_pwws, best_acc_under_pwws))

            print("begin genetic attack")
            acc_under_genetic = genetic_attack(opt, model, opt.dataset, opt.genetic_test_num)
            best_acc_under_genetic = max(best_acc_under_genetic, acc_under_genetic)
            print("%d epoch test acc under genetic attack  %.4f, best: %.4f" % (epoch, acc_under_genetic, best_acc_under_genetic))

            if acc_under_ascc>=best_acc_under_ascc:
                best_save_dir=os.path.join(opt.out_path, "{}_best.pth".format(opt.model))
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, best_save_dir)

    # restore best according to dev set
    #if best_save_dir is not None:
    #    model = utils.set_params(model, best_save_dir)
    #acc=utils.evaluation(opt, device, model, test_iter)
    #print("test acc %.4f" % (acc))
    #adv_acc=utils.evaluation_by_ascc(opt, device, model, test_iter)
    #print("test acc under ascc attack %.4f" % (adv_acc))

    #torch.cuda.empty_cache()
    #print("begin genetic attack")
    #genetic_attack(opt, model, opt.dataset, opt.genetic_test_num)
    #print("begin pwws attack")
    #pwws_attack(opt, model, opt.dataset, opt.pwws_test_num)
    
def main():
    opt = opts.parse_opt()
    print(opt)
    torch.manual_seed(opt.torch_seed)

    train_iters_per_y, test_iter = get_data_iters(opt)

    train(opt, train_iters_per_y, test_iter)
    
if __name__=="__main__": 
    main()