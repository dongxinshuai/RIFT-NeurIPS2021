# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

def evaluation(opt, device, model, test_iter):
    model.plm.eval()
    model.cls_to_logit.eval()
    accuracy=[]

    for index, batch in enumerate(test_iter):
        text_x = batch[0].to(device)
        y = batch[1].to(device)
        attention_mask= batch[4].to(device)

        logit = model(text_x, attention_mask)

        _, idx = torch.max(logit, 1) 
        percision=(idx==y).float().mean()
        accuracy.append(percision.data.item())

    return np.mean(accuracy)

def evaluation_by_ascc(opt, device, model, test_iter):
    model.plm.eval()
    model.cls_to_logit.eval()
    accuracy=[]

    for index, batch in enumerate( test_iter):
        text_x = batch[0].to(device)
        y = batch[1].to(device)
        text_subs= batch[2].to(device)
        text_subs_mask= batch[3].to(device)
        attention_mask= batch[4].to(device)
        
        # generate adv text by ascc
        ascc_attack_info = {
            'num_steps': opt.ascc_test_attack_iters,
            'loss_func': 'ce',
            'ascc_w_optm_lr': opt.ascc_w_optm_lr,
            'sparse_weight': opt.ascc_test_attack_sparse_weight,
            'out_type': "text"
        }
        adv_text_x = model.ascc_attack(text_x, attention_mask, y, text_subs, text_subs_mask, ascc_attack_info)
        logit_adv = model(adv_text_x, attention_mask)
        
        _, idx = torch.max(logit_adv, 1) 
        percision=(idx==y).float().mean()
        accuracy.append(percision.data.item())

    return np.mean(accuracy)


def set_params(net, resume_model_path, data_parallel=True):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()

    for key in state_dict.keys():
        if data_parallel:
            new_key = key
        else:
            key1, key2 = key.split('module.')[0], key.split('module.')[1]
            new_key = key1+key2

        sdict[new_key]=state_dict[key]

    try:
        net.load_state_dict(sdict)
    except:
        print("WARNING!!!!!!!! MISSING PARAMETERS. LOADING ANYWAY.")
        net.load_state_dict(sdict,strict=False)

    return net


from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
