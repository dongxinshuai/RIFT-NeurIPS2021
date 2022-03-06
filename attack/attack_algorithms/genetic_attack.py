# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
try:
    import cPickle as pickle
except ImportError:
    import pickle
from modified_tokenizers import get_tokenizers
from data import split_imdb_files, split_snli_files

def genetic_attack(opt, model, dataset, genetic_test_num):
    
    from ..attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    lm_file_path = opt.imdb_lm_file_path if opt.dataset=='imdb' else opt.snli_lm_file_path
    if opt.lm_constraint_on_genetic_attack:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.substitution_dict_path, lm_file_path)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_files(opt.substitution_dict_path, lm_file_path)

    tokenizer, _ = get_tokenizers(opt.plm_type)

    # process data
    if dataset == 'imdb':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        input_max_len = opt.imdb_input_max_len

        # randomly select test examples
        indexes = [i for i in range(len(test_labels))]
        random.seed(opt.rand_seed)
        random.shuffle(indexes)
        test_texts = [test_texts[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]

        indexes = []
        for i, x in enumerate(test_texts):
            words = x.split()
            if attack_surface.check_in(words):
                indexes.append(i)

        test_texts = [test_texts[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]
        test_num = min(len(test_labels), genetic_test_num)
        test_texts = test_texts[:test_num]
        test_labels = test_labels[:test_num]

        wrapped_model = WrappedModelCls(tokenizer, input_max_len)

    elif dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
        input_max_len = opt.snli_input_max_len
        # randomly select test examples
        indexes = [i for i in range(len(test_labels))]
        random.seed(opt.rand_seed)
        random.shuffle(indexes)
        test_perms = [test_perms[i] for i in indexes]
        test_hypos = [test_hypos[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]

        indexes = []
        for i, x_h in enumerate(test_hypos):
            words = x_h.split()
            if attack_surface.check_in(words):
                indexes.append(i)

        test_perms = [test_perms[i] for i in indexes]
        test_hypos = [test_hypos[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]
        test_num = min(len(test_labels), genetic_test_num)
        test_perms = test_perms[:test_num]
        test_hypos = test_hypos[:test_num]
        test_labels = test_labels[:test_num]

        wrapped_model = WrappedModelNli(tokenizer, input_max_len, opt.plm_type)
    else:
        raise NotImplementedError


    model.plm.eval()
    model.cls_to_logit.eval()
    # genetic attack
    genetic_adversary = GeneticAdversary(opt, attack_surface, num_iters=opt.genetic_iters, pop_size=opt.genetic_pop_size)

    # run genetic attack by multiprocessing
    from multiprocessing import Process, Pipe
    conn_main = []
    conn_p = []
    for i in range(test_num):
        c1, c2 = Pipe()
        conn_main.append(c1)
        conn_p.append(c2)

    process_list = []
    for i in range(test_num):
        if dataset == 'imdb':
            p = Process(target=genetic_adversary.attack_binary_classification, args=(conn_p[i], wrapped_model, test_texts[i], test_labels[i]))
        elif dataset == 'snli':
            p = Process(target=genetic_adversary.attack_nli, args=(conn_p[i], wrapped_model, test_perms[i], test_hypos[i], test_labels[i]))
        else:
            raise NotImplementedError
        p.start()
        process_list.append(p)

    accuracy = process_queries_for_genetic_attack(model, test_num, input_max_len, process_list, conn_main)
    #print("acc under genetic attack: ", accuracy)
    return accuracy

def process_queries_for_genetic_attack(model, test_num, input_max_len, process_list, conn_main, batch_size = 32):

    tested = 0
    correct = 0

    process_done=[False for i in range(test_num)]
    t_start = time.clock()

    device = model.device
    text_x = torch.zeros(batch_size, input_max_len, dtype=torch.long).to(device)
    attention_mask = torch.zeros(batch_size, input_max_len, dtype=torch.long).to(device)

    process_id = 0

    process_id = 0
    bs_count=0
    res_dict = {}

    # polling
    while(1):
        # stop when all test examples are processed
        if tested == test_num:
            break

        # collect queries
        if process_done[process_id]==False:
            cm=conn_main[process_id]
            if cm.poll():
                msg = cm.recv()
                # msg == 0 or 1 means genetic attack for this example is finished
                if msg == 0 or msg == 1:
                    tested += 1
                    correct += msg
                    cm.close()
                    process_done[process_id]=True
                    process_list[process_id].join()
                    print('acc under genetic {}/{}, time cost: {}'.format(correct, tested, time.clock() - t_start))

                else:
                    new_text_x, new_attention_mask = msg
                    text_x[bs_count] = new_text_x.to(device)
                    attention_mask[bs_count] = new_attention_mask.to(device)
                    res_dict[process_id]=bs_count
                    bs_count +=1

        # process queries by batches
        if bs_count==batch_size or bs_count>=(test_num-tested):
            with torch.no_grad():
                logit = model(text_x, attention_mask).detach().cpu()
                    
            for key in res_dict.keys():
                cm=conn_main[key]
                cm.send(logit[res_dict[key]])

            bs_count = 0
            res_dict = {}

        # increase process_id
        process_id=(process_id+1)%test_num

    return correct/tested

class GeneticAdversary(object):
    def __init__(self, opt, attack_surface, num_iters=20, pop_size=60):
        super(GeneticAdversary, self).__init__()
        self.attack_surface = attack_surface
        self.num_iters = num_iters
        self.pop_size = pop_size
        self.opt = opt

    def perturb_binary_classification(self, words, choices, model, y, conn_p):
        if all(len(c) == 1 for c in choices): return words
        good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
        idx = random.sample(good_idxs, 1)[0]
        x_list = [' '.join(words[:idx] + [w_new] + words[idx+1:])
                for w_new in choices[idx]]
        
        logits = [model.query(x, conn_p) for x in x_list]
        preds = [F.softmax(logit, dim=-1).cpu().numpy() for logit in logits]

        preds_of_y = [p[y] for p in preds]
        best_idx = min(enumerate(preds_of_y), key=lambda x: x[1])[0]
        cur_words = list(words)
        cur_words[idx] = choices[idx][best_idx]
        return cur_words, preds_of_y[best_idx]

    def attack_binary_classification(self, conn_p, model, x, y):

        random.seed(self.opt.rand_seed)

        words = x.split()
        y = np.argmax(y) 
        # First query the ori example
        original_logit = model.query(x, conn_p)
        original_preds = F.softmax(original_logit, dim=-1).cpu().numpy()
        if np.argmax(original_preds) != y :
            conn_p.send(0)
            conn_p.close()
            return 

        # Now run adversarial search
        words = x.split()
        swaps = self.attack_surface.get_swaps(words)
        choices = [[w] + cur_swaps for w, cur_swaps in zip(words, swaps)]
        found = False
        population = [self.perturb_binary_classification(words, choices, model, y, conn_p)
                        for i in range(self.pop_size)]
        for g in range(self.num_iters):
            best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
            #print('Iteration %d: %.6f' % (g, population[best_idx][1]))
            if population[best_idx][1] < 0.5: # because it is binary classification
                found = True
                #print('ADVERSARY SUCCESS')
                conn_p.send(0)
                conn_p.close()
                return
            new_population = [population[best_idx]]
            p_y = np.array([m for c, m in population])
            temp = 1-p_y + 1e-8
            sample_probs = (temp) / np.sum(temp) 
            #sample_probs = sample_probs + 1e-8
            for i in range(1, self.pop_size):
                parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
                child_mut, new_p_y = self.perturb_binary_classification(child, choices, model, y, conn_p)
                new_population.append((child_mut, new_p_y))
            population = new_population
        else:
            #print('ADVERSARY FAILURE', 'Iteration %d: %.6f' % (g, population[best_idx][1]))
            conn_p.send(1)# correct
            conn_p.close()
            return

    def get_margins_nli(self, model_output, gold_labels):
        logits = model_output

        true_class_pred = logits[gold_labels]

        temp = logits.copy()
        temp[gold_labels]=-1e20
        highest_false_pred = temp.max()
        value_margin = true_class_pred - highest_false_pred
        return value_margin

    def perturb_nli(self, x_p, hypo, choices, model, y, conn_p):
        if all(len(c) == 1 for c in choices):
            value_margin = self.get_margins_nli( model.query(x_p, ' '.join(hypo), conn_p), y)
            return hypo, value_margin.item()
        good_idxs = [i for i, c in enumerate(choices) if len(c) > 1]
        idx = random.sample(good_idxs, 1)[0]
        x_h_list = [' '.join(hypo[:idx] + [w_new] + hypo[idx+1:])
                for w_new in choices[idx]]
        querry_list = [model.query(x_p, x_h, conn_p) for x_h in x_h_list]
        best_replacement_idx = None
        worst_margin = float('inf')
        for idx_in_choices, logits in enumerate(querry_list):
            value_margin = self.get_margins_nli(logits, y)
            if best_replacement_idx is None or value_margin.item() < worst_margin:
                best_replacement_idx = idx_in_choices
                worst_margin = value_margin.item()

        cur_words = list(hypo)
        cur_words[idx] = choices[idx][best_replacement_idx]
        return cur_words, worst_margin

    def attack_nli(self, conn_p, model, x_p, x_h, y):
        random.seed(self.opt.rand_seed)

        y = np.argmax(y) 
        # First query the ori example
        orig_pred = model.query(x_p, x_h, conn_p)
        if np.argmax(orig_pred) != y :
            conn_p.send(0)
            conn_p.close()
            return 

        x_h_words = x_h.split()
        swaps = self.attack_surface.get_swaps(x_h_words)
        choices = [[w] + cur_swaps for w, cur_swaps in zip(x_h_words, swaps)]
        found = False
        population = [self.perturb_nli(x_p, x_h_words, choices, model, y, conn_p)
                        for i in range(self.pop_size)]
        for g in range(self.num_iters):
            best_idx = min(enumerate(population), key=lambda x: x[1][1])[0]
            #print('Iteration %d: %.6f' % (g, population[best_idx][1]))
            if population[best_idx][1] < 0:
                found = True
                #print('ADVERSARY SUCCESS')
                conn_p.send(0)
                conn_p.close()
                return
            new_population = [population[best_idx]]

            margins = np.array([m for c, m in population])
            adv_probs = 1 / (1 + np.exp(margins)) + 1e-4
            # Sigmoid of negative margin, for probabilty of wrong class
            # Add 1e-4 for numerical stability
            sample_probs = adv_probs / np.sum(adv_probs)

            for i in range(1, self.pop_size):
                parent1 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                parent2 = population[np.random.choice(range(len(population)), p=sample_probs)][0]
                child = [random.sample([w1, w2], 1)[0] for (w1, w2) in zip(parent1, parent2)]
                child_mut, new_margin = self.perturb_nli(x_p, child, choices, model, y, conn_p)
                new_population.append((child_mut, new_margin))
            population = new_population

        #print('ADVERSARY FAILURE', 'Iteration %d: %.6f' % (g, population[best_idx][1]))
        conn_p.send(1)# correct
        conn_p.close()
        return

class WrappedModelCls(object):
    def __init__(self, tokenizer, input_max_len):
        super(WrappedModelCls, self).__init__()
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len

    def tokenize(self, x):
        return self.tokenizer(x, self.input_max_len)

    def get(self, msg, conn_p):
        conn_p.send(msg)
        return conn_p.recv()

    def query(self, x, conn_p):

        token = self.tokenize(x)

        text = np.array([token['input_ids']])
        text = torch.tensor(text,dtype=torch.long).to('cpu')
        attention_mask = np.array([token['attention_mask']])
        attention_mask = torch.tensor(attention_mask,dtype=torch.long).to('cpu')

        logit = self.get((text, attention_mask), conn_p).squeeze(0)
        #prediction = F.softmax(logit, dim=-1).detach().cpu().numpy()
        return logit
        

class WrappedModelNli(object):
    def __init__(self, tokenizer, input_max_len, plm_type):
        super(WrappedModelNli, self).__init__()
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.plm_type = plm_type

    def tokenize(self, x_p, x_h):
        if self.plm_type=="bert":
            token = self.tokenizer(x_p+" [SEP] "+x_h, self.input_max_len)
        elif self.plm_type=="roberta":
            token = self.tokenizer(x_p+"</s>"+x_h, self.input_max_len)
        else:
            raise NotImplementedError

        return token

    def get(self, msg, conn_p):
        conn_p.send(msg)
        return conn_p.recv()

    def query(self, x_p, x_h, conn_p):

        token = self.tokenize(x_p, x_h)

        text = np.array([token['input_ids']])
        text = torch.tensor(text,dtype=torch.long).to('cpu')
        attention_mask = np.array([token['attention_mask']])
        attention_mask = torch.tensor(attention_mask,dtype=torch.long).to('cpu')

        logit = self.get((text, attention_mask), conn_p).squeeze(0)

        return logit.detach().cpu().numpy()
        #prediction = F.softmax(logit, dim=-1).detach().cpu().numpy()
        #return prediction