# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
from modified_tokenizers import get_tokenizers
import torch.utils.data
from tqdm import tqdm
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle

def read_imdb_files(opt, filetype):
    
    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    all_texts = []
    file_list = []
    path = os.path.join('dataset/aclImdb/')
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            from nltk import word_tokenize
            x_raw = f.readlines()[0].strip().replace('<br />', ' ')
            x_toks = word_tokenize(x_raw)
            all_texts.append(' '.join(x_toks))

    return all_texts, all_labels

def split_imdb_files(opt):
    filename = opt.split_imdb_files_path
    if os.path.exists(filename):
        print('Read processed IMDB dataset')
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_texts=saved['train_texts']
        train_labels=saved['train_labels']
        test_texts=saved['test_texts']
        test_labels=saved['test_labels']
        dev_texts=saved['dev_texts']
        dev_labels=saved['dev_labels']
    else:
        print('Processing IMDB dataset')
        train_texts, train_labels = read_imdb_files(opt, 'train')
        test_texts, test_labels = read_imdb_files(opt, 'test')
        dev_texts = test_texts[12500-500:12500] + test_texts[25000-500:25000]
        dev_labels = test_labels[12500-500:12500] + test_labels[25000-500:25000]

        test_texts = test_texts[:12500] + test_texts[12500:25000]
        test_labels = test_labels[:12500] + test_labels[12500:25000]

        f=open(filename,'wb')
        saved={}
        saved['train_texts']=train_texts
        saved['train_labels']=train_labels
        saved['test_texts']=test_texts
        saved['test_labels']=test_labels
        saved['dev_texts']=dev_texts
        saved['dev_labels']=dev_labels
        pickle.dump(saved,f)
        f.close()
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels



def read_snli_files(opt, filetype):
    def label_switch(str):
        if str == "entailment":
            return [1, 0, 0]
        if str == "contradiction":
            return [0, 1, 0]
        if str == "neutral":
            return [0, 0, 1]
        raise NotImplementedError

    split = filetype
    totals = {'train': 550152, 'dev': 10000, 'test': 10000}
    all_prem = []
    all_hypo = []
    all_labels = []

    fn = os.path.join('dataset/snli_1.0/snli_1.0_{}.jsonl'.format(split))
    with open(fn) as f:
        for line in tqdm(f, total=totals[split]):
            example = json.loads(line)
            prem, hypo, gold_label = example['sentence1'], example['sentence2'], example['gold_label']
            try:
                one_hot_label = label_switch(gold_label)

                from nltk import word_tokenize
                prem = ' '.join(word_tokenize(prem))
                hypo = ' '.join(word_tokenize(hypo))

                all_prem.append(prem)
                all_hypo.append(hypo)
                all_labels.append(one_hot_label)

            except:
                continue
    return all_prem, all_hypo, all_labels

def split_snli_files(opt):
    filename = opt.split_snli_files_path
    if os.path.exists(filename):
        print('Read processed SNLI dataset')
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        train_perms=saved['train_perms']
        train_hypos=saved['train_hypos']
        train_labels=saved['train_labels']
        test_perms=saved['test_perms']
        test_hypos=saved['test_hypos']
        test_labels=saved['test_labels']
        dev_perms=saved['dev_perms']
        dev_hypos=saved['dev_hypos']
        dev_labels=saved['dev_labels']
    else:
        print('Processing SNLI dataset')
        train_perms, train_hypos, train_labels = read_snli_files(opt, 'train')
        dev_perms, dev_hypos, dev_labels = read_snli_files(opt, 'dev')
        test_perms, test_hypos, test_labels = read_snli_files(opt, 'test')
        f=open(filename,'wb')
        saved={}
        saved['train_perms']=train_perms
        saved['train_hypos']=train_hypos
        saved['train_labels']=train_labels
        saved['test_perms']=test_perms
        saved['test_hypos']=test_hypos
        saved['test_labels']=test_labels
        saved['dev_perms']=dev_perms
        saved['dev_hypos']=dev_hypos
        saved['dev_labels']=dev_labels
        pickle.dump(saved,f)
        f.close()
    return train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels


class ImdbData(torch.utils.data.Dataset):
    
    def __init__(self, opt, x, y, tokenized_subs_dict, seq_max_len, tokenizer, given_class):
        self.opt = opt
        self.max_substitution_num = 10

        self.x = x.copy()
        self.y = y.copy() #y is onehot
        self.tokenized_subs_dict = tokenized_subs_dict.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer

        if given_class is not None:
            xx=[]
            yy=[]
            for i, _ in enumerate(self.y):
                if self.y[i].argmax()==given_class:
                    xx.append(self.x[i])
                    yy.append(self.y[i])
            self.x = xx
            self.y = yy

    def transform(self, sent, label, text_subs, text_subs_mask, attention_mask):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(text_subs,dtype=torch.long), torch.tensor(text_subs_mask,dtype=torch.float), torch.tensor(attention_mask, dtype = torch.long)

    def __getitem__(self, index):
    
        encoded = self.tokenizer(self.x[index], self.seq_max_len)

        sent = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        text_subs=[]
        text_subs_mask=[]
        for token in sent:
            text_subs_mask.append([])
            text_subs.append([token])

        splited_words = self.x[index].split()

        for i in range(min(self.seq_max_len-2, len(splited_words))):
            word = splited_words[i]
            if word in self.tokenized_subs_dict:
                text_subs[i+1].extend(self.tokenized_subs_dict[word])

        label = self.y[index].argmax() # y is one hot here

        for i in range(len(sent)):
            text_subs_mask[i] = [1 for times in range(len(text_subs[i]))]
            
            while(len(text_subs[i])<self.max_substitution_num):
                text_subs[i].append(0)
                text_subs_mask[i].append(0)

        return self.transform(sent, label, text_subs, text_subs_mask, attention_mask)

    def __len__(self):
        return len(self.y)

class SnliData(torch.utils.data.Dataset):
    
    def __init__(self, opt, perm, hypo, y, tokenized_subs_dict, seq_max_len, tokenizer, given_class):
        self.opt=opt
        self.perm = perm.copy()
        self.hypo = hypo.copy()
        self.y = y.copy()
        self.tokenized_subs_dict = tokenized_subs_dict.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer
        self.max_substitution_num=10

        if given_class is not None:
            permperm=[]
            hypohypo=[]
            yy=[]
            for i, label in enumerate(self.y):
                if self.y[i].argmax()==given_class:
                    permperm.append(self.perm[i])
                    hypohypo.append(self.hypo[i])
                    yy.append(self.y[i])
            self.perm = permperm
            self.hypo = hypohypo
            self.y = yy

    def transform(self, sent, label, text_subs, text_subs_mask, attention_mask):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(text_subs,dtype=torch.long), torch.tensor(text_subs_mask,dtype=torch.float), torch.tensor(attention_mask, dtype = torch.long)

    def __getitem__(self, index):

        if self.opt.plm_type=="bert":
            input_text = self.perm[index]+" [SEP] "+self.hypo[index]
        elif self.opt.plm_type=="roberta":
            input_text = self.perm[index]+"</s>"+self.hypo[index]
 
        encoded = self.tokenizer(input_text, self.seq_max_len)

        sent = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        text_subs=[]
        text_subs_mask=[]
        for token in sent:
            text_subs_mask.append([])
            text_subs.append([token])

        splited_words = self.hypo[index].split()

        h_start = len(self.perm[index].split())+2
        len_h = len(self.hypo[index].split())

        for i in range(h_start, min(self.seq_max_len-1, h_start+len_h), 1):
            word = splited_words[i-h_start]
            if word in self.tokenized_subs_dict:
                text_subs[i].extend(self.tokenized_subs_dict[word])

        label = self.y[index].argmax()

        for i, x in enumerate(sent):
            text_subs_mask[i] = [1 for times in range(len(text_subs[i]))]
            
            while(len(text_subs[i])<self.max_substitution_num):
                text_subs[i].append(0)
                text_subs_mask[i].append(0)


        return self.transform(sent, label, text_subs, text_subs_mask, attention_mask)

    def __len__(self):
        return len(self.y)

def make_given_y_iter_imdb(opt, tokenized_subs_dict, tokenizer):

    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)

    seq_max_len = opt.imdb_input_max_len

    train_data_y0 = ImdbData(opt, train_texts, np.array(train_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt.batch_size//opt.label_size, shuffle=True, num_workers=16)

    train_data_y1 = ImdbData(opt, train_texts, np.array(train_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt.batch_size//opt.label_size, shuffle=True, num_workers=16)

    ## lm constraint
    #from attack.attack_surface import LMConstrainedAttackSurface
    #attack_surface = LMConstrainedAttackSurface.from_files(opt.substitution_dict_path, opt.imdb_lm_file_path)
    #filtered_test_texts=[]
    #filtered_test_labels=[]
    #for i, text in enumerate(test_texts):
    #    if attack_surface.check_in(text.split(' ')):
    #        filtered_test_texts.append(text)
    #        filtered_test_labels.append(test_texts[i])

    test_data = ImdbData(opt, test_texts, np.array(test_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=None)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=16)

    return train_loader_y0, train_loader_y1, test_loader


def make_given_y_iter_snli(opt, tokenized_subs_dict, tokenizer):

    train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)

    seq_max_len = opt.snli_input_max_len

    train_data_y0 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt.batch_size//opt.label_size, shuffle=True, num_workers=16)

    train_data_y1 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt.batch_size//opt.label_size, shuffle=True, num_workers=16)

    train_data_y2 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=2)
    train_loader_y2 = torch.utils.data.DataLoader(train_data_y2, opt.batch_size//opt.label_size, shuffle=True, num_workers=16)

    #from attack.attack_surface import LMConstrainedAttackSurface
    #attack_surface = LMConstrainedAttackSurface.from_files(opt.substitution_dict_path, opt.snli_lm_file_path)

    test_data = SnliData(opt, test_perms, test_hypos, np.array(test_labels), tokenized_subs_dict, seq_max_len, tokenizer, given_class=None)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=16)

    return train_loader_y0, train_loader_y1, train_loader_y2, test_loader

def get_substitution_dict(file_path):
    import json
    with open(file_path) as f:
        subs_dict = json.load(f)
    return subs_dict

def get_data_iters(opt):

    tokenizer, substitution_tokenizer = get_tokenizers(opt.plm_type)

    if opt.plm_type == 'bert':
        tokenized_subs_dict_path = opt.bert_tokenized_subs_dict_path
    elif opt.plm_type == 'roberta':
        tokenized_subs_dict_path = opt.roberta_tokenized_subs_dict_path
    else:
        raise NotImplementedError

    if os.path.exists(tokenized_subs_dict_path):
        f=open(tokenized_subs_dict_path,'rb')
        saved=pickle.load(f)
        f.close()
        tokenized_subs_dict = saved["tokenized_subs_dict"]
    else:

        subs_dict = get_substitution_dict(opt.substitution_dict_path)
        tokenized_subs_dict = {} # key is text word, contents are tokenized substitution words

        # Tokenize syn data
        print("tokenizing substitution words")
        for key in subs_dict:
            if len(subs_dict[key])!=0:
                #temp = tokenizer.encode_plus(subs_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']
                temp = substitution_tokenizer(subs_dict[key])['input_ids']
                temp = [x[0] for x in temp]
                tokenized_subs_dict[key] = temp
        
        print("done")

        filename= tokenized_subs_dict_path
        f=open(filename,'wb')
        saved={}
        saved['tokenized_subs_dict']=tokenized_subs_dict
        pickle.dump(saved,f)
        f.close()

    if opt.dataset == "imdb":
        opt.label_size = 2
        train_iter_y0, train_iter_y1, test_iter = make_given_y_iter_imdb(opt, tokenized_subs_dict, tokenizer)
        return (train_iter_y0, train_iter_y1), test_iter

    elif opt.dataset == "snli":
        opt.label_size = 3
        train_iter_y0, train_iter_y1, train_iter_y2, test_iter = make_given_y_iter_snli(opt, tokenized_subs_dict, tokenizer)
        return(train_iter_y0, train_iter_y1, train_iter_y2), test_iter

    else:
        raise NotImplementedError


if __name__ =="__main__":
    import opts
