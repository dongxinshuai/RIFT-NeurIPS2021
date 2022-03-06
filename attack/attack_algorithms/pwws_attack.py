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
from functools import partial
import attr
import nltk
import spacy
try:
    import cPickle as pickle
except ImportError:
    import pickle

from modified_tokenizers import get_tokenizers
from data import split_imdb_files, split_snli_files, get_substitution_dict

@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()

def pwws_attack(opt, model, dataset, test_num):
    print('pwws_test_num:', test_num)

    tokenizer, _ = get_tokenizers(opt.plm_type)
    syn_dict = get_substitution_dict(opt.substitution_dict_path)

    # process data
    if dataset == 'imdb':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        assert(test_num<len(test_labels))
        input_max_len = opt.imdb_input_max_len

        # randomly select test examples
        indexes = [i for i in range(len(test_labels))]
        random.seed(opt.rand_seed)
        random.shuffle(indexes)
        test_texts = [test_texts[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]

        test_texts = test_texts[:test_num]
        test_labels = test_labels[:test_num]

        grad_guide = WrappedModelClsForPWWS(dataset, input_max_len, model, tokenizer)

    elif dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
        assert(test_num<len(test_labels))
        input_max_len = opt.snli_input_max_len

        # randomly select test examples
        indexes = [i for i in range(len(test_labels))]
        random.seed(opt.rand_seed)
        random.shuffle(indexes)
        test_perms = [test_perms[i] for i in indexes]
        test_hypos = [test_hypos[i] for i in indexes]
        test_labels = [test_labels[i] for i in indexes]

        test_perms = test_perms[:test_num]
        test_hypos = test_hypos[:test_num]
        test_labels = test_labels[:test_num]

        grad_guide = WrappedModelNliForPWWS(dataset, input_max_len, model, tokenizer, opt.plm_type)
    else:
        raise NotImplementedError


    model.plm.eval()
    model.cls_to_logit.eval()

    if dataset == 'imdb':
        ori_prediction = [grad_guide.predict_class(text) for text in test_texts]
        #print("clean acc:", sum([ori_prediction[i]==np.argmax(test_labels[i]) for i in range(test_num)])/test_num)
        failed_attack_num = 0
        t_start = time.clock()
        tested=0

        for index, text in enumerate(test_texts):
            
            #print('__PWWS attack data point {}__.'.format(index))

            if np.argmax(test_labels[index]) == ori_prediction[index]:
                #print('ori prediction is correct')
                # If the ground_true label is the same as the predicted label
                adv_text, adv_prediction, sub_rate, NE_rate = pwws_attack_cls(opt, text, np.argmax(test_labels[index]), grad_guide, input_max_len, syn_dict)
                if adv_prediction == np.argmax(test_labels[index]):
                    failed_attack_num += 1
                    #print('{}. Failure.'.format(index))
                else:
                    #print('{}. Successful example crafted.'.format(index))
                    pass

            else:
                #print('ori prediction is wrong')
                pass

            tested+=1
            print('acc under pwws {}/{}, time cost: {}'.format(failed_attack_num, tested, time.clock() - t_start))

        accuracy = 1.0*failed_attack_num/tested

    elif dataset == 'snli':
        ori_prediction = [grad_guide.predict_class(text_p, text_h) for text_p, text_h in zip(test_perms, test_hypos)]
        #print("clean acc:", sum([ori_prediction[i]==np.argmax(test_labels[i]) for i in range(test_num)])/test_num)
        failed_attack_num = 0
        t_start = time.clock()
        tested=0

        for index, (text_p, text_h) in enumerate(zip(test_perms, test_hypos)):
            
            #print('__PWWS attack data point {}__.'.format(index))

            if np.argmax(test_labels[index]) == ori_prediction[index]:
                #print('ori prediction is correct')
                # If the ground_true label is the same as the predicted label
                adv_text_p, adv_text_h, adv_prediction, sub_rate, NE_rate = pwws_attack_nli(opt, text_p, text_h, np.argmax(test_labels[index]), grad_guide, input_max_len, syn_dict)
                if adv_prediction == np.argmax(test_labels[index]):
                    failed_attack_num += 1
                    #print('{}. Failure.'.format(index))
                else:
                    #print('{}. Successful example crafted.'.format(index))
                    pass

            else:
                #print('ori prediction is wrong')
                pass

            tested+=1 
            print('acc under pwws {}/{}, time cost: {}'.format(failed_attack_num, tested, time.clock() - t_start))

        accuracy = 1.0*failed_attack_num/tested

    else:
        raise NotImplementedError
    
    #print("accuracy under pwws attack: ", accuracy)
    return accuracy


class WrappedModelClsForPWWS:
    def __init__(self, dataset, input_max_len, model, tokenizer):

        self.dataset = dataset
        self.model=model
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.device = model.device

    def predict_prob(self, text):

        token = self.tokenizer(text, self.input_max_len)

        text_x = np.array([token['input_ids']])
        text_x = torch.tensor(text_x,dtype=torch.long).to(self.device)
        attention_mask = np.array([token['attention_mask']])
        attention_mask = torch.tensor(attention_mask,dtype=torch.long).to(self.device)

        logit = self.model(text_x, attention_mask).squeeze(0)

        return F.softmax(logit).detach().cpu().numpy()

    def predict_class(self, text):
        prediction = self.predict_prob(text)
        classes = np.argmax(prediction, axis=-1)
        return classes

class WrappedModelNliForPWWS:
    def __init__(self, dataset, input_max_len, model, tokenizer, plm_type):

        self.dataset = dataset
        self.model=model
        self.tokenizer = tokenizer
        self.input_max_len = input_max_len
        self.plm_type=plm_type
        self.device = model.device

    def predict_prob(self, x_p, x_h):

        if self.plm_type=="bert":
            token = self.tokenizer(x_p+" [SEP] "+x_h, self.input_max_len)
        elif self.plm_type=="roberta":
            token = self.tokenizer(x_p+"</s>"+x_h, self.input_max_len)
        else:
            raise NotImplementedError

        text_x = np.array([token['input_ids']])
        text_x = torch.tensor(text_x,dtype=torch.long).to(self.device)
        attention_mask = np.array([token['attention_mask']])
        attention_mask = torch.tensor(attention_mask,dtype=torch.long).to(self.device)

        logit = self.model(text_x, attention_mask).squeeze(0)

        return F.softmax(logit).detach().cpu().numpy()

    def predict_class(self, x_p, x_h):
        prediction = self.predict_prob(x_p, x_h)
        classes = np.argmax(prediction, axis=-1)
        return classes


def pwws_attack_cls(opt, input_text, true_y, grad_guide, input_max_len, syn_dict):

    def halt_condition_fn(perturbed_text):
        adv_y = grad_guide.predict_class(perturbed_text)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        candidate.candidate_word
        text_splited = text.split()
        perturbed_text_splited = text_splited
        perturbed_text_splited[candidate.token_position]=candidate.candidate_word
        perturbed_text = " ".join(perturbed_text_splited) 

        origin_prob = grad_guide.predict_prob(text)
        perturbed_prob = grad_guide.predict_prob(perturbed_text)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    def evaluate_word_saliency_cls(input_text, grad_guide, input_y):
        word_saliency_list = []
        input_text_splited = input_text.split(" ")

        origin_prob = grad_guide.predict_prob(input_text)
        for position in range(len(input_text_splited)):
            if position >= input_max_len:
                break
            temp = input_text_splited[position]
            input_text_splited[position] = "NULL"
            without_word_text = ' '.join(input_text_splited)
            input_text_splited[position] = temp

            prob_without_word = grad_guide.predict_prob(without_word_text)

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, input_text_splited[position], word_saliency, None))

        return word_saliency_list

    def PWWS_cls(opt, input_text, true_y, word_saliency_list=None, heuristic_fn=None, halt_condition_fn=None):

        def _generate_synonym_candidates_from_dict(opt, word, token_position):
            candidates = []
            if not word in syn_dict:
                return candidates
            else:
                for candidate_word in syn_dict[word]:
                    candidate = SubstitutionCandidate(
                        token_position=token_position,
                        similarity_rank=None,
                        original_token=word,
                        candidate_word=candidate_word)
                    candidates.append(candidate)
                return candidates

        # defined in Eq.(8)
        def softmax(x):
            exp_x = np.exp(x)
            softmax_x = exp_x / np.sum(exp_x)
            return softmax_x

        substitute_count = 0  # calculate how many substitutions used in a doc
        substitute_tuple_list = []  # save the information of substitute word

        word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
        word_saliency_array = softmax(word_saliency_array)

        syn_dict = get_substitution_dict(opt.substitution_dict_path)

        # for each word w_i in x, use WordNet to build a synonym set L_i
        for (position, word, word_saliency, tag) in word_saliency_list:
            if position >= input_max_len:
                break

            candidates = []
            candidates = _generate_synonym_candidates_from_dict(opt, word, position)

            if len(candidates) == 0:
                continue

            # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
            sorted_candidates = zip(map(partial(heuristic_fn, input_text), candidates), candidates)
            # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
            sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

            # delta_p_star is defined in Eq.(5); substitute is w_i^*
            delta_p_star, substitute = sorted_candidates.pop()

            # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
            substitute_tuple_list.append(
                (position, word, substitute, delta_p_star * word_saliency_array[position], None))

        # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
        sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

        # replace w_i in x^(i-1) with w_i^* to craft x^(i)
        NE_count = 0  # calculate how many NE used in a doc
        change_tuple_list = []

        perturbed_text =  input_text
        for (position, word, substitute, score, tag) in sorted_substitute_tuple_list:

            perturbed_text = perturbed_text.split(" ")
            perturbed_text[substitute.token_position]=substitute.candidate_word
            perturbed_text = " ".join(perturbed_text)
            
            substitute_count += 1
            if halt_condition_fn(perturbed_text):
                #print("use", substitute_count, "substitution; use", NE_count, 'NE')
                sub_rate = substitute_count / len(perturbed_text.split())
                NE_rate = NE_count / substitute_count
                return perturbed_text, sub_rate, NE_rate, change_tuple_list

        #print("use", substitute_count, "substitution; use", NE_count, 'NE')
        sub_rate = substitute_count / len(perturbed_text.split())
        NE_rate = NE_count / (substitute_count+1)
        return perturbed_text, sub_rate, NE_rate, change_tuple_list

    # PWWS
    word_saliency_list = evaluate_word_saliency_cls(input_text, grad_guide, true_y)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS_cls(opt, input_text, true_y,
                                                                word_saliency_list=word_saliency_list,
                                                                heuristic_fn=heuristic_fn,
                                                                halt_condition_fn=halt_condition_fn)

    perturbed_y = grad_guide.predict_class(perturbed_text)
    origin_prob = grad_guide.predict_prob(input_text)
    perturbed_prob = grad_guide.predict_prob(perturbed_text)
    raw_score = origin_prob[true_y] - perturbed_prob[true_y]
    #print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y], '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, sub_rate, NE_rate


def pwws_attack_nli(opt, input_text_p, input_text_h, true_y, grad_guide, input_max_len, syn_dict):

    def halt_condition_fn(input_text_p, perturbed_text_h):
        adv_y = grad_guide.predict_class(input_text_p, perturbed_text_h)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text_p, text_h, candidate_h):

        candidate_h.candidate_word
        text_h_splited = text_h.split()
        perturbed_text_h_splited = text_h_splited
        perturbed_text_h_splited[candidate_h.token_position]=candidate_h.candidate_word
        perturbed_text_h = " ".join(perturbed_text_h_splited) 

        origin_prob = grad_guide.predict_prob(text_p, text_h)
        perturbed_prob = grad_guide.predict_prob(text_p, perturbed_text_h)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    def evaluate_word_saliency_nli(input_text_p, input_text_h, grad_guide, input_y):
        word_saliency_list = []
        input_text_h_splited = input_text_h.split(" ")

        origin_prob = grad_guide.predict_prob(input_text_p, input_text_h)
        for position in range(len(input_text_h_splited)):
            if position >= input_max_len:
                break

            temp = input_text_h_splited[position]
            input_text_h_splited[position] = "NULL"
            without_word_text_h = ' '.join(input_text_h_splited)
            input_text_h_splited[position] = temp

            prob_without_word = grad_guide.predict_prob(input_text_p, without_word_text_h)

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, input_text_h_splited[position], word_saliency, None))

        return word_saliency_list

    def PWWS_nli(opt, input_text_p, input_text_h, true_y, word_saliency_list_h=None, heuristic_fn=None, halt_condition_fn=None):

        def _generate_synonym_candidates_from_dict(opt, word, token_position):
            candidates = []
            if not word in syn_dict:
                return candidates
            else:
                for candidate_word in syn_dict[word]:
                    candidate = SubstitutionCandidate(
                        token_position=token_position,
                        similarity_rank=None,
                        original_token=word,
                        candidate_word=candidate_word)
                    candidates.append(candidate)
                return candidates

        # defined in Eq.(8)
        def softmax(x):
            exp_x = np.exp(x)
            softmax_x = exp_x / np.sum(exp_x)
            return softmax_x

        substitute_count = 0  # calculate how many substitutions used in a doc
        substitute_tuple_list = []  # save the information of substitute word

        word_saliency_array_h = np.array([word_tuple[2] for word_tuple in word_saliency_list_h])
        word_saliency_array_h = softmax(word_saliency_array_h)

        # for each word w_i in x, use WordNet to build a synonym set L_i
        for (position, word, word_saliency, tag) in word_saliency_list_h:
            if position >= input_max_len:
                break

            candidates = []
            candidates = _generate_synonym_candidates_from_dict(opt, word, position)

            if len(candidates) == 0:
                continue

            # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
            sorted_candidates = zip(map(partial(heuristic_fn, input_text_p, input_text_h), candidates), candidates)
            # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
            sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

            # delta_p_star is defined in Eq.(5); substitute is w_i^*
            delta_p_star, substitute = sorted_candidates.pop()

            # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
            substitute_tuple_list.append(
                (position, word, substitute, delta_p_star * word_saliency_array_h[position], None))

        # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
        sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

        # replace w_i in x^(i-1) with w_i^* to craft x^(i)
        NE_count = 0  # calculate how many NE used in a doc
        change_tuple_list = []

        perturbed_text_h =  input_text_h

        for (position, word, substitute, score, tag) in sorted_substitute_tuple_list:
            perturbed_text_h = perturbed_text_h.split(" ")
            perturbed_text_h[substitute.token_position]=substitute.candidate_word
            perturbed_text_h = " ".join(perturbed_text_h)
            
            substitute_count += 1
            if halt_condition_fn(input_text_p, perturbed_text_h):
                #print("use", substitute_count, "substitution; use", NE_count, 'NE')
                sub_rate = substitute_count / len(perturbed_text_h.split())
                NE_rate = NE_count / substitute_count
                return input_text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list


        #print("use", substitute_count, "substitution; use", NE_count, 'NE')
        sub_rate = substitute_count / len(perturbed_text_h.split())
        NE_rate = NE_count / (substitute_count+1)
        return input_text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list

    # PWWS
    word_saliency_list_h = evaluate_word_saliency_nli(input_text_p, input_text_h, grad_guide, true_y)
    text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list = PWWS_nli(opt, input_text_p, input_text_h, true_y,
                                                                                word_saliency_list_h=word_saliency_list_h,
                                                                                heuristic_fn=heuristic_fn,
                                                                                halt_condition_fn=halt_condition_fn)

    perturbed_y = grad_guide.predict_class(text_p, perturbed_text_h)

    origin_prob = grad_guide.predict_prob(text_p, input_text_h)
    perturbed_prob = grad_guide.predict_prob(text_p, perturbed_text_h)
    raw_score = origin_prob[true_y] - perturbed_prob[true_y]
    #print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y], '. Prob shift: ', raw_score)
    return text_p, perturbed_text_h, perturbed_y, sub_rate, NE_rate