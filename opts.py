import argparse, os, re
import configparser
import sys

def parse_opt():

    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default="imdb", help='dataset') # snli
    parser.add_argument('--torch_seed', type=float, default=0, help='')
    parser.add_argument('--rand_seed', type=float, default=0, help='')
    parser.add_argument('--substitution_dict_path', type=str, default="attack/counterfitted_neighbors.json", help='') 

    parser.add_argument('--resume', type=str, default=None, help='') 

    parser.add_argument('--weight_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_kl', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_clean', type=float, default=0, help='') 
    parser.add_argument('--weight_mi_giveny_adv', type=float, default=0, help='') 
    parser.add_argument('--weight_params_l2', type=float, default=0, help='') 
    parser.add_argument('--infonce_sim_metric', type=str, default='normal', help='') 

    parser.add_argument('--freeze_plm', type=str, default='false', help='') 
    parser.add_argument('--freeze_plm_teacher', type=str, default='true', help='')

    # ascc attack parameters
    parser.add_argument('--ascc_test_attack_iters', type=int, default=10, help='') 
    parser.add_argument('--ascc_test_attack_sparse_weight', type=float, default=15, help='') 
    parser.add_argument('--ascc_train_attack_iters', type=int, default=10, help='')             
    parser.add_argument('--ascc_train_attack_sparse_weight', type=float, default=15, help='') 
    parser.add_argument('--ascc_w_optm_lr', type=float, default=1, help='') 

    # genetic attack and pwws attack
    parser.add_argument('--pwws_test_num', type=int, default=300, help='') 
    parser.add_argument('--genetic_test_num', type=int, default=300, help='') 
    parser.add_argument('--genetic_iters', type=int, default=40, help='') 
    parser.add_argument('--genetic_pop_size', type=int, default=60, help='') 

    parser.add_argument('--lm_constraint_on_genetic_attack', type=str, default='true', help='')
    parser.add_argument('--imdb_lm_file_path', type=str, default="attack/lm_scores/imdb_all.txt", help='') 
    parser.add_argument('--snli_lm_file_path', type=str, default="attack/lm_scores/snli_all_save.txt", help='') 

    # data related
    parser.add_argument('--bert_tokenized_subs_dict_path', type=str, default="processed_data/bert.tokenized.dict.pkl", help='')
    parser.add_argument('--roberta_tokenized_subs_dict_path', type=str, default="processed_data/roberta.tokenized.dict.pkl", help='')

    parser.add_argument('--split_imdb_files_path', type=str, default="processed_data/split_imdb_files.pkl", help='')
    parser.add_argument('--split_snli_files_path', type=str, default="processed_data/split_snli_files.pkl", help='')
     
    parser.add_argument('--out_path', type=str, default="./", help='')          
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')    

    parser.add_argument('--imdb_input_max_len', type=int, default=300, help='imdb_input_max_len')   
    parser.add_argument('--snli_input_max_len', type=int, default=80, help='snli_input_max_len')   

    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size') # 32 for imdb 120 for snli
    parser.add_argument('--test_batch_size', type=int, default=32, help='test_batch_size') 
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight_decay')
    parser.add_argument('--optimizer', type=str, default="adamw", help='optimizer')
    parser.add_argument('--training_epochs', type=int, default=20, help='')

    # model
    parser.add_argument('--model', type=str, default="adv_plm", help='model name')
    parser.add_argument('--plm_type', type=str, default="bert", help='') # e.g., bert, roberta, xlnet
    parser.add_argument('--mixout_p', type=float, default=0, help='')

    # Mutual information related 
    parser.add_argument('--infonce_temperature', type=float, default=0.2, help='')

    args = parser.parse_args()
    
    # process the type for bool and list    
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg])==str:
            if args.__dict__[arg].lower()=="true":
                args.__dict__[arg]=True
            elif args.__dict__[arg].lower()=="false":
                args.__dict__[arg]=False
            elif "," in args.__dict__[arg]:
                args.__dict__[arg]= [int(i) for i in args.__dict__[arg].split(",") if i!='']
            else:
                pass

    return args 