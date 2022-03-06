
ulimit -n 10000

out_path="log_imdb_bert/kl10_advmigiveny0.1"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=0,1
nohup python -u rift_train.py --weight_clean 1 --weight_kl 10 --weight_mi_giveny_adv 0.1 --batch_size 32  --plm_type bert --dataset imdb  --out_path $out_path > $out_path/train.log 2>&1 &

out_path="log_snli_bert/kl5_advmigiveny0.7"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=2,3
nohup python -u rift_train.py --weight_clean 1 --weight_kl 5 --weight_mi_giveny_adv 0.7 --batch_size 120  --plm_type bert --dataset snli  --out_path $out_path > $out_path/train.log 2>&1 &
