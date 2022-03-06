Pytorch implementation of NeurIPS 2021 paper, "How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?"(https://arxiv.org/pdf/2112.11668.pdf). (Our code is partly based on https://github.com/robinjia/certified-word-sub and https://github.com/JHL-HUST/PWWS.)

# Environment

conda env create --name rift --file environment.yml
python -m spacy download en
mkdir dataset
mkdir processed_data

# Data

IMDB cd dataset wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz tar -xvzf aclImdb_v1.tar.gz

SNLI cd dataset wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip unzip snli_1.0.zip 

# Run

source rift_train.sh
