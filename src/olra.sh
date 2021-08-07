#!/bin/zsh
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to wh

# TRAIN
# cc
python train_olra.py --n_epochs 15 --use_gru --lr 0.001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_gru_0.001_no_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.001 --fasttext_subtype wiki --output_folder olra_wiki_gru_0.001_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.0001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_gru_0.0001_no_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.0001 --fasttext_subtype wiki --output_folder olra_wiki_gru_0.0001_decay
python train_olra.py --n_epochs 15 --lr 0.001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.001_no_decay
python train_olra.py --n_epochs 15 --lr 0.001 --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.001_decay
python train_olra.py --n_epochs 15 --lr 0.0001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.0001_no_decay
python train_olra.py --n_epochs 15 --lr 0.0001 --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.0001_decay

# wiki
python train_olra.py --n_epochs 15 --use_gru --lr 0.001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_gru_0.001_no_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.001 --fasttext_subtype wiki --output_folder olra_wiki_gru_0.001_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.0001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_gru_0.0001_no_decay
python train_olra.py --n_epochs 15 --use_gru --lr 0.0001 --fasttext_subtype wiki --output_folder olra_wiki_gru_0.0001_decay
python train_olra.py --n_epochs 15 --lr 0.001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.001_no_decay
python train_olra.py --n_epochs 15 --lr 0.001 --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.001_decay
python train_olra.py --n_epochs 15 --lr 0.0001 --no_apply_decay --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.0001_no_decay
python train_olra.py --n_epochs 15 --lr 0.0001 --fasttext_subtype wiki --output_folder olra_wiki_lstm_0.0001_decay

