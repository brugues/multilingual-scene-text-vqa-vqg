#!/bin/zsh
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to wh

python train_olra.py --use_gru --n_epochs 15 --fasttext_subtype cc --output_folder olra_b15_no_att
python train_olra.py --use_gru --n_epochs 30 --fasttext_subtype cc --output_folder olra_b30_no_att

python train_olra.py --use_gru --n_epochs 15 --fasttext_subtype cc --multimodal_attention --output_folder olra_b15_att
python train_olra.py --use_gru --n_epochs 30 --fasttext_subtype cc --multimodal_attention --output_folder olra_b30_att

python train_olra.py --use_gru --n_epochs 15 --fasttext_subtype cc --multimodal_attention --lr 0.00005 --output_folder olra_b15_att_lrhalf
python train_olra.py --use_gru --n_epochs 30 --fasttext_subtype cc --multimodal_attention --lr 0.00005 --output_folder olra_b30_att_lrhalf

#python train_olra.py --n_epochs 15 --fasttext_subtype cc --multimodal_attention --lr 0.0002 --output_folder olra_b15_att_lrdouble
#python train_olra.py --n_epochs 30 --fasttext_subtype cc --multimodal_attention --lr 0.0002 --output_folder olra_b30_att_lrdouble
#
#python train_olra.py --n_epochs 15 --fasttext_subtype cc --output_folder olra_b15_no_att_nodecay
#python train_olra.py --n_epochs 15 --fasttext_subtype cc --multimodal_attention --output_folder olra_b30_att_nodecay
