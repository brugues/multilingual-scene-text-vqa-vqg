#!/bin/sh
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 24576
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to wh

python train.py --embedding_type fasttext --fasttext_subtype cc --language en-ca --fasttext_aligned --fasttext_aligned_pair en-ca --output_folder cc_combined_en_ca
python train.py --embedding_type fasttext --fasttext_subtype cc --language en-es --fasttext_aligned --fasttext_aligned_pair en-es --output_folder cc_combined_en_es
python train.py --embedding_type fasttext --fasttext_subtype cc --language en-zh --fasttext_aligned --fasttext_aligned_pair en-zh --output_folder cc_combined_en_zh

python train.py --embedding_type smith --language en-ca --output_folder smith_combined_en_ca
python train.py --embedding_type smith --language en-es --output_folder smith_combined_en_es
python train.py --embedding_type smith --language en-zh --output_folder smith_combined_en_zh

