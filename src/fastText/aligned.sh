#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#

python train.py --language es --fasttext_subtype cc --fasttext_aligned_pair es-en --fasttext_aligned --output_folder aligned_cc_es
python train.py --language ca --fasttext_subtype cc --fasttext_aligned_pair ca-en --fasttext_aligned --output_folder aligned_cc_ca
python train.py --language es --fasttext_subtype wiki --fasttext_aligned_pair es-en --fasttext_aligned --output_folder aligned_wiki_es
python train.py --language ca --fasttext_subtype wiki --fasttext_aligned_pair ca-en --fasttext_aligned --output_folder aligned_wiki_ca

