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

python train.py --language zh --fasttext_subtype cc --output_folder cc_zh
python train.py --language zh --fasttext_subtype wiki --output_folder wiki_zh
python train.py --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder bpemb_zh
python train.py --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder multi_bpemb_zh
python train.py --language zh --fasttext_subtype cc --fasttext_aligned_pair zh-en --fasttext_aligned --output_folder aligned_cc_zh
python train.py --language zh --fasttext_subtype wiki --fasttext_aligned_pair zh-en --fasttext_aligned --output_folder aligned_wiki_zh
