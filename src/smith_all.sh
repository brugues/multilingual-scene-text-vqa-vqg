#!/bin/zsh
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to wh
python train.py --language en --embedding_type smith --output_folder smith_en
#python train.py --language ca --embedding_type smith --bpemb_subtype multi --output_folder multi_bpemb_ca
#python train.py --language es --embedding_type smith --bpemb_subtype multi --output_folder multi_bpemb_es