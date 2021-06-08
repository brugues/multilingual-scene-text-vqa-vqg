#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 16384 # 16GB.
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/outputs/slurm
#SBATCH -p mhigh,mlow # or mlow Partition to submit to master low prioriy queue
#SBATCH -q masterlow
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

python train.py
