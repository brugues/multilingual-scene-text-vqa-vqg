#!/bin/zsh
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to wh

python train.py --language zh --embedding_type fasttext --fasttext_subtype cc --output_folder estvqa_cc_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type fasttext --fasttext_subtype wiki --output_folder estvqa_wiki_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned_pair zh-en --fasttext_aligned --output_folder estvqa_aligned_cc_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type fasttext --fasttext_subtype wiki --fasttext_aligned_pair zh-en --fasttext_aligned --output_folder estvqa_aligned_wiki_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder estvqa_bpemb_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder estvqa_multi_bpemb_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
python train.py --language zh --embedding_type smith --output_folder estvqa_smith_zh --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json
