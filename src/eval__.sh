#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

# ALIGNED.SH
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --fasttext_aligned_pair en-ca --fasttext_aligned --output_folder en_aligned_cc_en_ca --model_to_evaluate ./outputs/models/aligned_cc_en_ca
#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder ca_aligned_cc_en_ca --model_to_evaluate ./outputs/models/aligned_cc_en_ca
#
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --fasttext_aligned_pair en-es --fasttext_aligned --output_folder en_aligned_cc_en_es --model_to_evaluate ./outputs/models/aligned_cc_en_es
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder es_aligned_cc_en_es --model_to_evaluate ./outputs/models/aligned_cc_en_es
#
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --fasttext_aligned_pair en-zh --fasttext_aligned --output_folder en_aligned_cc_en_zh --model_to_evaluate ./outputs/models/aligned_cc_en_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder zh_aligned_cc_en_zh --model_to_evaluate ./outputs/models/aligned_cc_en_zh


# ZH_ALL.SH
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder zh_bpemb_en --model_to_evaluate ./outputs/models/bpemb_en
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder zh_bpemb_ca --model_to_evaluate ./outputs/models/bpemb_ca
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder zh_bpemb_es --model_to_evaluate ./outputs/models/bpemb_es
#
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype wiki --output_folder en_bpemb_zh --model_to_evaluate ./outputs/models/bpemb_zh
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype wiki --output_folder ca_bpemb_zh --model_to_evaluate ./outputs/models/bpemb_zh
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype wiki --output_folder es_bpemb_zh --model_to_evaluate ./outputs/models/bpemb_zh
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype wiki --output_folder zh_bpemb_zh --model_to_evaluate ./outputs/models/bpemb_zh
#
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --output_folder en_multi_bpemb_en --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --output_folder ca_multi_bpemb_en --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --output_folder es_multi_bpemb_en --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder zh_multi_bpemb_en --model_to_evaluate ./outputs/models/multi_bpemb_en
#
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --output_folder en_multi_bpemb_ca --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --output_folder ca_multi_bpemb_ca --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --output_folder es_multi_bpemb_ca --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder zh_multi_bpemb_ca --model_to_evaluate ./outputs/models/multi_bpemb_ca
#
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --output_folder en_multi_bpemb_es --model_to_evaluate ./outputs/models/multi_bpemb_es
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --output_folder ca_multi_bpemb_es --model_to_evaluate ./outputs/models/multi_bpemb_es
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --output_folder es_multi_bpemb_es --model_to_evaluate ./outputs/models/multi_bpemb_es
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder zh_multi_bpemb_es --model_to_evaluate ./outputs/models/multi_bpemb_es
#
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --output_folder en_multi_bpemb_zh --model_to_evaluate ./outputs/models/multi_bpemb_zh
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --output_folder ca_multi_bpemb_zh --model_to_evaluate ./outputs/models/multi_bpemb_zh
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --output_folder es_multi_bpemb_zh --model_to_evaluate ./outputs/models/multi_bpemb_zh
#python eval.py --batch_size 55 --language zh --embedding_type bpemb --bpemb_subtype multi --output_folder zh_multi_bpemb_zh --model_to_evaluate ./outputs/models/multi_bpemb_zh



# smiths
#python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder zh_smith_en --model_to_evaluate ./outputs/models/smith_en
#
#python eval.py --batch_size 55 --language en --embedding_type smith --output_folder en_smith_ca --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder ca_smith_ca --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language es --embedding_type smith --output_folder es_smith_ca --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder zh_smith_ca --model_to_evaluate ./outputs/models/smith_ca
#
#python eval.py --batch_size 55 --language en --embedding_type smith --output_folder en_smith_es --model_to_evaluate ./outputs/models/smith_es
#python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder ca_smith_es --model_to_evaluate ./outputs/models/smith_es
#python eval.py --batch_size 55 --language es --embedding_type smith --output_folder es_smith_es --model_to_evaluate ./outputs/models/smith_es
#python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder zh_smith_es --model_to_evaluate ./outputs/models/smith_es
#
#python eval.py --batch_size 55 --language en --embedding_type smith --output_folder en_smith_zh --model_to_evaluate ./outputs/models/smith_zh
#python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder ca_smith_zh --model_to_evaluate ./outputs/models/smith_zh
#python eval.py --batch_size 55 --language es --embedding_type smith --output_folder es_smith_zh --model_to_evaluate ./outputs/models/smith_zh
#python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder zh_smith_zh --model_to_evaluate ./outputs/models/smith_zh


# ESTVQA.SH
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --fasttext_subtype cc --output_folder zh_estvqa_cc_zh --model_to_evaluate ./outputs/models/estvqa_cc_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --fasttext_subtype wiki --output_folder zh_estvqa_wiki_zh --model_to_evaluate ./outputs/models/estvqa_wiki_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --embedding_type smith --output_folder zh_estvqa_smith_zh --model_to_evaluate ./outputs/models/estvqa_smith_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder zh_estvqa_aligned_cc_zh --model_to_evaluate ./outputs/models/estvqa_aligned_cc_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder zh_estvqa_aligned_wiki_zh --model_to_evaluate ./outputs/models/estvqa_aligned_wiki_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --embedding_type bpemb --output_folder zh_estvqa_bpemb_zh --model_to_evaluate ./outputs/models/estvqa_bpemb_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json
#python eval.py --no_server_evaluation --batch_size 27 --language zh --dataset estvqa --embedding_type bpemb --bpemb_subtype multi --output_folder zh_estvqa_multi_bpemb_zh --model_to_evaluate ./outputs/models/estvqa_multi_bpemb_zh --image_path data/EST-VQA-v1.0 --gt_eval_file data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json

python eval.py --no_server_evaluation --batch_size 55 --language en --fasttext_subtype cc --output_folder test --model_to_evaluate ./outputs/models/cc_en
