#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

python eval.py --batch_size 1 --language ca --fasttext_subtype cc --output_folder cc_ca_custom --model_to_evaluate ./outputs/models/cc_ca --image_path data/ --gt_eval_file data/vqa-catala/custom_catala.json --no_server_evaluation --max_len 40
python eval.py --batch_size 1 --language ca --fasttext_subtype wiki --output_folder wiki_ca_custom --model_to_evaluate ./outputs/models/wiki_ca --image_path data/ --gt_eval_file data/vqa-catala/custom_catala.json --no_server_evaluation --max_len 40
python eval.py --batch_size 1 --language ca --embedding_type bpemb --bpemb_subtype multi --output_folder bpemb_ca_custom --model_to_evaluate ./outputs/models/bpemb_ca --image_path data/ --gt_eval_file data/vqa-catala/custom_catala.json --no_server_evaluation --max_len 40


#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder cc_en_zh --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder cc_ca_zh --model_to_evaluate ./outputs/models/cc_ca
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder cc_es_zh --model_to_evaluate ./outputs/models/cc_es
# python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder cc_zh_ca --model_to_evaluate ./outputs/models/cc_zh
# python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder cc_zh_es --model_to_evaluate ./outputs/models/cc_zh
# python eval.py --batch_size 55 --language en --fasttext_subtype cc --output_folder cc_zh_en --model_to_evaluate ./outputs/models/cc_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder cc_zh_zh --model_to_evaluate ./outputs/models/cc_zh


#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder cc_en_ca --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder cc_en_es --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --output_folder cc_en_en --model_to_evaluate ./outputs/models/cc_en

#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --output_folder wiki_en_zh --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --output_folder wiki_ca_zh --model_to_evaluate ./outputs/models/wiki_ca
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --output_folder wiki_es_zh --model_to_evaluate ./outputs/models/wiki_es
# python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --output_folder wiki_zh_ca --model_to_evaluate ./outputs/models/wiki_zh
# python eval.py --batch_size 55 --language es --fasttext_subtype wiki --output_folder wiki_zh_es --model_to_evaluate ./outputs/models/wiki_zh
# python eval.py --batch_size 55 --language en --fasttext_subtype wiki --output_folder wiki_zh_en --model_to_evaluate ./outputs/models/wiki_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --output_folder wiki_zh_zh --model_to_evaluate ./outputs/models/wiki_zh


#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --output_folder wiki_en_ca --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --output_folder wiki_en_es --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language en --fasttext_subtype wiki --output_folder wiki_en_en --model_to_evaluate ./outputs/models/wiki_en

#python eval.py --batch_size 55 --language ca --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_ca_ca --model_to_evaluate ./outputs/models/bpemb_ca
#python eval.py --batch_size 55 --language es --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_ca_es --model_to_evaluate ./outputs/models/bpemb_ca
#python eval.py --batch_size 55 --language en --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_ca_en --model_to_evaluate ./outputs/models/bpemb_ca
#
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_es_ca --model_to_evaluate ./outputs/models/bpemb_es
#python eval.py --batch_size 55 --language es --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_es_es --model_to_evaluate ./outputs/models/bpemb_es
#python eval.py --batch_size 55 --language en --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_es_en --model_to_evaluate ./outputs/models/bpemb_es
#
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_en_ca --model_to_evaluate ./outputs/models/bpemb_en
#python eval.py --batch_size 55 --language es --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_en_es --model_to_evaluate ./outputs/models/bpemb_en
#python eval.py --batch_size 55 --language en --embedding_type bpemb --fasttext_subtype wiki --output_folder bpemb_en_en --model_to_evaluate ./outputs/models/bpemb_en

#python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder smith_en_ca --model_to_evaluate ./outputs/models/smith_en
#python eval.py --batch_size 55 --language es --embedding_type smith --output_folder smith_en_es --model_to_evaluate ./outputs/models/smith_en
#python eval.py --batch_size 55 --language en --embedding_type smith --output_folder smith_en_en --model_to_evaluate ./outputs/models/smith_en

#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --output_folder aligned_cc_ca_zh --model_to_evaluate ./outputs/models/aligned_cc_ca
# python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder aligned_cc_ca_ca_not_aligned --model_to_evaluate ./outputs/models/aligned_cc_ca

# python eval.py --batch_size 55 --language en --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_es_en --model_to_evaluate ./outputs/models/aligned_cc_es
# python eval.py --batch_size 55 --language ca --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_es_ca --model_to_evaluate ./outputs/models/aligned_cc_es
# python eval.py --batch_size 55 --language es --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_cc_en_es --model_to_evaluate ./outputs/models/aligned_cc_es
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_es_zh --model_to_evaluate ./outputs/models/aligned_cc_es
# python eval.py --batch_size 55 --language es --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_es_es_not_aligned --model_to_evaluate ./outputs/models/aligned_cc_es

# python eval.py --batch_size 55 --language en --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_zh_en --model_to_evaluate ./outputs/models/aligned_cc_zh
# python eval.py --batch_size 55 --language ca --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_zh_ca --model_to_evaluate ./outputs/models/aligned_cc_zh
# python eval.py --batch_size 55 --language es --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_zh_es --model_to_evaluate ./outputs/models/aligned_cc_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder aligned_cc_zh_zh --model_to_evaluate ./outputs/models/aligned_cc_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --embedding_type fasttext --output_folder aligned_cc_zh_zh_not_aligned --model_to_evaluate ./outputs/models/aligned_cc_zh

#python eval.py --batch_size 55 --language en --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_es_en --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_es_ca --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_wiki_es_es --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_es_zh --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_es_es_not_aligned --model_to_evaluate ./outputs/models/aligned_wiki_es

# python eval.py --batch_size 55 --language en --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_zh_en --model_to_evaluate ./outputs/models/aligned_wiki_zh
# python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_zh_ca --model_to_evaluate ./outputs/models/aligned_wiki_zh
# python eval.py --batch_size 55 --language es --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_zh_es --model_to_evaluate ./outputs/models/aligned_wiki_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder aligned_wiki_zh_zh --model_to_evaluate ./outputs/models/aligned_wiki_zh
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --embedding_type fasttext --output_folder aligned_wiki_zh_zh_not_aligned --model_to_evaluate ./outputs/models/aligned_wiki_zh
