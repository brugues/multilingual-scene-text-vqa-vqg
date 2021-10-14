#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

python eval.py --batch_size 55 --language en --embedding_type smith --output_folder smith_combined_en_ca_evaluatedIn_EN --model_to_evaluate ./outputs/models/smith_combined_en_ca
python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder smith_combined_en_ca_evaluatedIn_CA --model_to_evaluate ./outputs/models/smith_combined_en_ca
python eval.py --batch_size 55 --language es --embedding_type smith --output_folder smith_combined_en_ca_evaluatedIn_ES --model_to_evaluate ./outputs/models/smith_combined_en_ca
python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder smith_combined_en_ca_evaluatedIn_ZH --model_to_evaluate ./outputs/models/smith_combined_en_ca

python eval.py --batch_size 55 --language en --embedding_type smith --output_folder smith_combined_en_es_evaluatedIn_EN --model_to_evaluate ./outputs/models/smith_combined_en_es
python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder smith_combined_en_es_evaluatedIn_CA --model_to_evaluate ./outputs/models/smith_combined_en_es
python eval.py --batch_size 55 --language es --embedding_type smith --output_folder smith_combined_en_es_evaluatedIn_ES --model_to_evaluate ./outputs/models/smith_combined_en_es
python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder smith_combined_en_es_evaluatedIn_ZH --model_to_evaluate ./outputs/models/smith_combined_en_es

python eval.py --batch_size 55 --language en --embedding_type smith --output_folder smith_combined_en_zh_evaluatedIn_EN --model_to_evaluate ./outputs/models/smith_combined_en_zh
python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder smith_combined_en_zh_evaluatedIn_CA --model_to_evaluate ./outputs/models/smith_combined_en_zh
python eval.py --batch_size 55 --language es --embedding_type smith --output_folder smith_combined_en_zh_evaluatedIn_ES --model_to_evaluate ./outputs/models/smith_combined_en_zh
python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder smith_combined_en_zh_evaluatedIn_ZH --model_to_evaluate ./outputs/models/smith_combined_en_zh

python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_subtype cc --output_folder cc_combined_en_ca_evaluatedIn_EN --model_to_evaluate ./outputs/models/cc_combined_en_ca
python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-ca --output_folder cc_combined_en_ca_evaluatedIn_CA --model_to_evaluate ./outputs/models/cc_combined_en_ca
python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-es --output_folder cc_combined_en_ca_evaluatedIn_ES --model_to_evaluate ./outputs/models/cc_combined_en_ca
python eval.py --batch_size 55 --language zh --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-zh --output_folder cc_combined_en_ca_evaluatedIn_ZH --model_to_evaluate ./outputs/models/cc_combined_en_ca

python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_subtype cc --output_folder cc_combined_en_es_evaluatedIn_EN --model_to_evaluate ./outputs/models/cc_combined_en_es
python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-ca --output_folder cc_combined_en_es_evaluatedIn_CA --model_to_evaluate ./outputs/models/cc_combined_en_es
python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-es --output_folder cc_combined_en_es_evaluatedIn_ES --model_to_evaluate ./outputs/models/cc_combined_en_es
python eval.py --batch_size 55 --language zh --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-zh --output_folder cc_combined_en_es_evaluatedIn_ZH --model_to_evaluate ./outputs/models/cc_combined_en_es

python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_subtype cc --output_folder smith_combined_en_zh_evaluatedIn_EN --model_to_evaluate ./outputs/models/cc_combined_en_zh
python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-ca --output_folder cc_combined_en_zh_evaluatedIn_CA --model_to_evaluate ./outputs/models/cc_combined_en_zh
python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-es --output_folder cc_combined_en_zh_evaluatedIn_ES --model_to_evaluate ./outputs/models/cc_combined_en_zh
python eval.py --batch_size 55 --language zh --embedding_type fasttext --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair en-zh --output_folder cc_combined_en_zh_evaluatedIn_ZH --model_to_evaluate ./outputs/models/cc_combined_en_zh

#python eval.py --batch_size 55 --language ca --embedding_type smith --output_folder smith_ca_CA --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language es --embedding_type smith --output_folder smith_ca_ES --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language en --embedding_type smith --output_folder smith_ca_EN --model_to_evaluate ./outputs/models/smith_ca
#python eval.py --batch_size 55 --language zh --embedding_type smith --output_folder smith_ca_ZH --model_to_evaluate ./outputs/models/smith_ca

#python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_en --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_ca --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_es --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language zh --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_zh --model_to_evaluate ./outputs/models/aligned_wiki_ca

#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder cc_en_ca --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair es-en --output_folder cc_en_es --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language zh --fasttext_subtype cc --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder cc_en_zh --model_to_evaluate ./outputs/models/cc_en
#
#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder wiki_en_ca --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair es-en --output_folder wiki_en_es --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language zh --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair zh-en --output_folder wiki_en_zh --model_to_evaluate ./outputs/models/wiki_en


#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder cc_ca_ca --model_to_evaluate ./outputs/models/cc_ca
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder cc_ca_es --model_to_evaluate ./outputs/models/cc_ca
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --output_folder cc_ca_en --model_to_evaluate ./outputs/models/cc_ca
#
#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder cc_es_ca --model_to_evaluate ./outputs/models/cc_es
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder cc_es_es --model_to_evaluate ./outputs/models/cc_es
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --output_folder cc_es_en --model_to_evaluate ./outputs/models/cc_es
#
#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --output_folder cc_en_ca --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --output_folder cc_en_es --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language en --fasttext_subtype cc --output_folder cc_en_en --model_to_evaluate ./outputs/models/cc_en

#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --output_folder wiki_ca_ca --model_to_evaluate ./outputs/models/wiki_ca
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --output_folder wiki_ca_es --model_to_evaluate ./outputs/models/wiki_ca
#python eval.py --batch_size 55 --language en --fasttext_subtype wiki --output_folder wiki_ca_en --model_to_evaluate ./outputs/models/wiki_ca
#
#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --output_folder wiki_es_ca --model_to_evaluate ./outputs/models/wiki_es
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --output_folder wiki_es_es --model_to_evaluate ./outputs/models/wiki_es
#python eval.py --batch_size 55 --language en --fasttext_subtype wiki --output_folder wiki_es_en --model_to_evaluate ./outputs/models/wiki_es
#
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

#python eval.py --batch_size 55 --language ca --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder aligned_cc_en_ca --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language es --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_cc_en_es --model_to_evaluate ./outputs/models/cc_en
#python eval.py --batch_size 55 --language ca --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder aligned_wiki_en_ca --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language es --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_wiki_en_es --model_to_evaluate ./outputs/models/wiki_en
#python eval.py --batch_size 55 --language en --fasttext_subtype wiki --embedding_type fasttext --output_folder wiki_en --model_to_evaluate ./outputs/models/wiki_en

#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_en_en --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_en_ca --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_en_es --model_to_evaluate ./outputs/models/multi_bpemb_en
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_ca_en --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_ca_ca --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_ca_es --model_to_evaluate ./outputs/models/multi_bpemb_ca
#python eval.py --batch_size 55 --language en --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_es_en --model_to_evaluate ./outputs/models/multi_bpemb_es
#python eval.py --batch_size 55 --language ca --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_es_ca --model_to_evaluate ./outputs/models/multi_bpemb_es
#python eval.py --batch_size 55 --language es --embedding_type bpemb --bpemb_subtype multi --embedding_type bpemb --fasttext_aligned --output_folder multi_bpemb_es_es --model_to_evaluate ./outputs/models/multi_bpemb_es

#python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_ca_en --model_to_evaluate ./outputs/models/aligned_cc_ca
#python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_ca_ca --model_to_evaluate ./outputs/models/aligned_cc_ca
#python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_ca_es --model_to_evaluate ./outputs/models/aligned_cc_ca
#python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_es_en --model_to_evaluate ./outputs/models/aligned_cc_es
#python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_es_ca --model_to_evaluate ./outputs/models/aligned_cc_es
#python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_aligned --fasttext_subtype cc --fasttext_aligned --output_folder aligned_cc_es_es --model_to_evaluate ./outputs/models/aligned_cc_es
#
#python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_en --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_ca --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_ca_es --model_to_evaluate ./outputs/models/aligned_wiki_ca
#python eval.py --batch_size 55 --language en --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_es_en --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language ca --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_es_ca --model_to_evaluate ./outputs/models/aligned_wiki_es
#python eval.py --batch_size 55 --language es --embedding_type fasttext --fasttext_aligned --fasttext_subtype wiki --output_folder aligned_wiki_es_es --model_to_evaluate ./outputs/models/aligned_wiki_es
