#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1
#SBATCH -D /home/jbrugues/multilingual-scene-text-vqa/src/
#SBATCH -p dcc # or mlow Partition to submit to master low prioriy queue
#SBATCH --mem 16384
#SBATCH --gres gpu:Pascal:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

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

python eval.py --batch_size 55 --language en --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder aligned_cc_en_ca --model_to_evaluate ./outputs/models/cc_en
python eval.py --batch_size 55 --language en --fasttext_subtype cc --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_cc_en_es --model_to_evaluate ./outputs/models/cc_en
python eval.py --batch_size 55 --language en --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder aligned_wiki_en_ca --model_to_evaluate ./outputs/models/wiki_en
python eval.py --batch_size 55 --language en --fasttext_subtype wiki --embedding_type fasttext --fasttext_aligned --fasttext_aligned_pair es-en --output_folder aligned_wiki_en_es --model_to_evaluate ./outputs/models/wiki_en
