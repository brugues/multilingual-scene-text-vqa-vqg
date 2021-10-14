#!/bin/sh

#python train_olra.py --use_gru --lr 0.001 --n_epochs 15 --no_shuffle --fasttext_subtype wiki --language en --output_folder ALL_olra_wiki_en_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 15 --no_shuffle --fasttext_subtype wiki --language en --output_folder ATT_olra_wiki_en_4000 --vocabulary_size 4000 --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 15 --no_shuffle --fasttext_subtype wiki --language en --output_folder OCR_olra_wiki_en_4000 --vocabulary_size 4000 --ocr_consistency
#python train_olra.py --use_gru --lr 0.001 --n_epochs 15 --no_shuffle --fasttext_subtype wiki --language en --output_folder NONE_olra_wiki_en_4000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10ALL_olra_wiki_en_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10ATT_olra_wiki_en_4000 --vocabulary_size 4000 --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10OCR_olra_wiki_en_4000 --vocabulary_size 4000 --ocr_consistency
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10NONE_olra_wiki_en_4000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder ALL_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/ALL_olra_wiki_en_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder ATT_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/ATT_olra_wiki_en_4000 --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder OCR_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/OCR_olra_wiki_en_4000 --ocr_consistency
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder NONE_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/NONE_olra_wiki_en_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10ALL_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/10ALL_olra_wiki_en_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10ATT_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/10ATT_olra_wiki_en_4000 --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10OCR_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/10OCR_olra_wiki_en_4000 --ocr_consistency
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10NONE_olra_wiki_en_4000 --model_to_evaluate outputs/models/olra/10NONE_olra_wiki_en_4000

# Monolingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en --output_folder DEF_olra_wiki_en_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language ca --output_folder DEF_olra_wiki_ca_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language es --output_folder DEF_olra_wiki_es_4000 --ocr_consistency --multimodal_attention

## Bilingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair ca-en --language en-ca --output_folder olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair es-en --language en-es --output_folder olra_wiki_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair es-ca --language ca-es --output_folder olra_wiki_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language ca-es --output_folder olra_bpemb_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-ca --output_folder olra_bpemb_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-es --output_folder olra_bpemb_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000


## Trilingual
python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair en-ca-es --language en-ca-es --tokenizer_language en-ca-es --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_12000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language en-ca-es --output_folder olra_bpemb_en_ca_es_12000 --ocr_consistency --multimodal_attention
#
## Ablations
## OCR CONSISTENCY
## Monolingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en --output_folder olra_wiki_en_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca --output_folder olra_wiki_ca_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language es --output_folder olra_wiki_es_ocr_4000 --ocr_consistency --multimodal_attention
#
## Bilingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --output_folder olra_wiki_en_ca_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --output_folder olra_wiki_en_es_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --output_folder olra_wiki_ca_es_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --output_folder olra_bpemb_en_ca_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --output_folder olra_bpemb_en_es_ocr_4000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --output_folder olra_bpemb_ca_es_ocr_4000 --ocr_consistency --multimodal_attention
#
## Trilingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_ocr_12000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_12000 --ocr_consistency --multimodal_attention
#
## VOCABULARY SIZE
## Bilingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-ca --vocabulary_size 6000 --output_folder olra_wiki_en_ca_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-es --vocabulary_size 6000 --output_folder olra_wiki_en_es_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language ca-es --vocabulary_size 6000 --output_folder olra_wiki_ca_es_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-ca --vocabulary_size 8000 --output_folder olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-es --vocabulary_size 8000 --output_folder olra_wiki_en_es_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language ca-es --vocabulary_size 8000 --output_folder olra_wiki_ca_es_8000 --ocr_consistency --multimodal_attention
#
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 6000 --output_folder olra_bpemb_en_ca_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 6000 --output_folder olra_bpemb_en_es_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 6000 --output_folder olra_bpemb_ca_es_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 8000 --output_folder olra_bpemb_en_ca_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 8000 --output_folder olra_bpemb_en_es_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 8000 --output_folder olra_bpemb_ca_es_8000 --ocr_consistency --multimodal_attention
#
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --vocabulary_size 6000 --output_folder olra_wiki_en_ca_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --vocabulary_size 6000 --output_folder olra_wiki_en_es_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --vocabulary_size 6000 --output_folder olra_wiki_ca_es_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --vocabulary_size 8000 --output_folder olra_wiki_en_ca_ocr_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --vocabulary_size 8000 --output_folder olra_wiki_en_es_ocr_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --vocabulary_size 8000 --output_folder olra_wiki_ca_es_ocr_8000 --ocr_consistency --multimodal_attention
#
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 6000 --output_folder olra_bpemb_en_ca_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 6000 --output_folder olra_bpemb_en_es_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 6000 --output_folder olra_bpemb_ca_es_ocr_6000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 8000 --output_folder olra_bpemb_en_ca_ocr_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 8000 --output_folder olra_bpemb_en_es_ocr_8000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 8000 --output_folder olra_bpemb_ca_es_ocr_8000 --ocr_consistency --multimodal_attention
#
#
## Trilingual
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-ca-es --vocabulary_size 9000 --output_folder olra_wiki_en_ca_es_9000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 9000 --language ca-es --output_folder olra_bpemb_en_ca_es_9000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --fasttext_subtype wiki --language en-ca-es --vocabulary_size 15000 --output_folder olra_wiki_en_ca_es_15000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 15000 --language ca-es --output_folder olra_bpemb_en_ca_es_15000 --ocr_consistency --multimodal_attention
#
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 9000 --output_folder olra_wiki_en_ca_es_ocr_9000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 9000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_9000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 15000 --output_folder olra_wiki_en_ca_es_ocr_15000 --ocr_consistency --multimodal_attention
#python train_olra.py --use_gru --lr 0.001 --n_epochs 10 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 15000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_15000 --ocr_consistency --multimodal_attention