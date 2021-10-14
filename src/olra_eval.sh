#!/bin/sh

#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 10_olra_wiki_en_ca_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/10_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder 10_olra_wiki_en_ca_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/10_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 15_olra_wiki_en_ca_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/15_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder 15_olra_wiki_en_ca_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/15_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 20_olra_wiki_en_ca_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/20_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder 20_olra_wiki_en_ca_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/20_olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000

# Monolingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder olra_bpemb_en_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_bpemb_en_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder olra_bpemb_ca_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_bpemb_ca_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language es --output_folder olra_bpemb_es_4000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_bpemb_es_4000 --ocr_consistency --multimodal_attention

## Bilingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --tokenizer_language en-ca --fasttext_subtype wiki --language en --output_folder olra_wiki_en_ca_8000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --tokenizer_language en-ca --fasttext_subtype wiki --language ca --fasttext_aligned --fasttext_aligned_pair ca-en --output_folder olra_wiki_en_ca_8000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_wiki_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --tokenizer_language en-es --fasttext_subtype wiki --language en --output_folder olra_wiki_en_es_8000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_wiki_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --tokenizer_language en-es --fasttext_subtype wiki --language es --output_folder olra_wiki_en_es_8000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_wiki_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000

#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder olra_wiki_ca_es_8000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_wiki_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language es --output_folder olra_wiki_ca_es_8000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_wiki_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000

#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 20_olra_wiki_en_ca_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/20_olra_wiki_en_ca_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder 20_olra_wiki_en_ca_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/20_olra_wiki_en_ca_4000 --ocr_consistency --multimodal_attention
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en --output_folder 20_olra_wiki_en_es_4000_EvaluatedInEN --model_to_evaluate outputs/models/olra/20_olra_wiki_en_es_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language es --output_folder 20_olra_wiki_en_es_4000_EvaluatedInES --model_to_evaluate outputs/models/olra/20_olra_wiki_en_es_4000 --ocr_consistency --multimodal_attention
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca --output_folder 20_olra_wiki_ca_es_4000_EvaluatedInCA --model_to_evaluate outputs/models/olra/20_olra_wiki_ca_es_4000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language es --output_folder 20_olra_wiki_ca_es_4000_EvaluatedInES --model_to_evaluate outputs/models/olra/20_olra_wiki_ca_es_4000 --ocr_consistency --multimodal_attention

#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language en-ca  --language en --output_folder olra_bpemb_en_ca_8000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_bpemb_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language en-ca  --language ca --output_folder olra_bpemb_en_ca_8000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_bpemb_en_ca_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language en-es  --language en --output_folder olra_bpemb_en_es_8000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_bpemb_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language en-es  --language es --output_folder olra_bpemb_en_es_8000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_bpemb_en_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language ca-es  --language ca --output_folder olra_bpemb_ca_es_8000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_bpemb_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --tokenizer_language ca-es  --language es --output_folder olra_bpemb_ca_es_8000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_bpemb_ca_es_8000 --ocr_consistency --multimodal_attention --vocabulary_size 8000


### Trilingual
python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair en-ca-es --tokenizer_language en-ca-es --language en --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_12000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_wiki_en_ca_es_12000 --ocr_consistency --multimodal_attention
python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair en-ca-es --tokenizer_language en-ca-es --language ca --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_12000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_wiki_en_ca_es_12000 --ocr_consistency --multimodal_attention
python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --fasttext_aligned --fasttext_aligned_pair en-ca-es --tokenizer_language en-ca-es --language es --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_12000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_wiki_en_ca_es_12000 --ocr_consistency --multimodal_attention
#
#python eval_olra.py --use_gru --batch_size 1 --tokenizer_language en-ca-es --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language en --output_folder olra_bpemb_en_ca_es_12000_EvaluatedInEN --model_to_evaluate outputs/models/olra/olra_bpemb_en_ca_es_12000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --tokenizer_language en-ca-es --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language ca --output_folder olra_bpemb_en_ca_es_12000_EvaluatedInCA --model_to_evaluate outputs/models/olra/olra_bpemb_en_ca_es_12000 --ocr_consistency --multimodal_attention
#python eval_olra.py --use_gru --batch_size 1 --tokenizer_language en-ca-es --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language es --output_folder olra_bpemb_en_ca_es_12000_EvaluatedInES --model_to_evaluate outputs/models/olra/olra_bpemb_en_ca_es_12000 --ocr_consistency --multimodal_attention

# Ablations
# OCR CONSISTENCY
# Monolingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en --output_folder olra_wiki_en_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca --output_folder olra_wiki_ca_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language es --output_folder olra_wiki_es_ocr_4000
#
## Bilingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --output_folder olra_wiki_en_ca_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --output_folder olra_wiki_en_es_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --output_folder olra_wiki_ca_es_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --output_folder olra_bpemb_en_ca_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --output_folder olra_bpemb_en_es_ocr_4000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --output_folder olra_bpemb_ca_es_ocr_4000
#
## Trilingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 12000 --output_folder olra_wiki_en_ca_es_ocr_12000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 12000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_12000
#
## VOCABULARY SIZE
## Bilingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-ca --vocabulary_size 6000 --output_folder olra_wiki_en_ca_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-es --vocabulary_size 6000 --output_folder olra_wiki_en_es_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca-es --vocabulary_size 6000 --output_folder olra_wiki_ca_es_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-ca --vocabulary_size 8000 --output_folder olra_wiki_en_ca_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-es --vocabulary_size 8000 --output_folder olra_wiki_en_es_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language ca-es --vocabulary_size 8000 --output_folder olra_wiki_ca_es_8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 6000 --output_folder olra_bpemb_en_ca_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 6000 --output_folder olra_bpemb_en_es_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 6000 --output_folder olra_bpemb_ca_es_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 8000 --output_folder olra_bpemb_en_ca_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 8000 --output_folder olra_bpemb_en_es_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 8000 --output_folder olra_bpemb_ca_es_8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --vocabulary_size 6000 --output_folder olra_wiki_en_ca_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --vocabulary_size 6000 --output_folder olra_wiki_en_es_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --vocabulary_size 6000 --output_folder olra_wiki_ca_es_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca --vocabulary_size 8000 --output_folder olra_wiki_en_ca_ocr_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-es --vocabulary_size 8000 --output_folder olra_wiki_en_es_ocr_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language ca-es --vocabulary_size 8000 --output_folder olra_wiki_ca_es_ocr_8000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 6000 --output_folder olra_bpemb_en_ca_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 6000 --output_folder olra_bpemb_en_es_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 6000 --output_folder olra_bpemb_ca_es_ocr_6000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language ca-es --vocabulary_size 8000 --output_folder olra_bpemb_en_ca_ocr_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-ca --vocabulary_size 8000 --output_folder olra_bpemb_en_es_ocr_8000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --language en-es --vocabulary_size 8000 --output_folder olra_bpemb_ca_es_ocr_8000
#
#
## Trilingual
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-ca-es --vocabulary_size 9000 --output_folder olra_wiki_en_ca_es_9000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 9000 --language ca-es --output_folder olra_bpemb_en_ca_es_9000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --fasttext_subtype wiki --language en-ca-es --vocabulary_size 15000 --output_folder olra_wiki_en_ca_es_15000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 15000 --language ca-es --output_folder olra_bpemb_en_ca_es_15000
#
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 9000 --output_folder olra_wiki_en_ca_es_ocr_9000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 9000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_9000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --fasttext_subtype wiki --language en-ca-es --vocabulary_size 15000 --output_folder olra_wiki_en_ca_es_ocr_15000
#python eval_olra.py --use_gru --batch_size 1 --no_shuffle --ocr_consistency --embedding_type bpemb --bpemb_subtype multi --vocabulary_size 15000 --language ca-es --output_folder olra_bpemb_en_ca_es_ocr_15000
