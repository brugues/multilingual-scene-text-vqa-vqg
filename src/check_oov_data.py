import os
import json
from gensim.models.keyedvectors import KeyedVectors


if __name__ == '__main__':
    train_files = {
        'ca': 'data/stvqa_train_ca.json',
        'es': 'data/stvqa_train_es.json',
        'en': 'data/stvqa_train.json'
    }

    eval_files = {
        'ca': 'data/stvqa_eval_ca.json',
        'es': 'data/stvqa_eval_es.json',
        'en': 'data/stvqa_eval.json'
    }

    model_files = {
        'ca': 'models/bin/fasttext/aligned_word_vectors/wiki.ca.align.vec',
        'es': 'models/bin/fasttext/aligned_word_vectors/wiki.es.align.vec',
        'en': 'models/bin/fasttext/aligned_word_vectors/wiki.en.align.vec',
    }

    oov_words = {
        'ca': {
            'num_eval_words': 0,
            'num_train_words': 0,
            'train_words': [],
            'eval_words': []
        },

        'es': {
            'num_eval_words': 0,
            'num_train_words': 0,
            'train_words': [],
            'eval_words': []
        },

        'en': {
            'num_eval_words': 0,
            'num_train_words': 0,
            'train_words': [],
            'eval_words': []
        },
    }

    langs = ['ca', 'es', 'en']

    for lang in langs:
        # Load model
        model = KeyedVectors.load_word2vec_format(model_files[lang], binary=False)

        # Train file
        with open(train_files[lang], 'r') as file:
            train_data = json.load(file)

        with open(eval_files[lang], 'r') as file:
            eval_data = json.load(file)

        for entry in train_data:
            words = []
            words.extend(entry['answer'])
            for bbox in entry['ans_bboxes']:
                words.extend(bbox['text'])
            for bbox in entry['ocr_bboxes']:
                words.extend(bbox['text'])
            words.extend(entry['question'])

            for word in words:
                try:
                    emb = model.get_vector(word)
                except:
                    oov_words[lang]['train_words'].append(word)
                    oov_words[lang]['num_train_words'] += 1

        # Eval file
        for entry in eval_data:
            words = []
            for bbox in entry['ocr_bboxes']:
                words.extend(bbox['text'])
            words.extend(entry['question'])

            for word in words:
                try:
                    emb = model.get_vector(word)
                except:
                    oov_words[lang]['eval_words'].append(word)
                    oov_words[lang]['num_eval_words'] += 1

    output_folder = 'outputs/oov'
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, 'results.json'), 'w+') as file:
        json.dump(oov_words, file, ensure_ascii=False)
