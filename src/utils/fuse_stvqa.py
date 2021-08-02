import json


CATALAN_TRAIN = '../data/stvqa_train_ca.json'
SPANISH_TRAIN = '../data/stvqa_train_es.json'
ENGLISH_TRAIN = '../data/stvqa_train.json'
CHINESE_TRAIN = '../data/stvqa_train_zh.json'

CATALAN_EVAL = '../data/stvqa_eval_ca.json'
SPANISH_EVAL = '../data/stvqa_eval_es.json'
ENGLISH_EVAL = '../data/stvqa_eval.json'
CHINESE_EVAL = '../data/stvqa_eval_zh.json'

ESTVQA_EVAL = '../data/EST-VQA-v1.0/annotations/eval_subsample_all_answers.json'


def fuse():
    with open(CATALAN_TRAIN, 'r') as f:
        catalan = json.load(f)

    with open(SPANISH_TRAIN, 'r') as f:
        spanish = json.load(f)

    with open(ENGLISH_TRAIN, 'r') as f:
        english = json.load(f)

    with open(CHINESE_TRAIN, 'r') as f:
        chinese = json.load(f)

    for entry in catalan:
        entry['lang'] = 'ca'
    for entry in spanish:
        entry['lang'] = 'es'
    for entry in english:
        entry['lang'] = 'en'
    for entry in chinese:
        entry['lang'] = 'zh'

    combined_en_ca = []
    combined_en_ca.extend(english)
    combined_en_ca.extend(catalan)

    combined_en_es = []
    combined_en_es.extend(english)
    combined_en_es.extend(spanish)

    combined_en_zh = []
    combined_en_zh.extend(english)
    combined_en_zh.extend(chinese)

    with open(CATALAN_TRAIN.replace('ca', 'en_ca'), 'w+') as f:
        json.dump(combined_en_ca, f)

    with open(SPANISH_TRAIN.replace('es', 'en_es'), 'w+') as f:
        json.dump(combined_en_ca, f)

    with open(CHINESE_TRAIN.replace('zh', 'en_zh'), 'w+') as f:
        json.dump(combined_en_ca, f)


def add_language_field():
    with open(CATALAN_TRAIN, 'r') as f:
        catalan = json.load(f)

    with open(SPANISH_TRAIN, 'r') as f:
        spanish = json.load(f)

    with open(ENGLISH_TRAIN, 'r') as f:
        english = json.load(f)

    with open(CHINESE_TRAIN, 'r') as f:
        chinese = json.load(f)

    for entry in catalan:
        entry['lang'] = 'ca'
    for entry in spanish:
        entry['lang'] = 'es'
    for entry in english:
        entry['lang'] = 'en'
    for entry in chinese:
        entry['lang'] = 'zh'

    with open(ENGLISH_TRAIN, 'w+') as f:
        json.dump(english, f)

    with open(CATALAN_TRAIN, 'w+') as f:
        json.dump(catalan, f)

    with open(SPANISH_TRAIN, 'w+') as f:
        json.dump(spanish, f)

    with open(CHINESE_TRAIN, 'w+') as f:
        json.dump(chinese, f)


def add_question_id_field():
    with open(ESTVQA_EVAL, 'r') as f:
        estvqa = json.load(f)

    counter = 1
    for entry in estvqa:
        entry['question_id'] = int(counter)
        counter += 1

    with open(ESTVQA_EVAL, 'w+') as f:
        json.dump(estvqa, f)


if __name__ == '__main__':
    add_question_id_field()
