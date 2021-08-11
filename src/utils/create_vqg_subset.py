import os
import json
import random


def create_subset():
    stvqa_path = '../data/ST-VQA/annotations'

    with open(os.path.join(stvqa_path, 'stvqa_train_ca.json'), 'r') as f:
        catalan = json.load(f)
    with open(os.path.join(stvqa_path, 'stvqa_train_es.json'), 'r') as f:
        spanish = json.load(f)
    with open(os.path.join(stvqa_path, 'stvqa_train.json'), 'r') as f:
        english = json.load(f)

    catalan_test = random.sample(catalan, int(len(catalan) * 0.15))
    spanish_test = random.sample(spanish, int(len(spanish) * 0.15))
    english_test = random.sample(english, int(len(english) * 0.15))

    for element in catalan_test:
        catalan.remove(element)
    for element in spanish_test:
        spanish.remove(element)
    for element in english_test:
        english.remove(element)

    with open(os.path.join(stvqa_path, 'olra', 'stvqa_train_olra_ca.json'), 'w+') as f:
        json.dump(catalan, f)
    with open(os.path.join(stvqa_path, 'olra', 'stvqa_eval_olra_ca.json'), 'w+') as f:
        json.dump(catalan_test, f)

    with open(os.path.join(stvqa_path, 'olra', 'stvqa_train_olra_es.json'), 'w+') as f:
        json.dump(spanish, f)
    with open(os.path.join(stvqa_path, 'olra', 'stvqa_eval_olra_es.json'), 'w+') as f:
        json.dump(spanish_test, f)

    with open(os.path.join(stvqa_path, 'olra', 'stvqa_train_olra.json'), 'w+') as f:
        json.dump(english, f)
    with open(os.path.join(stvqa_path, 'olra', 'stvqa_eval_olra.json'), 'w+') as f:
        json.dump(english_test, f)


if __name__ == '__main__':
    create_subset()
