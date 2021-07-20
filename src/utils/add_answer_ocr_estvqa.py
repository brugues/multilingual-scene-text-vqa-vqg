import os
import json
from tqdm import tqdm


def main():
    files = ['../data/EST-VQA-v1.0/annotations/train_chinese_subsample.json',
             '../data/EST-VQA-v1.0/annotations/eval_chinese_subsample.json']

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        for entry in tqdm(data):
            answer = entry['answer'][0]
            answer_bboxes = entry['ans_bboxes'][0]

            found = False
            for ocr in entry['ocr_bboxes']:
                text = ocr['text']
                if text == answer:
                    found = True

            if not found:
                entry['ocr_bboxes'].append(answer_bboxes)

        with open(file.replace('.json', '_all_answers.json'), 'w+') as f:
            json.dump(data, f)


def check_dataset_answers():
    files = ['../data/EST-VQA-v1.0/annotations/train_chinese_subsample.json',
             '../data/EST-VQA-v1.0/annotations/eval_chinese_subsample.json']

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        counter = 0
        for entry in tqdm(data):
            answer = entry['answer'][0]
            answer_bboxes = entry['ans_bboxes'][0]

            found = False
            for ocr in entry['ocr_bboxes']:
                text = ocr['text']
                if text == answer:
                    counter += 1

        print("{} has {}/{} answers".format(file.split('/')[-1],
                                            counter,
                                            len(data)))


if __name__ == '__main__':
    main()
