import os
import json
import glob


def stvqa():
    data_folder = '../data'
    file_list = glob.glob(os.path.join(data_folder, '*.json'))

    for filename in file_list:
        with open(filename, 'r') as file:
            data = json.load(file)

        if 'train' in filename:
            subset = data[:1000]

            for i, entry in enumerate(subset):
                entry['question_id'] = i
        else:
            subset = data[:200]

        subset_file = filename.replace('.json', '_subset.json')

        with open(subset_file, 'w+') as file:
            json.dump(subset, file, ensure_ascii=False)


# def estvqa():
#     import random
#
#     file = "/home/josep/Documents/Git/tfm/multilingual-scene-text-vqa/src/data/EST-VQA-v1.0/annotations/train_chinese_adapted.json"
#
#     with open(file, 'r') as f:
#         data = json.load(f)
#
#     sample = random.sample(data, int(len(data)*0.075))
#
#     for element in sample:
#         data.remove(element)
#
#     with open(file.replace('_chinese_adapted', '_chinese_subsample'), 'w+') as f:
#         json.dump(data, f, ensure_ascii=False)
#
#     with open(file.replace('train_chinese_adapted', 'eval_chinese_subsample'), 'w+') as f:
#         json.dump(sample, f, ensure_ascii=False)


def estvqa():
    import random

    train = "/Volumes/Transcend/Git/tfm/multilingual-scene-text-vqa/src/data/EST-VQA-v1.0/annotations/original/train_chinese_subsample.json"
    eval = "/Volumes/Transcend/Git/tfm/multilingual-scene-text-vqa/src/data/EST-VQA-v1.0/annotations/original/eval_chinese_subsample.json"

    with open(train, 'r') as f:
        train_data = json.load(f)

    with open(eval, 'r') as f:
        eval_data = json.load(f)

    train_entries = len(train_data)
    eval_entries = len(eval_data)

    train_answers_in_ocr_tokens = 0
    train_entries_with_answer_in_ocr_token = []
    for entry in train_data:
        texts = []
        for boxes in entry['ocr_bboxes']:
            texts.append(boxes['text'])

        if entry['answer'][0] in texts:
            train_answers_in_ocr_tokens += 1
            train_entries_with_answer_in_ocr_token.append(entry)

    eval_answers_in_ocr_tokens = 0
    eval_entries_with_answer_in_ocr_token = []
    for entry in eval_data:
        texts = []
        for boxes in entry['ocr_bboxes']:
            texts.append(boxes['text'])

        if entry['answer'][0] in texts:
            eval_answers_in_ocr_tokens += 1
            eval_entries_with_answer_in_ocr_token.append(entry)

    assert len(eval_entries_with_answer_in_ocr_token) == eval_answers_in_ocr_tokens
    assert len(train_entries_with_answer_in_ocr_token) == train_answers_in_ocr_tokens

    train_max_possible_score = train_answers_in_ocr_tokens / train_entries
    eval_max_possible_score = eval_answers_in_ocr_tokens / eval_entries

    with open(train.replace('original/train_chinese_subsample', 'train_subsample'), 'w+') as f:
        json.dump(train_entries_with_answer_in_ocr_token, f)

    with open(eval.replace('original/eval_chinese_subsample', 'eval_subsample'), 'w+') as f:
        json.dump(eval_entries_with_answer_in_ocr_token, f)


if __name__ == '__main__':
    estvqa()
