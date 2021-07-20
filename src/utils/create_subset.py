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


def estvqa():
    import random

    file = "/home/josep/Documents/Git/tfm/multilingual-scene-text-vqa/src/data/EST-VQA-v1.0/annotations/train_chinese_adapted.json"

    with open(file, 'r') as f:
        data = json.load(f)

    sample = random.sample(data, int(len(data)*0.075))

    for element in sample:
        data.remove(element)

    with open(file.replace('_chinese_adapted', '_chinese_subsample'), 'w+') as f:
        json.dump(data, f, ensure_ascii=False)

    with open(file.replace('train_chinese_adapted', 'eval_chinese_subsample'), 'w+') as f:
        json.dump(sample, f, ensure_ascii=False)


if __name__ == '__main__':
    estvqa()
