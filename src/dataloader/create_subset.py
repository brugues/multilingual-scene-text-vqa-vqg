import os
import json
import glob


if __name__ == '__main__':
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
