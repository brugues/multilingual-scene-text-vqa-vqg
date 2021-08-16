import unidecode
import glob
import json
from tqdm import tqdm


def main():
    files = glob.glob('../data/ST-VQA/annotations/olra/*.json')

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        for entry in tqdm(data, "Unidecoding {}".format(file.split('/')[-1])):
            for key in entry:
                if key in ['answer', 'question']:
                    for i, word in enumerate(entry[key]):
                        entry[key][i] = unidecode(word)

                if key in ['ans_bboxes', 'ocr_bboxes']:
                    for i, text in enumerate(entry[key]['text']):
                        entry[key]['text'][i] = unidecode(text)

        with open(file, 'w+') as f:
            json.dump(data, f)


if __name__ == '__main__':
    main()