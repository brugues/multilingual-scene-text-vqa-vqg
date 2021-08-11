import json


def process_dataset(file):
    with open(file, 'r') as f:
        train_data = json.load(f)

    new_train_data = []
    new_english_train_data = []
    idx = 0
    for image in train_data:
        chinese = True
        for annotation in image['annotation']:
            try:
                str(str(annotation['question']) + str(annotation['answer'])).encode('ascii')
                chinese = False
            except UnicodeEncodeError:
                print("it was not a ascii-encoded unicode string")
            except TypeError:
                print("as")
                idx += 1
            else:
                print("It may have been an ascii-encoded unicode string")
                chinese = False

        if chinese:
            new_train_data.append(image)
        else:
            new_english_train_data.append(image)

    with open(file.replace('.json', '_english.json'), 'w+') as f:
        json.dump(new_english_train_data, f, ensure_ascii=False)


def main():
    test_json = "../data/EST-VQA-v1.0/annotations/original/test.json"
    train_json = "../data/EST-VQA-v1.0/annotations/original/train.json"

    datasets = [test_json, train_json]

    for dataset in datasets:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
