import os
import json
import glob
import string


def check_word(word):
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    accents = ['à', 'á', 'è', 'é', 'ï', 'ì', 'í', 'ò', 'ó', 'ú', 'ü', 'ù', '?', '!']

    apostrophe = False
    apostrophe_position = 0
    space = False
    for i, c in enumerate(word):

        if c not in string.ascii_letters:
            if c not in numbers and c.lower() not in accents:
                if c == " ":
                    space = True
                else:
                    apostrophe = True
                    apostrophe_position = i

    if apostrophe and space:
        if apostrophe_position > 1:
            return word[:apostrophe_position-1] + word[apostrophe_position+1:]
        else:
            return word[apostrophe_position+1:]
    elif apostrophe:
        if apostrophe_position <= 1:
            return word[2:]
        elif space:
            return word
        else:
            return word[:apostrophe_position]
    else:
        return word


if __name__ == '__main__':
    data_folder = '../data'
    file_list = glob.glob(os.path.join(data_folder, '*.json'))

    for filename in file_list:
        if 'ca' in filename:
            with open(filename, 'r') as file:
                data = json.load(file, )

            new_data = []
            for entry in data:
                if 'train' in filename:
                    new_entry = {
                        'file_path': entry['file_path']
                    }

                    new_ocr_bbox = []
                    for bbox in entry['ocr_bboxes']:
                        new_ocr_bbox.append({'bbox': bbox['bbox'], 'text': check_word(bbox['text'])})
                    new_entry['ocr_bboxes'] = new_ocr_bbox

                    new_ans_bbox = []
                    for bbox in entry['ans_bboxes']:
                        new_ans_bbox.append({'bbox': bbox['bbox'], 'text': check_word(bbox['text'])})
                    new_entry['ans_bboxes'] = new_ans_bbox

                    new_question = []
                    for word in entry['question']:
                        new_question.append(check_word(word))
                    new_entry['question'] = new_question

                    new_answer = []
                    for word in entry['answer']:
                        new_answer.append(check_word(word))
                    new_entry['answer'] = new_answer

                    new_data.append(new_entry)

                else:
                    new_entry = {
                        'file_path': entry['file_path'],
                        'question_id': entry['question_id'],
                        'answer': entry['answer']
                    }

                    new_ocr_bbox = []
                    for bbox in entry['ocr_bboxes']:
                        new_ocr_bbox.append({'bbox': bbox['bbox'], 'text': check_word(bbox['text'])})
                    new_entry['ocr_bboxes'] = new_ocr_bbox

                    new_ans_bbox = []
                    for bbox in entry['ans_bboxes']:
                        new_ans_bbox.append({'bbox': bbox['bbox'], 'text': check_word(bbox['text'])})
                    new_entry['ans_bboxes'] = new_ans_bbox

                    new_question = []
                    for word in entry['question']:
                        new_question.append(check_word(word))
                    new_entry['question'] = new_question

                    new_data.append(new_entry)

            with open(filename, 'w') as file:
                json.dump(new_data, file, ensure_ascii=False)
