import six
import json
from time import time_ns, sleep
from tqdm import tqdm
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account


class GoogleCloudTranslate:
    def __init__(self, config):
        self.config = config
        self.source_language = config.source_language
        self.dest_language = config.dest_language
        self.dataset = config.dataset

        # This is to avoid publishing sensible information on GitHub
        with open(config.json_config_file, 'r') as file:
            self.google_cloud_config = json.load(file)

        self.project_id = self.google_cloud_config['project_id']
        self.parent = "projects/{}/locations/global".format(self.project_id)
        credentials = service_account.Credentials.from_service_account_file(self.google_cloud_config['service_account'])

        self.client = translate.Client(credentials=credentials)

        # Load dataset json data
        with open('./data/stvqa_{}.json'.format(self.dataset), 'r') as file:
            self.dataset_original_data = json.load(file)

    def translate_dataset(self):
        dataset_new_data = []

        for entry in tqdm(self.dataset_original_data, 'Translating {} dataset'.format(self.dataset)):

            if self.dataset == 'eval':
                new_entry = {
                    'file_path': entry['file_path'],
                    'question_id': entry['question_id']
                }
            else:
                new_entry = {
                    'file_path': entry['file_path']
                }

            # Translate step by step
            # Ans Boxes

            answers = []
            ans_bboxes = []
            for box in entry['ans_bboxes']:
                new_box = {
                    'bbox': box['bbox']
                }

                text = box['text']

                if isinstance(text, six.binary_type):
                    text = text.decode("utf-8")

                response = self.client.translate(text,
                                                 source_language=self.source_language,
                                                 target_language=self.dest_language)
                # if "&#39;" in response['translatedText']:
                #     print("A")
                new_box['text'] = response['translatedText']
                ans_bboxes.append(new_box)

            if len(ans_bboxes) > 0:
                for box in ans_bboxes:
                    answers.append(box['text'])
            else:
                answers = 'na'

            new_entry['answer'] = answers
            new_entry['ans_bboxes'] = ans_bboxes

            ocr_boxes = []
            # OCR Boxes
            for box in entry['ocr_bboxes']:
                new_box = {
                    'bbox': box['bbox']
                }

                text = box['text']

                if isinstance(text, six.binary_type):
                    text = text.decode("utf-8")

                if len(text) > 1:
                    response = self.client.translate(text,
                                                     source_language=self.source_language,
                                                     target_language=self.dest_language)
                    new_box['text'] = response['translatedText']
                else:
                    new_box['text'] = text
                ocr_boxes.append(new_box)
            
            new_entry['ocr_bboxes'] = ocr_boxes
            
            # Question
            question = ''
            for word in entry['question']:
                question += word + ' '

            if isinstance(question, six.binary_type):
                question = text.decode("utf-8")

            response = self.client.translate(question,
                                             source_language=self.source_language,
                                             target_language=self.dest_language)
            question_translated = response['translatedText']
            question_translated = question_translated.split(' ')

            new_entry['question'] = question_translated

            dataset_new_data.append(new_entry)

        with open('./data/stvqa_{}_{}.json'.format(self.dataset, self.dest_language), 'w+') as file:
            json.dump(dataset_new_data, file, ensure_ascii=False)
