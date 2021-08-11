import io
import os
import json
from tqdm import tqdm
from google.cloud import vision
from google.oauth2 import service_account


class GoogleCloudOCR:
    def __init__(self, config):
        self.config = config
        self.language = config.language
        self.dataset = config.dataset

        # This is to avoid publishing sensible information on GitHub
        with open(config.json_config_file, 'r') as file:
            self.google_cloud_config = json.load(file)

        self.project_id = self.google_cloud_config['project_id']
        self.parent = "projects/{}/locations/global".format(self.project_id)
        credentials = service_account.Credentials.from_service_account_file(self.google_cloud_config['service_account'])

        self.client = vision.ImageAnnotatorClient(credentials=credentials)
        self.base_path = './data/EST-VQA-v1.0'

        # Load dataset json data
        with open(os.path.join(self.base_path, 'annotations', 'original',
                               '{}_english.json'.format(self.dataset)), 'r') as file:
            self.dataset_original_data = json.load(file)

    def order_texts_on_answer(self):
        pass

    def run_ocr_on_image(self, image_path):
        with io.open(os.path.join(self.base_path, 'images', self.dataset, image_path), 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = self.client.text_detection(image=image,
                                              image_context={"language_hints": [self.config.language]})
        texts = response.text_annotations
        ocr_boxes = []

        for text in texts:
            vertices = ([(vertex.x, vertex.y)
                         for vertex in text.bounding_poly.vertices])
            bbox = [int(vertices[0][0]),
                    int(vertices[0][1]),
                    int(vertices[2][0]),
                    int(vertices[1][1])]
            ocr = text.description

            if ocr is not ":":
                if "\n" not in ocr:
                    ocr_box = {
                        "text": ocr,
                        "bbox": bbox
                    }

                    ocr_boxes.append(ocr_box)

        return ocr_boxes

    def run_ocr_on_dataset(self):
        dataset_new_data = []

        for entry in tqdm(self.dataset_original_data,
                          'Running OCR on {} dataset'.format(self.dataset)):
            ocr_boxes = self.run_ocr_on_image(entry['image'])

            for annotation in entry['annotation']:
                new_entry = {
                    'answer': [annotation['answer']],
                    'ocr_bboxes': [],
                    'question': annotation['question'].split(' '),
                    'file_path': os.path.join('images', self.dataset, entry['image']),
                    'ans_bboxes': []
                }

                pt1 = (entry['annotation'][0]['evidence'][0], entry['annotation'][0]['evidence'][1])
                pt2 = (entry['annotation'][0]['evidence'][-4], entry['annotation'][0]['evidence'][-3])
                box = [pt1[0], pt1[1], pt2[0], pt2[1]]
                new_entry['ans_bboxes'].append({
                    "text": annotation['answer'],
                    "bbox": box
                })

                # texts_on_answers = {}
                # for i, ocr in enumerate(ocr_boxes):
                #     text = ocr['text']
                #
                #     if text in annotation['answer']:
                #         texts_on_answers[i] = ocr

                for ocr in ocr_boxes:
                    new_entry['ocr_bboxes'].append(ocr)

                dataset_new_data.append(new_entry)

        with open(os.path.join(self.base_path, 'annotations',
                               '{}_english_adapted.json'.format(self.dataset)), 'w+') as f:
            json.dump(dataset_new_data, f, ensure_ascii=False)
