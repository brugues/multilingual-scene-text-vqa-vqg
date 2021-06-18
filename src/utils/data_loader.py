import os
import json

import fasttext
from bpemb import BPEmb
import cv2
import numpy as np

from utils.utils import print_info, print_ok
from utils.yolo_utils import yolo_image_preprocess


class STVQADataGenerator:

    def __init__(self, config, training=True):

        self.training = training
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.max_len = config.max_len
        self.image_path = config.image_path
        self.dim_txt = config.dim_txt
        self.curr_idx = 0
        self.input_size = config.img_size
        self.language = config.language
        self.embedding_type = config.embedding_type
        self.fasttext_subtype = config.fasttext_subtype
        self.bpemb_subtype = config.bpemb_subtype

        if self.embedding_type == 'fasttext':
            fasttext_path = os.path.join(config.txt_embeddings_path, 'fasttext')
            if self.language == 'en' or self.language == 'ca' or self.language == 'es':
                if self.fasttext_subtype == 'wiki':
                    try:
                        self.txt_model = fasttext.load_model(os.path.join(fasttext_path, 'wiki_word_vectors',
                                                                          'wiki.{}.bin'.format(self.language)))
                    except:
                        print("Error loading model. Check that the file exists")
                elif self.fasttext_subtype == 'wiki-news':
                    try:
                        print("Be aware! This model is only available in english, so the english model is being loaded")
                        self.txt_model = fasttext.load_model(os.path.join(config.txt_embeddings_path,
                                                                          'wiki-news-300d-1M-subword.bin'))
                    except:
                        print("Error loading model. Check that the file exists. ")
                elif self.fasttext_subtype == 'cc':
                    try:
                        self.txt_model = fasttext.load_model(os.path.join(fasttext_path, 'cc_word_vectors',
                                                                          'cc.{}.300.bin'.format(self.language)))
                    except:
                        print("Error loading model. Check that the file exists")
                elif self.fasttext_subtype == 'aligned':
                    try:
                        self.txt_model = fasttext.load_model(os.path.join(fasttext_path, 'aligned_word_vectors',
                                                                          'wiki.{}.align.bin'.format(self.language)))
                    except:
                        print("Error loading model. Check that the file exists")
                else:
                    raise AttributeError('Invalid fasttext subtype. Choose between wiki, wiki-news, cc or aligned')
            else:
                raise AttributeError('Invalid language. Select between English (en), Catalan (ca) or Spanish (es)')

        elif self.embedding_type == 'bpemb':
            bpemb_path = os.path.join(config.txt_embeddings_path, 'bpemb')
            if self.language == 'en' or self.language == 'ca' or self.language == 'es' or self.language == 'multi':
                bin_file = os.path.join(bpemb_path, '{}.wiki.bpe.vs200000.d300.w2v.bin'.format(self.language))

                if self.language == 'multi':
                    model_file = os.path.join(bpemb_path, 'multi.wiki.bpe.vs200000.d300.w2v.bin')
                else:
                    model_file = os.path.join(bpemb_path, '{}.wiki.bpe.vs200000.d300.w2v.bin'.format(self.language))

                assert str(self.dim_txt) in bin_file

                self.txt_model = BPEmb(emb_file=bin_file,
                                       model_file=model_file,
                                       dim=self.dim_txt)
            else:
                raise AttributeError('Invalid language. Select between English (en), Catalan (ca) or Spanish (es) or '
                                     'All of them (multi)')
        else:
            raise AttributeError('Invalid embedding type. Select either fasttext or bpemb')

        # load gt file
        print_info('Loading GT file...')
        if training:
            with open(config.gt_file) as f:
                self.gt_original = json.load(f)

            if self.language != 'en':
                with open(config.gt_file.replace('train', 'train_{}'.format(self.language))) as f:
                    self.gt = json.load(f)
            else:
                with open(config.gt_file) as f:
                    self.gt = json.load(f)

        else:
            with open(config.gt_eval_file) as f:
                self.gt_original = json.load(f)

            if self.language != 'en':
                with open(config.gt_eval_file.replace('eval', 'eval_{}'.format(self.language))) as f:
                    self.gt = json.load(f)
            else:
                with open(config.gt_eval_file) as f:
                    self.gt = json.load(f)

        print_ok('Done!\n')

        # TODO filter questions by max_len?

    def len(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.gt) / self.batch_size))

    def next(self):
        # select next batch idxs
        if self.shuffle:
            batch_idxs = np.random.choice(len(self.gt), self.batch_size)
        else:
            if self.curr_idx + self.batch_size > len(self.gt): self.curr_idx = 0
            batch_idxs = range(self.curr_idx, self.curr_idx + self.batch_size)
            self.curr_idx = (self.curr_idx + self.batch_size) % len(self.gt)

        batch_x_image = []
        batch_x_textual = np.zeros((self.batch_size, 38, 38, self.dim_txt))  # TODO do not hardcode contants
        batch_x_questions = np.zeros((self.batch_size, self.max_len, self.dim_txt))
        batch_y = np.zeros((self.batch_size, 38, 38), dtype=np.int8)

        if not self.training:
            batch_ocr = np.chararray((self.batch_size, 38, 38), itemsize=35)
            batch_ocr[:] = ''
            gt_questions = []
            gt_answers = []
            gt_ids = []
            batch_ocr_original = np.chararray((self.batch_size, 38, 38), itemsize=35)
            batch_ocr_original[:] = ''

        # foreach question in batch
        for i, idx in enumerate(batch_idxs):

            # load image
            # print os.path.join(self.image_path, self.gt[idx]['file_path'])
            image = cv2.imread(os.path.join(self.image_path, self.gt[idx]['file_path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # get fasttext vectors and bboxes for all ocr_tokens
            gt_boxes = [w['bbox'] for w in self.gt[idx]['ocr_bboxes']]
            if self.embedding_type == 'fasttext':
                gt_text_vectors = [self.txt_model.get_word_vector(w['text']) for w in self.gt[idx]['ocr_bboxes']]
            else:
                gt_text_vectors = [self.txt_model.embed(w['text']) for w in self.gt[idx]['ocr_bboxes']]
            gt_texts = [w['text'] for w in self.gt[idx]['ocr_bboxes']]

            if self.language in ['ca', 'es']:
                gt_texts_original = [w['text'] for w in self.gt_original[idx]['ocr_bboxes']]

            # store indexes of those bboxes wich are the answer
            gt_ans_boxes = [w['bbox'] for w in self.gt[idx]['ans_bboxes']]
            gt_ans_idxs = [gt_boxes.index(b) for b in gt_ans_boxes]
            gt_boxes = np.array(gt_boxes)

            # TODO data augmentation?

            # preprocess image
            if len(gt_boxes) > 0:
                image_data, gt_boxes = yolo_image_preprocess(image,
                                                             [self.input_size, self.input_size],
                                                             gt_boxes=gt_boxes)
            else:
                image_data = yolo_image_preprocess(image, [self.input_size, self.input_size])

                gt_boxes = np.array(())
            batch_x_image.append(image_data)

            # assign fasttext vectors to cells in a 38x38 grid
            for w in range(gt_boxes.shape[0]):
                cell_coords = gt_boxes[w, :] // 16  # TODO do not hardcode constants
                batch_x_textual[i, cell_coords[1]:cell_coords[3] + 1, cell_coords[0]:cell_coords[2] + 1, :] = \
                    gt_text_vectors[w]

                if w in gt_ans_idxs:
                    batch_y[i, cell_coords[1]:cell_coords[3] + 1, cell_coords[0]:cell_coords[2] + 1] = 1
                if not self.training:
                    batch_ocr[i, cell_coords[1]:cell_coords[3] + 1, cell_coords[0]:cell_coords[2] + 1] = gt_texts[w]

                    if self.language in ['ca', 'es']:
                        batch_ocr_original[i, cell_coords[1]:cell_coords[3] + 1,
                        cell_coords[0]:cell_coords[2] + 1] = gt_texts_original[w]

            # question encode with fasttext or bpemb
            question = self.gt[idx]['question']
            for w in range(self.max_len - len(question), self.max_len):
                if self.embedding_type == 'fasttext':
                    batch_x_questions[i, w, :] = self.txt_model.get_word_vector(
                        question[w - (self.max_len - len(question))])
                else:
                    if self.bpemb_subtype == 'multi':
                        batch_x_questions[i, w, :] = self.txt_model.embed(
                            question[w - (self.max_len - len(question))])[1]
                    else:
                        batch_x_questions[i, w, :] = self.txt_model.embed(
                            question[w - (self.max_len - len(question))])[0]

            # if not training return gt for evaluation
            if not self.training:
                gt_questions.append(' '.join(self.gt[idx]['question']))
                gt_answers.append(' '.join(self.gt[idx]['answer']))

                if 'question_id' in self.gt[idx].keys():
                    gt_ids.append(self.gt[idx]['question_id'])

        if self.training:
            return [batch_x_image, batch_x_textual, batch_x_questions, batch_y]
        else:
            return [batch_x_image, batch_x_textual, batch_x_questions, batch_y, batch_ocr, gt_questions, gt_answers,
                    gt_ids, batch_ocr_original]
