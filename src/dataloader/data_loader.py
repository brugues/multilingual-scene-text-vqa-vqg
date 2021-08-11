import io
import os
import json

import fasttext
from bpemb import BPEmb
import cv2
import numpy as np
import tensorflow as tf

from dataloader.utils import print_info, print_ok
from dataloader.yolo_utils import vqa_image_preprocess, olra_image_preprocess


def load_fasttext_transform(fname, d1=300, d2=300):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    return R


def load_embeddings(config):
    txt_models, transformations = {}, {}

    if config.embedding_type == 'fasttext':
        fasttext_path = os.path.join(config.txt_embeddings_path, 'fasttext')

        if 'en' in config.language or 'ca' in config.language or 'es' in config.language or 'zh' in config.language:
            if config.fasttext_subtype == 'wiki':
                try:
                    languages = config.language.split('-')
                    for language in languages:
                        txt_models[language] = fasttext.load_model(os.path.join(fasttext_path, 'wiki_word_vectors',
                                                                                'wiki.{}.bin'.format(language)))

                except:
                    print("Error loading model. Check that the file exists")
            elif config.fasttext_subtype == 'wiki-news':
                try:
                    print("Be aware! This model is only available in english, so the english model is being loaded")
                    txt_models[config.language] = fasttext.load_model(os.path.join(config.txt_embeddings_path,
                                                                                   'wiki-news-300d-1M-subword.bin'))
                except:
                    print("Error loading model. Check that the file exists. ")
            elif config.fasttext_subtype == 'cc':
                try:
                    languages = config.language.split('-')
                    for language in languages:
                        txt_models[language] = fasttext.load_model(os.path.join(fasttext_path, 'cc_word_vectors',
                                                                                'cc.{}.300.bin'.format(language)))

                except:
                    print("Error loading model. Check that the file exists")
            else:
                raise AttributeError('Invalid fasttext subtype. Choose between wiki, wiki-news, cc or aligned')

        else:
            raise AttributeError('Invalid language. Select between English (en), Catalan (ca) or Spanish (es)')

        if config.fasttext_aligned:
            base_path = 'fastText/alignment/res'
            languages = config.language.split('-')
            for language in languages:
                if language == 'en':
                    transformations[language] = np.ones((config.dim_txt, config.dim_txt))
                else:
                    transformations[language] = \
                        load_fasttext_transform(os.path.join(base_path,
                                                             '{}.{}.vec-mat'.format(config.fasttext_subtype,
                                                                                    config.fasttext_aligned_pair)))

    elif config.embedding_type == 'bpemb':
        bpemb_path = os.path.join(config.txt_embeddings_path, 'bpemb')
        if 'en' in config.language or 'ca' in config.language or 'es' in config.language or 'zh' in config.language:

            if config.bpemb_subtype == 'multi':
                bin_file = os.path.join(bpemb_path, 'multi.wiki.bpe.vs1000000.d300.w2v.bin')
                model_file = os.path.join(bpemb_path, 'multi.wiki.bpe.vs1000000.model')
            else:
                bin_file = os.path.join(bpemb_path, '{}.wiki.bpe.vs200000.d300.w2v.bin'.format(config.language))
                model_file = os.path.join(bpemb_path, '{}.wiki.bpe.vs200000.model'.format(config.language))

            assert str(config.dim_txt) in bin_file
            try:
                languages = config.language.split('-')
                for language in languages:
                    txt_models[language] = BPEmb(emb_file=bin_file,
                                                 model_file=model_file,
                                                 dim=config.dim_txt)
            except:
                print("Error loading model. Check that the file exists")

        else:
            raise AttributeError('Invalid language. Select between English (en), Catalan (ca) or Spanish (es) or '
                                 'All of them (multi)')

    elif config.embedding_type == 'smith':
        smith_path = os.path.join(config.txt_embeddings_path, 'fasttext', 'wiki_word_vectors')
        try:
            languages = config.language.split('-')
            for language in languages:
                txt_models[language] = fasttext.load_model(os.path.join(smith_path,
                                                                        'wiki.{}.bin'.format(language)))
        except:
            print("Error loading model. Check that the file exists")

        languages = config.language.split('-')
        for language in languages:
            transformations[language] = np.loadtxt(os.path.join(config.txt_embeddings_path, 'smith',
                                                                'transformations',
                                                                '{}.txt'.format(language)))

    else:
        raise AttributeError('Invalid embedding type. Select either fasttext or bpemb')

    return txt_models, transformations


def load_gt(config, training):
    if config.dataset == 'stvqa':
        pivot_lang = 'en'
    elif config.dataset == 'estvqa':
        pivot_lang = 'zh'
    else:
        raise AttributeError('Invalid dataset type. Options are stvqa and estvqa')

    if training:
        with open(config.gt_file) as f:
            gt_original = json.load(f)

        if config.language != pivot_lang:
            lang = config.language
            if '-' in lang:
                lang = lang.replace('-', '_')
            with open(config.gt_file.replace('train', 'train_{}'.format(lang))) as f:
                gt = json.load(f)
        else:
            with open(config.gt_file) as f:
                gt = json.load(f)

    else:
        with open(config.gt_eval_file) as f:
            gt_original = json.load(f)

        if config.language != pivot_lang:
            lang = config.language
            if '-' in lang:
                lang = lang.replace('-', '_')
            with open(config.gt_eval_file.replace('eval', 'eval_{}'.format(lang))) as f:
                gt = json.load(f)
        else:
            with open(config.gt_eval_file) as f:
                gt = json.load(f)

    return gt_original, gt


class VQADataGenerator:

    def __init__(self, config, training=True):
        self.config = config
        self.training = training
        self.dataset = config.dataset
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

        self.txt_models, self.transformations = load_embeddings(self.config)

        # load gt file
        print_info('Loading GT file...')
        self.gt_original, self.gt = load_gt(self.config, training)
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
        batch_x_textual = np.zeros((self.batch_size, 38, 38, self.dim_txt))  # TODO do not hardcode constants
        batch_x_questions = np.zeros((self.batch_size, self.max_len, self.dim_txt))
        batch_y = np.zeros((self.batch_size, 38, 38), dtype=np.int8)

        if not self.training:
            batch_ocr = np.chararray((self.batch_size, 38, 38), itemsize=35, unicode=True)
            batch_ocr[:] = ''
            gt_questions = []
            gt_answers = []
            gt_ids = []
            batch_ocr_original = np.chararray((self.batch_size, 38, 38), itemsize=35, unicode=True)
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
                gt_text_vectors = [self.txt_models[self.gt[idx]['lang']].get_word_vector(w['text'])
                                   for w in self.gt[idx]['ocr_bboxes']]

                if self.config.fasttext_aligned and len(gt_text_vectors) > 0:
                    gt_text_vectors = np.dot(gt_text_vectors, self.transformations[self.gt[idx]['lang']].T)

            elif self.embedding_type == 'smith':
                gt_text_vectors = [np.matmul(self.txt_models[self.gt[idx]['lang']].get_word_vector(w['text']),
                                             self.transformations[self.gt[idx]['lang']])
                                   for w in self.gt[idx]['ocr_bboxes']]

            else:
                gt_text_vectors = []
                for w in self.gt[idx]['ocr_bboxes']:
                    if len(w['text']) > 0:
                        gt_text_vectors.append(self.txt_models[self.gt[idx]['lang']].embed(w['text'])[-1, :])
                    else:
                        gt_text_vectors.append(np.zeros(300, dtype=np.float32))

                # for w in self.gt[idx]['ocr_bboxes']:
                #     print(w['text'] + str(self.txt_model.embed(w['text']).shape))
                # gt_text_vectors = [self.txt_model.embed(w['text'])[-1, :] if w is not None else np.zeros(300)
                #                    for w in self.gt[idx]['ocr_bboxes']]
            gt_texts = [w['text'] for w in self.gt[idx]['ocr_bboxes']]

            if self.language in ['ca', 'es', 'zh']:
                gt_texts_original = [w['text'] for w in self.gt_original[idx]['ocr_bboxes']]

            # store indexes of those bboxes wich are the answer
            gt_ans_boxes = [w['bbox'] for w in self.gt[idx]['ans_bboxes']]
            gt_ans_idxs = [gt_boxes.index(b) for b in gt_ans_boxes]
            gt_boxes = np.array(gt_boxes)

            # --language zh --batch_size 32 --dataset estvqa --image_path data/EST-VQA-v1.0 --gt_file data/EST-VQA-v1.0/annotations/train_chinese_subsample_all_answers.json --gt_eval_file data/EST-VQA-v1.0/annotations/eval_chinese_subsample_all_answers.json

            # TODO data augmentation?

            # preprocess image
            if len(gt_boxes) > 0:
                image_data, gt_boxes = vqa_image_preprocess(image,
                                                            [self.input_size, self.input_size],
                                                            gt_boxes=gt_boxes)
            else:
                image_data = vqa_image_preprocess(image, [self.input_size, self.input_size])

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

                    if self.language in ['ca', 'es', 'zh']:
                        batch_ocr_original[i, cell_coords[1]:cell_coords[3] + 1,
                        cell_coords[0]:cell_coords[2] + 1] = gt_texts_original[w]

            # question encode with fasttext or bpemb
            question = self.gt[idx]['question']
            for w in range(self.max_len - len(question), self.max_len):
                if self.embedding_type == 'fasttext':
                    batch_x_questions[i, w, :] = self.txt_models[self.gt[idx]['lang']].get_word_vector(
                        question[w - (self.max_len - len(question))])

                    if self.config.fasttext_aligned:
                        batch_x_questions[i, w, :] = np.dot(batch_x_questions[i, w, :],
                                                            self.transformations[self.gt[idx]['lang']].T)

                elif self.embedding_type == 'smith':
                    batch_x_questions[i, w, :] = np.matmul(self.txt_models[self.gt[idx]['lang']].get_word_vector(
                        question[w - (self.max_len - len(question))]), self.transformations[self.gt[idx]['lang']])

                else:
                    emb = self.txt_models[self.gt[idx]['lang']].embed(question[w - (self.max_len - len(question))])
                    if emb.shape[0] == self.dim_txt:
                        batch_x_questions[i, w, :] = emb
                    elif emb.shape[0] > 0:
                        batch_x_questions[i, w, :] = emb[-1, :]
                    else:
                        batch_x_questions[i, w, :] = np.zeros(300)

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


class OLRADataGenerator:
    def __init__(self, config, training=True) -> None:
        self.config = config
        self.training = training
        self.dataset = self.config.dataset
        self.batch_size = self.config.batch_size
        self.shuffle = self.config.shuffle
        self.max_len = self.config.max_len
        self.image_path = self.config.image_path
        self.dim_txt = self.config.dim_txt
        self.curr_idx = 0
        self.input_size = self.config.img_size
        self.language = self.config.language
        self.embedding_type = self.config.embedding_type
        self.fasttext_subtype = self.config.fasttext_subtype
        self.bpemb_subtype = self.config.bpemb_subtype

        self.txt_models, self.transformations = load_embeddings(self.config)

        # load gt file
        print_info('Loading GT file...')
        self.gt_original, self.gt = load_gt(self.config, training)
        print_ok('Done!\n')

        for i, entry in enumerate(self.gt_original):
            if len(entry['answer']) == 1:
                self.gt_original.pop(i)
                self.gt.pop(i)

        self.vocabulary = []
        for entry in self.gt:
            sentence = '<{}>'.format(entry['lang'])
            for word in entry['question']:
                sentence += ' {}'.format(word)
            sentence += ' <end>'
            self.vocabulary.append(sentence)

        self.top_k = self.config.vocabulary_size
        os.makedirs(self.config.tokenizer_path, exist_ok=True)
        tokenizer_file = os.path.join(self.config.tokenizer_path,
                                      'olra_{}_{}_{}.json'.format(self.config.embedding_type,
                                                                  self.config.language,
                                                                  self.top_k))

        # Tokenize vocabulary
        if os.path.isfile(tokenizer_file):
            with open(tokenizer_file, 'r') as f:
                self.tokenizer = json.load(f)
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(self.tokenizer)
        else:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.top_k,
                                                                   oov_token="<unk>",
                                                                   filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
            self.tokenizer.fit_on_texts(self.vocabulary)

            self.tokenizer.word_index['<pad>'] = 0
            self.tokenizer.index_word[0] = '<pad>'

            tokenizer_json = self.tokenizer.to_json()

            with open(tokenizer_file, 'w+') as f:
                json.dump(tokenizer_json, f)

        # Create the tokenized vectors
        self.vocabulary_seqs = self.tokenizer.texts_to_sequences(self.vocabulary)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        self.cap_vector = tf.keras.preprocessing.sequence.pad_sequences(self.vocabulary_seqs,
                                                                        padding='post')
        # Calculates the max_length, which is used to store the attention weights
        self.max_len = self.calc_max_length(self.vocabulary_seqs)

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def len(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """

        return int(np.floor(len(self.gt) / self.batch_size))

    def next(self):
        # select next batch idxs
        if self.shuffle:
            batch_idxs = np.random.choice(len(self.gt), self.batch_size)
        else:
            if self.curr_idx + self.batch_size > len(self.gt):
                self.curr_idx = 0
            batch_idxs = range(self.curr_idx, self.curr_idx + self.batch_size)
            self.curr_idx = (self.curr_idx + self.batch_size) % len(self.gt)

        filenames = []
        batch_x_image = []
        batch_x_vector = np.zeros((self.batch_size, self.dim_txt))
        batch_x_position = np.zeros((self.batch_size, 4))
        batch_x_question = []
        batch_x_question_embed = np.zeros((self.batch_size, self.max_len))

        # foreach question in batch
        for i, idx in enumerate(batch_idxs):
            # load image
            # print os.path.join(self.image_path, self.gt[idx]['file_path'])
            filenames.append(self.gt[idx]['file_path'])
            image = cv2.imread(os.path.join(self.image_path, self.gt[idx]['file_path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            # store indexes of those bboxes wich are the answer
            gt_ans_boxes = [w['bbox'] for w in self.gt[idx]['ans_bboxes']]

            image, gt_ans_boxes = olra_image_preprocess(image,
                                                        [self.input_size, self.input_size],
                                                        gt_boxes=np.array(gt_ans_boxes))

            batch_x_image.append(image)
            batch_x_position[i] = gt_ans_boxes[0] / 640.
            batch_x_question.append(self.gt[idx]['question'])

            if self.embedding_type == 'fasttext':
                batch_x_vector[i, :] = self.txt_models[self.gt[idx]['lang']].get_word_vector(self.gt[idx]['answer'][0])

                if self.config.fasttext_aligned:
                    batch_x_vector[i, :] = np.dot(batch_x_vector[i, :], self.transformations[self.gt[idx]['lang']].T)

            elif self.embedding_type == 'smith':
                batch_x_vector[i, :] = np.matmul(
                    self.txt_models[self.gt[idx]['lang']].get_word_vector(self.gt[idx]['answer'][0]),
                    self.transformations[self.gt[idx]['lang']])


            else:
                emb = self.txt_models[self.gt[idx]['lang']].embed(self.gt[idx]['answer'][0])
                if emb.shape[0] == self.dim_txt:
                    batch_x_vector[i, :] = emb
                elif emb.shape[0] > 0:
                    batch_x_vector[i, :] = emb[-1, :]
                else:
                    batch_x_vector[i, :] = np.zeros(300)

            if self.training:
                batch_x_question_embed[i] = self.cap_vector[idx]
            else:
                batch_x_question_embed[i] = tf.expand_dims(
                    [self.tokenizer.word_index['<{}>'.format(self.gt[idx]['lang'])]],
                    0)

        batch_x = np.array(batch_x_image)
        batch_x_image = tf.keras.applications.resnet_v2.preprocess_input(batch_x)
        batch_x_image = batch_x_image.astype(np.float32)

        return [batch_x_image, batch_x_vector, batch_x_position, batch_x_question, batch_x_question_embed, filenames]
