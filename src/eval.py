import six
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from Levenshtein import distance as lev

from dataloader.data_loader import VQADataGenerator
from src.dataloader.utils import print_info, print_ok, update_eval_progress_bar
from config.config import Config
from models.stvqa import VQAModel


def run_evaluation_batch(model, data):
    # Extract image features from Yolo
    img_features = model.yolo_model.predict_on_batch(np.array(data[0]))
    img_features = tf.cast(img_features, tf.float64)

    out = model.attention_eval_step(img_features,
                                    data[1],
                                    data[2],
                                    data[3])

    return out


def get_anls_score(gt, predictions, threshold=0.5):
    scores = []

    for i, prediction in enumerate(predictions):
        word = prediction['answer']
        gt_word = gt[i]['answer']
        l_score = lev(word, gt_word) / (len(word) + len(gt_word))

        if l_score >= threshold:
            s = 0
        else:
            s = 1 - l_score

        scores.append(s)

    return np.mean(scores)


if __name__ == '__main__':

    config = Config().get_config()
    config.shuffle = False

    print_info('Building Attention model ...')
    stvqa_model = VQAModel(config, training=False)
    stvqa_model.yolo_model.summary()
    stvqa_model.attention_model.keras_model.summary()
    print_ok('Done!\n')

    print_info('Preparing data generator')
    eval_data_generator = VQADataGenerator(config, training=False)
    print_ok('Done!\n')

    print_info('Starting evaluation \n')

    count = 0
    step = 0
    eval_out = []
    num_batches = eval_data_generator.len()

    progress_bar = tqdm(range(num_batches),
                        total=num_batches, desc='Evaluation Progress')

    for batch in range(num_batches):

        batch_data = eval_data_generator.next()
        this_batch_size = len(batch_data[0])
        count += this_batch_size

        if this_batch_size != config.batch_size:  # deal with last batch
            batch_data[0] = np.resize(np.array(batch_data[0]), (config.batch_size, config.img_size, config.img_size, 3))
            batch_data[1] = np.resize(batch_data[1], (config.batch_size, 38, 38, config.text_embedding_dim))
            batch_data[2] = np.resize(batch_data[2], (config.batch_size, config.max_len, config.text_embedding_dim))

        stvqa_output = run_evaluation_batch(stvqa_model, batch_data)
        stvqa_output = tf.math.sigmoid(stvqa_output)
        stvqa_output = stvqa_output.numpy()

        if config.language in ['ca', 'es']:
            batch_ocr = batch_data[8]
        else:
            batch_ocr = batch_data[4]

        gt_ids = batch_data[7]

        for b in range(this_batch_size):
            one_pred = []
            all_pred = []
            cmb_pred = []
            best_prob = 0.
            for i in range(config.num_grids):
                for j in range(config.num_grids):
                    if stvqa_output[b, i, j] > 0.5 and batch_ocr[b, i, j] not in all_pred and batch_ocr[b, i, j] != '':
                        all_pred.append(batch_ocr[b, i, j])
                    if stvqa_output[b, i, j] > best_prob:
                        best_prob = stvqa_output[b, i, j]
                        one_pred = [batch_ocr[b, i, j]]
                    if stvqa_output[b, i, j] > 0.95 and batch_ocr[b, i, j] not in cmb_pred and batch_ocr[b, i, j] != '':
                        cmb_pred.append(batch_ocr[b, i, j])

            prediction = ""
            if len(cmb_pred) > 0:
                cmb_pred_alt = []
                for element in cmb_pred:
                    if isinstance(element, six.binary_type):
                        element = element.decode("utf-8")
                    cmb_pred_alt.append(element)
                prediction = ' '.join(cmb_pred_alt)
            else:
                one_pred_alt = []
                for element in one_pred:
                    if isinstance(element, six.binary_type):
                        element = element.decode("utf-8")
                    one_pred_alt.append(element)
                prediction = ' '.join(one_pred_alt)

            eval_out.append({'answer': prediction, 'question_id': int(gt_ids[b])})

        update_eval_progress_bar(progress_bar, batch, num_batches)
        step += 1

    if not config.server_evaluation:
        anls = get_anls_score(eval_data_generator.gt, eval_out)
        eval_out.insert(0, {'anls': anls})

    with open('eval_out_{}.json'.format(config.output_folder), 'w') as f:
        json.dump(eval_out, f)
