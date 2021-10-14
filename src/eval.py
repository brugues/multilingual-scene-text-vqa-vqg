import six
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from dataloader.data_loader import VQADataGenerator
from dataloader.utils import print_info, print_ok, update_eval_progress_bar
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


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'THRESHOLD': 0.5
    }


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_anls_score(gt, predictions):

    evaluationParams = default_evaluation_params()
    res_id_to_index = {int(r['question_id']): ix for ix, r in enumerate(predictions)}
    perSampleMetrics = {}

    totalScore = 0
    row = 0
    for gtObject in gt:

        q_id = int(gtObject['question_id'])
        res_ix = res_id_to_index[q_id]
        detObject = predictions[res_ix]

        values = []
        for answer in gtObject['answer']:
            dist = levenshtein_distance(answer.lower(), detObject['answer'].lower())
            length = max(len(answer.upper()), len(detObject['answer'].upper()))
            values.append(0.0 if length == 0 else float(dist) / float(length))

        question_result = 1 - min(values)

        if question_result < evaluationParams['THRESHOLD']:
            question_result = 0

        totalScore += question_result

        perSampleMetrics[str(gtObject['question_id'])] = {
            'score': question_result,
            'question': gtObject['question'],
            'gt': gtObject['answer'],
            'det': detObject['answer']
        }
        row = row + 1

    methodMetrics = {
        'score': 0 if len(gt) == 0 else totalScore / len(gt)
    }

    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics}
    return resDict


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

        if config.dataset == 'stvqa':
            if config.language != 'en':
                batch_ocr = batch_data[8]
            else:
                batch_ocr = batch_data[4]
        else:
            # if config.language != 'zh':
            #     batch_ocr = batch_data[8]
            # else:
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
            # eval_out.append({'answer': prediction, 'question_id': int(0)})

        update_eval_progress_bar(progress_bar, batch, num_batches)
        step += 1

    if not config.server_evaluation:
        anls = get_anls_score(eval_data_generator.gt, eval_out)
        eval_out.insert(0, {'anls': anls['method']})
        eval_out.insert(1, {'anls_per_sample': anls['per_sample']})

    with open('eval_out_{}.json'.format(config.output_folder), 'w') as f:
        json.dump(eval_out, f)
