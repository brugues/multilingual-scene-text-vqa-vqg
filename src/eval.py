import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.data_loader import STVQADataGenerator
from utils.utils import print_info, print_ok, update_eval_progress_bar
from config.config import Config
from models.stvqa import VQAModel


def evaluate_batch(model, data):
    # Extract image features from Yolo
    img_features = model.yolo_model.predict_on_batch(np.array(data[0]))
    img_features = tf.cast(img_features, tf.float64)

    out, loss = model.attention_eval_step(img_features,
                                          data[1],
                                          data[2],
                                          data[3])

    return out, loss


if __name__ == '__main__':

    config = Config().get_config()

    print_info('Building Attention model ...')
    stvqa_model = VQAModel(config, training=False)
    stvqa_model.yolo_model.summary()
    stvqa_model.attention_model.keras_model.summary()
    print_ok('Done!\n')

    print_info('Preparing data generator')
    eval_data_generator = STVQADataGenerator(config, training=False)
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

        stvqa_output, _ = evaluate_batch(stvqa_model, batch_data)

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

            prediction = ''
            if len(cmb_pred) > 0:
                for element in cmb_pred:
                    prediction.join(str(element))
            else:
                for element in one_pred:
                    prediction.join(str(element))

            eval_out.append({'answer': prediction, 'question_id': int(gt_ids[b])})

        update_eval_progress_bar(progress_bar, batch, num_batches)
        step += 1

    with open('eval_out.json', 'w') as f:
        json.dump(eval_out, f)