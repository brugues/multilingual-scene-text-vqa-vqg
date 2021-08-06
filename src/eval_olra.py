import json
import numpy as np
import tensorflow as tf
import tensorflow_text as text
from nltk.translate.bleu_score import sentence_bleu

from dataloader.utils import print_info, print_ok, update_eval_progress_bar
from config.config_olra import Config
from models.olra import OLRA
from tqdm import tqdm


def calculate_sentence_similarities(candidate, gt):

    bleu = sentence_bleu([gt], candidate)
    hypothesis = tf.ragged.constant([candidate])
    references = tf.ragged.constant([gt])
    rouge = text.metrics.rouge_l(hypothesis, references)

    return [bleu, rouge]


if __name__ == "__main__":
    config = Config().get_config()
    config.shuffle = False

    print_info('Building OLRA model')
    olra_model = OLRA(Config().get_config(), training=False)
    print_ok('Done\n')

    print_info('Starting evaluation \n')

    step = 0
    count = 0
    scores = {
        'bleu': [],
        'rouge_p': [],
        'rouge_f': [],
        'rouge_r': []
    }
    num_batches = olra_model.data_generator.len()

    for batch in tqdm(range(num_batches)):
        batch_data = olra_model.data_generator.next()
        this_batch_size = len(batch_data[0])
        count += this_batch_size

        # Extract image features from ResNet
        img_features = olra_model.resnet.predict_on_batch(np.array(batch_data[0]))
        img_features = tf.keras.layers.GlobalAvgPool2D()(img_features)

        output = olra_model.evaluation_step(batch_data[1], img_features, batch_data[2], batch_data[4])

        similarity = calculate_sentence_similarities(output, batch_data[3][0])
        scores[batch_data[5][0]] = {}
        scores[batch_data[5][0]]['gt'] = batch_data[3][0]
        scores[batch_data[5][0]]['generated'] = output
        scores[batch_data[5][0]]['bleu'] = similarity[0]
        scores[batch_data[5][0]]['rouge_f'] = similarity[1].f_measure.numpy()[0]
        scores[batch_data[5][0]]['rouge_p'] = similarity[1].p_measure.numpy()[0]
        scores[batch_data[5][0]]['rouge_r'] = similarity[1].r_measure.numpy()[0]
        scores['bleu'].append(scores[batch_data[5][0]]['bleu'])
        scores['rouge_f'] = scores[batch_data[5][0]]['rouge_f']
        scores['rouge_p'] = scores[batch_data[5][0]]['rouge_p']
        scores['rouge_r'] = scores[batch_data[5][0]]['rouge_r']

    scores['bleu'] = np.mean(np.array(scores['bleu']))

    with open('eval_out_{}.json'.format(config.output_folder), 'w+') as f:
        json.dump(scores, f)
