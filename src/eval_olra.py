import json
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu

from dataloader.utils import print_info, print_ok, update_eval_progress_bar
from config.config_olra import Config
from models.olra import OLRA
from tqdm import tqdm


def calculate_blew_score(candidate, gt):

    bleu = sentence_bleu(' '.join(gt), ' '.join(candidate), weights=(1.0, 0, 0, 0))
    # bleu_1 = sentence_bleu(gt, candidate, weights=(1.0, 0, 0, 0))
    # bleu_2 = sentence_bleu(gt, candidate, weights=(0.5, 0.5, 0, 0))
    # bleu_3 = sentence_bleu(gt, candidate, weights=(0.33, 0.33, 0.33, 0))
    # bleu_4 = sentence_bleu(gt, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # return [bleu, bleu_1, bleu_2, bleu_3, bleu_4]
    return bleu


if __name__ == "__main__":
    config = Config().get_config()
    config.shuffle = False

    print_info('Building OLRA model')
    olra_model = OLRA(Config().get_config(), training=False)
    print_ok('Done\n')

    print_info('Starting evaluation \n')

    step = 0
    count = 0
    bleu_scores = {
        'total_scores': []
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

        bleu_score = calculate_blew_score(output, batch_data[3][0])
        bleu_scores[batch_data[5][0]] = {}
        bleu_scores[batch_data[5][0]]['gt'] = batch_data[3][0]
        bleu_scores[batch_data[5][0]]['generated'] = output
        bleu_scores[batch_data[5][0]]['bleu'] = bleu_score
        # bleu_scores[batch_data[5][0]]['bleu_1'] = bleu_score[1]
        # bleu_scores[batch_data[5][0]]['bleu_2'] = bleu_score[2]
        # bleu_scores[batch_data[5][0]]['bleu_3'] = bleu_score[3]
        # bleu_scores[batch_data[5][0]]['bleu_4'] = bleu_score[4]

        bleu_scores['total_scores'] = bleu_score

    bleu_scores['mean_score'] = np.mean(np.array(bleu_scores['total_scores']))

    with open('eval_out_{}.json'.format(config.output_folder), 'w+') as f:
        json.dump(bleu_scores, f)
