import numpy as np
from tqdm import tqdm
import tensorflow as tf

from config.config import Config
from models.stvqa import VQAModel
from utils.utils import print_info, print_ok, update_progress_bar
from utils.data_loader import STVQADataGenerator


if __name__ == '__main__':

    config = Config().get_config()

    print_info('Building Attention model ...')
    stvqa_model = VQAModel(config, training=True)
    stvqa_model.yolo_model.summary()
    stvqa_model.attention_model.keras_model.summary()
    print_ok('Done!\n')

    print_info('')
    data_generator = STVQADataGenerator(config, True)
    print_ok('Done!\n')

    print_info('Starting training \n')

    step = 0
    num_batches = data_generator.len()

    for epoch in range(config.n_epochs):
        print_ok('Starting epoch number {}\n'.format(epoch))

        progress_bar = tqdm(range(num_batches),
                            total=num_batches, desc='Training')

        for batch in range(num_batches):
            # Get data for this batch
            batch_data = data_generator.next()

            # Extract image features from Yolo
            img_features = stvqa_model.yolo_model.predict_on_batch(np.array(batch_data[0]))
            img_features = tf.cast(img_features, tf.float64)

            image_emb = tf.concat([img_features, batch_data[1]], 3)

            # Train step
            loss = stvqa_model.attention_train_step(img_features, batch_data[1], batch_data[2], batch_data[3])

            if batch % config.logging_period == 0:
                stvqa_model.log_to_tensorboard(step, loss)

            if batch % config.checkpoint_period == 0:
                stvqa_model.save_attention_checkpoint()

            update_progress_bar(progress_bar, epoch, config.n_epochs, batch, num_batches, loss)

            step += 1

        stvqa_model.save_attention_checkpoint()
        stvqa_model.save_attention_model()
