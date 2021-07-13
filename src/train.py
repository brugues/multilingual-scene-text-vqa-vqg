import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from config.config import Config
from models.stvqa import VQAModel
from dataloader.utils import print_info, print_ok, update_train_progress_bar
from dataloader.data_loader import VQADataGenerator


if __name__ == '__main__':

    config = Config().get_config()

    print_info('Building Attention model ...')
    stvqa_model = VQAModel(config, training=True)
    stvqa_model.yolo_model.summary()
    stvqa_model.attention_model.keras_model.summary()
    print_ok('Done!\n')

    print_info('Preparing data generator\n')
    train_data_generator = VQADataGenerator(config)

    print_ok('Done!\n')

    print_info('Starting training \n')

    step = 0
    num_batches = train_data_generator.len()

    for epoch in range(config.n_epochs):
        print_ok('Starting epoch number {}\n'.format(epoch))

        #progress_bar = tqdm(range(num_batches),
        #                    total=num_batches, desc='Training Progress')

        for batch in range(num_batches):
            # Get data for this batch
            batch_data = train_data_generator.next()

            # Extract image features from Yolo
            img_features = stvqa_model.yolo_model.predict_on_batch(np.array(batch_data[0]))
            img_features = tf.cast(img_features, tf.float64)

            # Train step
            loss = stvqa_model.attention_train_step(img_features, batch_data[1], batch_data[2], batch_data[3])

            if batch % config.logging_period == 0:
                stvqa_model.log_to_tensorboard(step, loss)
                print("End of step {}/{} of epoch {}. Loss: {}".format(batch + 1, num_batches, epoch + 1, loss))

            if batch % config.checkpoint_period == 0:
                stvqa_model.save_attention_checkpoint()

            #update_train_progress_bar(progress_bar, epoch, config.n_epochs, batch, num_batches, loss)

            step += 1

        stvqa_model.save_attention_checkpoint()
        stvqa_model.save_attention_model()
        tf.keras.backend.clear_session()  # Just in case there a memory leak or sth
        #del progress_bar
