import numpy as np
from tqdm import tqdm
import tensorflow as tf
from config.config_olra import Config
from models.olra import OLRA, RNN_Decoder
from dataloader.utils import print_info, print_ok

if __name__ == '__main__':
    config = Config().get_config()
    print(config.__str__())

    print_info('Building OLRA model\n')
    olra_model = OLRA(Config().get_config())
    print_ok('Done\n')

    step = 0
    num_batches = olra_model.data_generator.len()

    for epoch in range(config.n_epochs):
        print_ok('Starting epoch number {}\n'.format(epoch))

        for batch in tqdm(range(num_batches)):
            batch_data = olra_model.data_generator.next()

            # Extract image features from ResNet
            if batch_data[6]:
                img_features = batch_data[5]
            else:
                img_features = olra_model.resnet.predict_on_batch(np.array(batch_data[0]))
                img_features = tf.keras.layers.GlobalAvgPool2D()(img_features)

            losses = olra_model.olra_train_step(batch_data[1], img_features, batch_data[2], batch_data[4])

            if batch % config.logging_period == 0:
                olra_model.log_to_tensorboard(step, losses)
                print("End of step {}/{} of epoch {}. Loss: {}".format(batch + 1,
                                                                       num_batches,
                                                                       epoch + 1, losses[0]))

            if batch % config.checkpoint_period == 0:
                olra_model.save_olra_checkpoint()

            step += 1

        olra_model.save_olra_checkpoint()
        tf.keras.backend.clear_session()
