import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from config.config import Config
from models.attention import TF2_Attention
from utils.data_loader import *
from utils.utils import *


@tf.function
def train_step():
    pass


if __name__ == '__main__':

    config = Config().get_config()

    print_info('Loading YOLO model...')
    yolo_inout = ["input/input_data:0", "conv61/Conv2D:0"]
    # input_images, image_features = yolo_read_pb_return_tensors(config.yolo_file, yolo_inout)
    print_ok('Done!\n')

    print_info('Building Attention model ...')
    attention_model = TF2_Attention(config, training=True)
    attention_model.build_model()
    print_ok('Done!\n')

    attention_model.keras_model.summary()

    for epoch in range(config.n_epochs):
        print_ok('Starting epoch number {}\n'.format(epoch))


