import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from models.olra import OLRA
from config.config_olra import Config
from dataloader.yolo_utils import olra_image_preprocess


def stvqa():
    config = Config().get_config()
    olra_model = OLRA(config)

    with open(config.gt_eval_file, 'r') as f:
        eval_data = json.load(f)

    with open(config.gt_file, 'r') as f:
        train_data = json.load(f)

    base_dir = 'data/ST-VQA'
    datasets = [eval_data, train_data]

    counter = 0
    for dataset in datasets:
        for entry in tqdm(dataset, "Processing dataset"):
            image = cv2.imread(os.path.join(base_dir, entry['file_path']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            image = olra_image_preprocess(image, [config.img_size, config.img_size])

            img_features = olra_model.resnet.predict_on_batch(np.expand_dims(image, axis=0))
            img_features = tf.keras.layers.GlobalAvgPool2D()(img_features)

            with open(os.path.join(base_dir, entry['file_path'].replace(entry['file_path'].split('.')[-1], 'npy')),
                      'wb+') as f:
                np.save(f, img_features[0, :])
