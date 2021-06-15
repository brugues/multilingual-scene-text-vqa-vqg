import os

import tensorflow as tf
from models.attention import Attention


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.validation_log_dir = os.path.join(log_dir, 'val')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.validation_summary_writer = tf.summary.create_file_writer(self.validation_log_dir)


class VQAModel:

    def __init__(self, config, training=True):
        self.config = config
        self.training = training
        self.attention_model = Attention(config, training=self.training)
        self.attention_model.build_model()
        self.yolo_model = self.load_yolo()
        self.models_path = config.models_path
        self.logging_path = config.logging_path

        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        if not os.path.isdir(self.logging_path):
            os.makedirs(self.logging_path)

        if training:
            self.experiment = '00{}'.format(len(os.listdir(self.models_path)) + 1)
            self.models_path = os.path.join(self.models_path, self.experiment)
            self.logging_path = os.path.join(self.logging_path, self.experiment)

            os.makedirs(os.path.join(self.models_path), exist_ok=True)
            os.makedirs(os.path.join(self.models_path, 'checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'val'), exist_ok=True)

            if config.load_checkpoint:
                folders = os.listdir(self.models_path)
                folders.sort()
                previous_model_folder = folders[-2]

                self.attention_model.keras_model.load_weights(os.path.join(previous_model_folder,
                                                                           'checkpoints', 'ckpt'))
        else:
            self.attention_model.keras_model.load_weights(os.path.join(config.model_to_evaluate,
                                                                       'checkpoints', 'ckpt'))

        self.tensorboard = TensorBoardLogger(self.logging_path)

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.lr,
                                                                        decay_rate=config.decay_factor,
                                                                        decay_steps=config.decay_steps,
                                                                        staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)

    def load_yolo(self):
        model = tf.keras.models.load_model(self.config.yolo_file)

        return tf.keras.Model(model.input,
                              model.get_layer('conv2d_96').output,
                              name='vqa_yolo_feature_extractor_module')

    @tf.function
    def attention_train_step(self, img_features, txt_features, question, labels):

        image_emb = tf.concat([img_features, txt_features], 3)

        with tf.GradientTape() as tape:
            model_outputs = self.attention_model.keras_model([question, image_emb],
                                                             training=True)
            # if self.loss is not None:
            #     loss = self.loss(labels, model_outputs)
            # else:
            #     loss = tf.compat.v1.losses.sigmoid_cross_entropy(labels, model_outputs)
            loss = self.loss(labels, model_outputs)

        # Apply gradients
        gradients = tape.gradient(loss, self.attention_model.keras_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.attention_model.keras_model.trainable_variables))

        return loss

    def attention_eval_step(self, img_features, txt_features, question, labels):
        image_emb = tf.concat([img_features, txt_features], 3)
        model_outputs = self.attention_model.keras_model.predict_on_batch([question, image_emb])

        return model_outputs

    def save_attention_checkpoint(self):
        path = os.path.join(self.models_path, 'checkpoints', 'ckpt')
        self.attention_model.keras_model.save_weights(path)

    def save_attention_model(self):
        self.attention_model.keras_model.save(os.path.join(self.models_path, 'model.h5'))

    def log_to_tensorboard(self, step, loss, task='train'):
        def _decayed_learning_rate():
            return self.config.lr * self.config.decay_factor ** (step / self.config.decay_steps)

        if task == 'train':
            with self.tensorboard.train_summary_writer.as_default():
                # Loss
                tf.summary.scalar('loss', loss, step=step)

                # Learning rate
                tf.summary.scalar('lr', _decayed_learning_rate(), step=step)

        elif task == 'val':
            pass
            # with self.tensorboard.train_summary_writer.as_default():
            #     # Loss
            #     tf.summary.scalar('loss', loss, step=step)
