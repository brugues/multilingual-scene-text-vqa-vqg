import os

import tensorflow as tf
from dataloader.utils import print_info, print_ok
from tensorflow.python.ops.gen_batch_ops import batch
from models.stvqa import TensorBoardLogger
from dataloader.data_loader import OLRADataGenerator


class OLRA:
    def __init__(self, config, training=True) -> None:
        self.config = config
        self.training = training
        self.logging_path = config.logging_path
        self.models_path = config.models_path

        print_info('Preparing data generator\n')
        self.data_generator = OLRADataGenerator(config, training=training)
        print_ok('Done\n')

        for i in range(self.data_generator.len()):
            _ = self.data_generator.next()

        self.resnet = tf.keras.applications.ResNet50(include_top=False,
                                                     weights='imagenet',
                                                     input_shape=(self.config.img_size,
                                                                  self.config.img_size,
                                                                  3))
        self.resnet = tf.keras.Model(self.resnet.input,
                                     self.resnet.layers[170].output)
        self.keras_model = self.build_model()

        if not os.path.isdir(self.models_path):
            os.makedirs(self.models_path)

        if not os.path.isdir(self.logging_path):
            os.makedirs(self.logging_path)

        if self.training:
            if self.config.output_folder is None:
                self.experiment = '00{}'.format(len(os.listdir(self.models_path)) + 1)
            else:
                self.experiment = self.config.output_folder

            self.models_path = os.path.join(self.models_path, self.experiment)
            self.logging_path = os.path.join(self.logging_path, self.experiment)

            os.makedirs(os.path.join(self.models_path), exist_ok=True)
            os.makedirs(os.path.join(self.models_path, 'checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'val'), exist_ok=True)

        else:
            self.keras_model.load_weights(os.path.join(config.model_to_evaluate,
                                                       'checkpoints', 'ckpt'))

        self.tensorboard = TensorBoardLogger(self.logging_path)

        self.l2_loss = tf.keras.losses.MeanSquaredError()
        self.mle_loss = None

        if self.config.apply_decay:
            self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.lr,
                                                                            decay_rate=len(self.data_generator),
                                                                            decay_steps=config.decay_steps,
                                                                            staircase=config.staircase)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)

    def build_model(self):
        """
            Builds the OLRA keras model
        """

        fasttext_input = tf.keras.layers.Input(shape=self.config.dim_txt,
                                               batch_size=self.config.batch_size,
                                               dtype=tf.float32)

        image_input = tf.keras.layers.Input(shape=(self.config.img_size,
                                                   self.config.img_size,
                                                   3),
                                            batch_size=self.config.batch_size,
                                            dtype=tf.float32)

        ocr_positions_input = tf.keras.layers.Input(shape=4,
                                                    batch_size=self.config.batch_size,
                                                    dtype=tf.float32)

        # Get image features from resnet
        image_features = self.resnet(image_input)
        image_features = tf.keras.layers.GlobalAveragePooling2D()(image_features)

        # Get fasttext features from B-LSTM
        fasttext_features = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(512,
                                 activation='tanh',
                                 recurrent_activation='tanh',
                                 dropout=self.config.dropout)
        )(fasttext_input)

        # Multimodal Fusion
        fused_features = tf.keras.layers.concatenate([fasttext_features,
                                                      image_features,
                                                      ocr_positions_input])

        # OCR consistency module
        ocr_consistency = tf.keras.layers.Dense(self.config.dim_txt,
                                                activation=None)

        # Ask module (Question generator module)
        lstm = tf.keras.layers.LSTM()

        output_question = lstm(tf.ones(1))

        # Create final model
        model = tf.keras.Model(inputs=[fasttext_input,
                                       image_input,
                                       ocr_positions_input],
                               outputs=[ocr_consistency, output_question])

        return model

    @tf.function
    def olra_train_step(self, fasttext_features, images, ocr_posistions, labels):
        with tf.GradientTape() as tape:
            ocr_consistency, question = self.keras_model([fasttext_features,
                                                          images,
                                                          ocr_posistions])
            l2_loss = self.l2_loss(labels[0], ocr_consistency)
            mle_loss = 0
            loss = self.config.lambda_loss * l2_loss + mle_loss

        gradients = tape.gradient(loss, self.keras_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.keras_model.trainable_variables))

        return [loss, l2_loss, mle_loss]

    def save_olra_checkpoint(self):
        path = os.path.join(self.models_path, 'checkpoints', 'ckpt')
        self.keras_model.save_weights(path)

    def save_olra_model(self):
        self.keras_model.save(os.path.join(self.models_path, 'model.h5'))

    def log_to_tensorboard(self, step, losses, task='train'):
        def _decayed_learning_rate():
            return self.config.lr * self.config.decay_factor ** (step / self.config.decay_steps)

        with self.tensorboard.train_summary_writer.as_default():
            # Loss
            tf.summary.scalar('loss', losses[0], step=step)
            tf.summary.scalar('l1', losses[1], step=step)
            tf.summary.scalar('mse', losses[2], step=step)

            # Learning rate
            if self.config.apply_decay:
                tf.summary.scalar('lr', _decayed_learning_rate(), step=step)
            else:
                tf.summary.scalar('lr', self.config.lr, step=step)
