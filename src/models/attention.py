# -*- coding: utf-8 -*-
import tensorflow as tf

# Tensorflow 2.5 >
# from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow_addons.rnn import PeepholeLSTMCell
from tensorflow.keras.layers import StackedRNNCells, Dropout


class SigmoidLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SigmoidLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.sigmoid(inputs)


class Attention:
    def __init__(self, config, training=True):
        self.config = config
        self.rnn_size = config.rnn_size
        self.rnn_layer = config.rnn_layer
        self.batch_size = config.batch_size
        self.dim_hidden = config.dim_hidden
        self.dim_att = config.dim_attention
        self.max_words_q = config.max_len
        self.text_embedding_dim = config.text_embedding_dim
        self.drop_out_rate = config.dropout
        self.training = training
        self.dim_emb = (38, 38, config.dim_image + config.text_embedding_dim)
        self.question_input_shape = (config.max_len, config.text_embedding_dim)
        self.txt_input = (config.txt_feature_shape[0],
                          config.txt_feature_shape[1],
                          config.txt_feature_shape[2])
        self.labels_input = (38, 38)
        self.keras_model = None

    def build_model(self):
        """


        """

        question_input_lstm = tf.keras.layers.Input(shape=self.question_input_shape,
                                                    batch_size=self.config.batch_size,
                                                    dtype=tf.float32)

        image_emb_input = tf.keras.layers.Input(shape=self.dim_emb,
                                                batch_size=self.config.batch_size,
                                                dtype=tf.float32)

        # Q encoder: RNN body
        lstm_1 = PeepholeLSTMCell(self.rnn_size, dropout=1 - self.drop_out_rate)
        lstm_2 = PeepholeLSTMCell(self.rnn_size, dropout=1 - self.drop_out_rate)
        stacked_rnn = StackedRNNCells([lstm_1, lstm_2])

        state = stacked_rnn.get_initial_state(question_input_lstm,
                                              batch_size=self.config.batch_size,
                                              dtype=tf.float32)

        for i in range(self.max_words_q):
            ques_emb = question_input_lstm[:, i, :]
            lstm_output, state = stacked_rnn(ques_emb, state)

        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])

        img_emb = tf.keras.layers.Conv2D(1024,
                                         (1, 1),
                                         strides=(1, 1),
                                         padding='same',
                                         activation='tanh',
                                         kernel_initializer='glorot_uniform'
                                         )(image_emb_input)

        # attention layers
        # attention layer 1
        question_att = tf.expand_dims(question_emb, 1)
        question_att = tf.tile(question_att, tf.constant([1, self.dim_emb[0] * self.dim_emb[1], 1]))
        question_att = tf.reshape(question_att, [-1, 38, 38, self.dim_hidden])
        question_att = tf.keras.layers.Conv2D(512,
                                              (1, 1),
                                              padding='same',
                                              activation='tanh')(question_att)
        image_att = tf.keras.layers.Conv2D(512,
                                           (1, 1),
                                           padding='same',
                                           activation=None)(img_emb)

        output_att = tf.math.tanh(image_att + question_att)
        output_att = tf.keras.layers.Dropout(1 - self.drop_out_rate)(output_att)

        prob_att1 = tf.keras.layers.Conv2D(1,
                                           (1, 1),
                                           padding='same',
                                           activation=None)(output_att)
        prob_att1 = tf.reshape(prob_att1, (self.batch_size, self.dim_emb[0] * self.dim_emb[1]))

        prob_att1 = tf.keras.activations.softmax(prob_att1)

        image_att = []
        image_emb1 = tf.reshape(img_emb, [self.batch_size, self.dim_emb[0] * self.dim_emb[1],
                                          self.dim_hidden])

        for b in range(self.batch_size):
            image_att.append(tf.linalg.matmul(tf.expand_dims(prob_att1[b, :], 0), image_emb1[b, :, :]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.math.add(image_att, question_emb)

        del image_att
        del prob_att1

        # attention layer 2
        comb_att = tf.expand_dims(comb_emb, 1)
        comb_att = tf.tile(comb_att, tf.constant([1, self.dim_emb[0] * self.dim_emb[1], 1]))
        comb_att = tf.reshape(comb_att, [-1, 38, 38, self.dim_hidden])
        comb_att = tf.keras.layers.Conv2D(512,
                                          (1, 1),
                                          padding='same',
                                          activation='tanh')(comb_att)
        image_att = tf.keras.layers.Conv2D(512,
                                           (1, 1),
                                           padding='same',
                                           activation=None)(img_emb)

        output_att = tf.keras.activations.tanh(image_att + comb_att)
        output_att = tf.keras.layers.Dropout(1 - self.drop_out_rate)(output_att)

        prob_att2 = tf.keras.layers.Conv2D(1,
                                           (1, 1),
                                           padding='same',
                                           activation=None)(output_att)
        prob_att2 = tf.reshape(prob_att2, (self.batch_size, self.dim_emb[0] * self.dim_emb[1]))

        # END OF ATTENTION ===============================================================

        # activation in the loss function during training
        if not self.training:
            prob_att2 = SigmoidLayer(name="sigmoid")(prob_att2)

        prob_att2 = tf.reshape(prob_att2, [-1, 38, 38],
                               name='output')

        self.keras_model = tf.keras.Model([question_input_lstm, image_emb_input],
                                          prob_att2,
                                          name='vqa_attention_module')
