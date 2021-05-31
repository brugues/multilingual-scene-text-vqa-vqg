# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Tensorflow > 1.10 < 1.15
try:
    rnn_cell = tf.contrib.rnn
except:
    rnn_cell = tf.nn

# Tensorflow 2.5 >
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.layers import StackedRNNCells, Dropout


class TF2_Attention:
    def __init__(self, config, training=True):
        self.config = config
        self.rnn_size = config.rnn_size
        self.rnn_layer = config.rnn_layer
        self.batch_size = config.batch_size
        self.dim_hidden = config.dim_hidden
        self.dim_att = config.dim_attention
        self.max_words_q = config.maxlen
        self.text_embedding_dim = config.text_embedding_dim
        self.drop_out_rate = config.dropout
        self.training = training
        self.dim_image = (38, 38, config.dim_image + config.text_embedding_dim)
        self.question_input_shape = (config.maxlen, config.text_embedding_dim)
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

        image_emb_input = tf.keras.layers.Input(shape=self.dim_image,
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
                                         kernel_initializer='glorot_uniform',
                                         use_bias=True)(image_emb_input)

        # attention layers
        # attention layer 1
        question_att = tf.expand_dims(question_emb, 1)
        question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1]))
        question_att = tf.reshape(question_att, [-1, 38, 38, self.dim_hidden])
        question_att = tf.keras.layers.Conv2D(512,
                                              (1, 1),
                                              activation='tanh')(question_att)
        image_att = tf.keras.layers.Conv2D(512,
                                           (1, 1),
                                           activation=None)(img_emb)
        output_att = tf.keras.activations.tanh(image_att + question_att)
        output_att = tf.keras.layers.Dropout(1 - self.drop_out_rate)(output_att)

        prob_att1 = tf.keras.layers.Conv2D(1, (1, 1))(output_att)
        prob_att1 = tf.reshape(prob_att1, (self.batch_size, self.dim_image[0] * self.dim_image[1]))

        prob_att1 = tf.keras.activations.softmax(prob_att1)

        image_att = []
        image_emb = tf.reshape(img_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1],
                                         self.dim_hidden])

        for b in range(self.batch_size):
            image_att.append(tf.linalg.matmul(tf.expand_dims(prob_att1[b, :], 0), image_emb[b, :, :]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.math.add(image_att, question_emb)

        del image_att
        del prob_att1

        # attention layer 2
        comb_att = tf.expand_dims(comb_emb, 1)
        comb_att = tf.tile(comb_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1]))
        comb_att = tf.reshape(comb_att, [-1, 38, 38, self.dim_hidden])
        comb_att = tf.keras.layers.Conv2D(512,
                                          (1, 1),
                                          activation='tanh')(comb_att)
        image_att = tf.keras.layers.Conv2D(512,
                                           (1, 1),
                                           activation=None)(img_emb)
        output_att = tf.keras.activations.tanh(image_att + comb_att)
        output_att = tf.keras.layers.Dropout(1 - self.drop_out_rate)(output_att)

        prob_att2 = tf.keras.layers.Conv2D(1, (1, 1))(output_att)
        prob_att2 = tf.reshape(prob_att2, (self.batch_size, self.dim_image[0] * self.dim_image[1]))

        prob_att2 = tf.keras.activations.softmax(prob_att2)

        # END OF ATTENTION ===============================================================

        # activation in the loss function during training
        if not self.training:
            prob_att2 = tf.math.sigmoid(prob_att2)

        prob_att2 = tf.reshape(prob_att2, [-1, 38, 38])

        self.keras_model = tf.keras.Model([question_input_lstm, image_emb_input],
                                          prob_att2)

    def attention_layer(self, shape1, shape2, activation='softmax'):
        """
            Implements an attention layer of the network that can be reused

        """

        x1 = tf.keras.layers.Input(shape=shape1,
                                   batch_size=self.config.batch_size,
                                   dtype=tf.float32)
        x2 = tf.keras.layers.Input(shape=shape2,
                                   batch_size=self.config.batch_size,
                                   dtype=tf.float32)

        question_att = tf.expand_dims(x1, 1)
        question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1]))
        question_att = tf.reshape(question_att, [-1, 38, 38, self.dim_hidden])
        question_att = self.conv_layer(512, (1, 1), activation='tanh')(question_att)

        image_att = self.conv_layer(512, (1, 1))(x2)
        output_att = tf.keras.activations.tanh(image_att + question_att)
        output_att = tf.keras.layers.Dropout(1 - self.drop_out_rate)(output_att)

        prob_att = self.conv_layer(1, (1, 1))(output_att)
        prob_att = tf.reshape(prob_att, (self.batch_size, self.dim_image[0] * self.dim_image[1]))

        if activation == 'softmax':
            prob_att = tf.keras.activations.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(self.img_emb_input, [self.batch_size, self.dim_image[0] * self.dim_image[1],
                                                    self.dim_hidden])

        for b in range(self.batch_size):
            image_att.append(tf.linalg.matmul(tf.expand_dims(prob_att[b, :], 0), image_emb[b, :, :]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.math.add(image_att, self.question_input)

        return prob_att, comb_emb


class Attention():
    def __init__(self, rnn_size, rnn_layer, batch_size, dim_image, dim_hidden, dim_attention, max_words_q,
                 text_embedding_dim, drop_out_rate, training=True):

        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.dim_image = dim_image
        self.dim_hidden = dim_hidden
        self.dim_att = dim_attention
        self.max_words_q = max_words_q
        self.text_embedding_dim = text_embedding_dim
        self.drop_out_rate = drop_out_rate
        self.training = training

        # Q encoder: RNN body
        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob=1 - self.drop_out_rate)
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob=1 - self.drop_out_rate)
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

    def build_model(self, image_features, txt_features, question):

        state = self.stacked_lstm.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope("embed"):
            for i in range(self.max_words_q):
                ques_emb = question[:, i, :]
                output, state = self.stacked_lstm(ques_emb, state)

        question_emb = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])

        image_emb = tf.concat([image_features, txt_features], 3)
        image_emb = self.conv_layer(image_emb, (1, 1, self.dim_image[2], self.dim_hidden), activation='tanh')

        # attention layers
        with tf.variable_scope("att1"):
            _, comb_emb = self.attention_layer(question_emb, image_emb, activation='softmax')
        with tf.variable_scope("att2"):
            prob_att2, _ = self.attention_layer(comb_emb, image_emb, activation=None)

        # activation in the loss function during training
        if not self.training:
            prob_att2 = tf.nn.sigmoid(prob_att2)

        prob_att = tf.reshape(prob_att2, [-1, 38, 38])
        return prob_att

    def conv_layer(self, inp, kernels_shape, activation=None):
        xavier = tf.contrib.layers.xavier_initializer()
        kernels = np.zeros(kernels_shape, dtype=np.float32)
        biases = np.zeros((kernels.shape[-1],), dtype=np.float32)
        conv_kernels = tf.Variable(xavier(kernels.shape), name='ConvKernels')
        conv_biases = tf.Variable(biases, name='ConvBiases')
        x = tf.nn.conv2d(inp, conv_kernels, padding='SAME', strides=[1, 1, 1, 1])
        x = tf.nn.bias_add(x, conv_biases)

        if activation == 'tanh':
            x = tf.tanh(x)

        return x

    def attention_layer(self, question_emb, image_emb, activation='softmax'):

        question_att = tf.expand_dims(question_emb, 1)
        question_att = tf.tile(question_att, tf.constant([1, self.dim_image[0] * self.dim_image[1], 1]))
        question_att = tf.reshape(question_att, [-1, 38, 38, self.dim_hidden])
        question_att = self.conv_layer(question_att, (1, 1, self.dim_hidden, self.dim_att), activation='tanh')

        image_att = self.conv_layer(image_emb, (1, 1, self.dim_hidden, self.dim_att), activation=None)

        output_att = tf.tanh(image_att + question_att)
        output_att = tf.nn.dropout(output_att, 1 - self.drop_out_rate)

        prob_att = self.conv_layer(output_att, (1, 1, self.dim_att, 1), activation=None)
        prob_att = tf.reshape(prob_att, [self.batch_size, self.dim_image[0] * self.dim_image[1]])

        if activation == 'softmax':
            prob_att = tf.nn.softmax(prob_att)

        image_att = []
        image_emb = tf.reshape(image_emb, [self.batch_size, self.dim_image[0] * self.dim_image[1], self.dim_hidden])
        for b in range(self.batch_size):
            image_att.append(tf.matmul(tf.expand_dims(prob_att[b, :], 0), image_emb[b, :, :]))

        image_att = tf.stack(image_att)
        image_att = tf.reduce_sum(image_att, 1)

        comb_emb = tf.add(image_att, question_emb)

        return prob_att, comb_emb
