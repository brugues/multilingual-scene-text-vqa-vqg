import os

import numpy as np
import tensorflow as tf
from dataloader.utils import print_info, print_ok
from tensorflow.python.ops.gen_batch_ops import batch
from models.stvqa import TensorBoardLogger
from dataloader.data_loader import OLRADataGenerator


class SeqSelfAttention(tf.keras.layers.Layer):
    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.
        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attention_activation = tf.keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = tf.keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'attention_activation': tf.keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = tf.shape(inputs)[-1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = tf.experimental.numpy.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = tf.experimental.numpy.arange(0, input_len) - self.attention_width // 2
            lower = tf.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = tf.expand_dims(tf.experimental.numpy.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - tf.cast(lower <= indices,
                                          tf.keras.backend.floatx()) * tf.cast(indices < upper,
                                                                               tf.keras.backend.floatx()))
        if mask is not None:
            mask = tf.expand_dims(tf.cast(mask, tf.keras.backend.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - tf.keras.backend.permute_dimensions(mask, (0, 2, 1))))

        # a_{t} = \text{softmax}(e_t)
        e = tf.math.exp(e - tf.math.reduce_max(e, axis=-1, keepdims=True))
        a = e / tf.math.reduce_sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = tf.matmul(a, inputs)

        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)

        q = tf.expand_dims(tf.matmul(inputs, self.Wt), 2)
        k = tf.expand_dims(tf.matmul(inputs, self.Wx), 1)

        if self.use_additive_bias:
            h = tf.keras.activations.tanh(q + k + self.bh)
        else:
            h = tf.keras.activations.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = tf.reshape(tf.matmul(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = tf.reshape(tf.matmul(h, self.Wa), (batch_size, input_len, input_len))

        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = tf.matmul(tf.matmul(inputs, self.Wa), tf.keras.backend.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = tf.cast(tf.shape(attention)[0], tf.keras.backend.floatx())
        input_len = tf.shape(attention)[-1]
        indices = tf.expand_dims(tf.experimental.numpy.arange(0, input_len), axis=0)
        diagonal = tf.expand_dims(tf.experimental.numpy.arange(0, input_len), axis=-1)
        eye = tf.cast(tf.math.equal(indices, diagonal), tf.keras.backend.floatx())
        return self.attention_regularizer_weight * tf.math.add(tf.math.square(tf.matmul(
            attention,
            tf.keras.backend.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


def reset_decoder_state(features):
    return features


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                                 self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRU(self.units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    def call(self, x, hidden_h, hidden_c, use_lstm=True):
        # defining attention as a separate model
        # context_vector, attention_weights = self.attention(features, hidden)
        #
        # # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # x = self.embedding(x)
        #
        #     # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        x = self.embedding(x)

        if use_lstm:
            # passing the concatenated vector to the LSTM
            output, state_h, state_c = self.lstm(x,
                                                 initial_state=[hidden_h, hidden_c])
        else:
            output, state_h = self.gru(x, initial_state=hidden_h)

        # # shape == (batch_size, max_length, hidden_size)
        # x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        output = self.fc2(output)

        if use_lstm:
            return output, state_h, state_c
        else:
            return output, state_h

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class OLRA:

    def __init__(self, config, training=True) -> None:
        self.config = config
        self.training = training
        self.logging_path = config.logging_path
        self.models_path = config.models_path

        print_info('Preparing data generator\n')
        self.data_generator = OLRADataGenerator(config, training=training)
        print_ok('Done\n')

        self.resnet = tf.keras.applications.ResNet50V2(include_top=False)
        self.resnet = tf.keras.Model(self.resnet.input,
                                     self.resnet.layers[-1].output)

        # self.decoder = self.build_decoder_model()
        self.decoder = RNN_Decoder(self.config.dim_hidden, self.config.rnn_size, self.data_generator.top_k + 1)
        self.feature_model = self.build_feature_model()

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
            os.makedirs(os.path.join(self.models_path, 'checkpoints_feature'), exist_ok=True)
            os.makedirs(os.path.join(self.models_path, 'checkpoints_decoder'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.logging_path, 'val'), exist_ok=True)

        else:
            self.feature_model.load_weights(os.path.join(config.model_to_evaluate,
                                                         'checkpoints_feature', 'ckpt'))
            self.decoder.load_weights(os.path.join(config.model_to_evaluate,
                                                   'checkpoints_decoder', 'ckpt'))

        self.tensorboard = TensorBoardLogger(self.logging_path)

        self.l2_loss = tf.keras.losses.MeanSquaredError()
        self.mle_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        if self.config.apply_decay:
            self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.lr,
                                                                            decay_rate=config.decay_factor,
                                                                            decay_steps=self.data_generator.len(),
                                                                            staircase=config.staircase)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.scheduler)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)

    def build_feature_model(self):
        """
            Builds the OLRA keras model
        """

        fasttext_input = tf.keras.layers.Input(shape=self.config.dim_txt,
                                               batch_size=self.config.batch_size,
                                               dtype=tf.float32)

        image_input = tf.keras.layers.Input(shape=2048,
                                            batch_size=self.config.batch_size,
                                            dtype=tf.float32)

        ocr_positions_input = tf.keras.layers.Input(shape=4,
                                                    batch_size=self.config.batch_size,
                                                    dtype=tf.float32)

        # Get fasttext features from B-LSTM
        # ocr_features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(self.config.rnn_size / 2),
        #                                                                   activation='tanh',
        #                                                                   recurrent_activation='tanh',
        #                                                                   dropout=self.config.dropout)
        #                                              )(tf.expand_dims(fasttext_input, axis=1))

        # Multimodal Fusion (Based on self attention)

        image_features = tf.keras.layers.Dense(512,
                                               activation='tanh')(image_input)

        fused_features = tf.keras.layers.concatenate([fasttext_input,
                                                      image_features,
                                                      ocr_positions_input])

        if self.config.multimodal_attention:
            fused_features = tf.expand_dims(fused_features, axis=1)
            fused_features = SeqSelfAttention(units=828,
                                              attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                              attention_activation='softmax')(fused_features)
            fused_features = tf.keras.layers.Dense(512)(fused_features)
            fused_features = SeqSelfAttention(units=512,
                                              attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                              attention_activation='softmax')(fused_features)
            fused_features = tf.reshape(fused_features, (self.config.batch_size, 512))

        fused_features = tf.keras.layers.Dense(818)(fused_features)
        fused_features = tf.keras.layers.Dense(512)(fused_features)

        # else:
        #     fused_features = tf.keras.layers.Dense(512)(fused_features)

        # OCR consistency module
        if self.config.use_ocr_consistency:
            ocr_consistency = tf.keras.layers.Dense(300)(fused_features)

            # Create final model
            model = tf.keras.Model(inputs=[fasttext_input,
                                           image_input,
                                           ocr_positions_input],
                                   outputs=[fused_features, fasttext_input, ocr_consistency])
        else:
            model = tf.keras.Model(inputs=[fasttext_input,
                                           image_input,
                                           ocr_positions_input],
                                   outputs=[fused_features, fasttext_input])

        return model

    #@tf.function
    def olra_train_step(self, fasttext_features, images, ocr_posistions, questions_input):
        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.mle_loss(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        dec_input = tf.expand_dims(questions_input[:, 0], 1)
        # hidden = self.decoder.reset_state(fused_features)
        l2_loss = 0
        mle_loss = 0

        with tf.GradientTape() as tape:
            if self.config.use_ocr_consistency:
                fused_features, ocr_features, ocr_consistency = self.feature_model([fasttext_features,
                                                                                    images,
                                                                                    ocr_posistions])
                l2_loss = self.l2_loss(ocr_features, ocr_consistency)
            else:
                fused_features, ocr_features = self.feature_model([fasttext_features,
                                                                   images,
                                                                   ocr_posistions])

            hidden_h = fused_features
            hidden_c = tf.zeros_like(fused_features)

            for i in range(1, self.data_generator.max_len):
                # passing the features through the decoder
                if self.config.use_lstm:
                    predictions, hidden_h, hidden_c = self.decoder(dec_input, hidden_h, hidden_c)
                else:
                    predictions, hidden_h = self.decoder(dec_input, hidden_h, None,
                                                         use_lstm=self.config.use_lstm)
                mle_loss += loss_function(questions_input[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(questions_input[:, i], 1)

            if self.config.use_ocr_consistency:
                loss = self.config.lambda_loss * l2_loss + (mle_loss / self.data_generator.max_len)
            else:
                loss = mle_loss / self.data_generator.max_len

        trainable_variables = self.feature_model.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return [loss, l2_loss, (mle_loss / self.data_generator.max_len)]

    def evaluation_step(self, fasttext_features, images, ocr_posistions, questions_input):

        if self.config.use_ocr_consistency:
            fused_features, ocr_features, ocr_consistency = self.feature_model([fasttext_features,
                                                                                images,
                                                                                ocr_posistions])
        else:
            fused_features, ocr_features = self.feature_model([fasttext_features,
                                                               images,
                                                               ocr_posistions])

        dec_input = tf.expand_dims(questions_input[:, 0], 1)
        hidden_h = fused_features
        hidden_c = fused_features
        result = []

        for i in range(self.config.max_len):
            if self.config.use_lstm:
                predictions, hidden_h, hidden_c = self.decoder(dec_input, hidden_h, hidden_c)
            else:
                predictions, hidden_h = self.decoder(dec_input, hidden_h, None, use_lstm=False)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

            try:
                result.append(self.data_generator.tokenizer.index_word[predicted_id])
                if self.data_generator.tokenizer.index_word[predicted_id] == '<end>':
                    return result
            except KeyError:
                result.append('oov')

            dec_input = tf.expand_dims([predicted_id], 0)

        return result

    def save_olra_checkpoint(self):
        path = os.path.join(self.models_path, 'checkpoints_feature', 'ckpt')
        self.feature_model.save_weights(path)
        path = os.path.join(self.models_path, 'checkpoints_decoder', 'ckpt')
        self.decoder.save_weights(path)

    def log_to_tensorboard(self, step, losses, task='train'):
        def _decayed_learning_rate():
            return self.config.lr * self.config.decay_factor ** (step / self.data_generator.len())

        with self.tensorboard.train_summary_writer.as_default():
            # Loss
            tf.summary.scalar('loss', losses[0], step=step)
            tf.summary.scalar('l1', losses[1], step=step)
            tf.summary.scalar('mle', losses[2], step=step)

            # Learning rate
            if self.config.apply_decay:
                tf.summary.scalar('lr', _decayed_learning_rate(), step=step)
            else:
                tf.summary.scalar('lr', self.config.lr, step=step)
