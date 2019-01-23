import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.layers import BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

import sys
sys.path.append("..")
import config

INPUT_DIM = config.input_dim
# Encoder
CONV_FILTERS = [32, 64, 128, 256]
CONV_KERNEL_SIZES = [(3, 4), (3, 4), (3, 4), (3, 4)]
CONV_STRIDES = [2, 2, 2, 2]
CONV_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']

DENSE_SIZE = 1024

# Decoder
CONV_T_FILTERS = [128, 64, 32, 3]
CONV_T_KERNEL_SIZES = [(4, 5), (4, 5), (4, 6), (6, 6)]
CONV_T_STRIDES = [2, 2, 2, 2]
CONV_T_ACTIVATIONS = ['relu', 'relu', 'relu', 'sigmoid']

Z_DIM = config.latent_vector_dim
EPOCHS = 10
BATCH_SIZE = 100


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon


class VAE():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM

    def _build(self):
        vae_x = Input(shape=INPUT_DIM)
        vae_c1_conv = Conv2D(filters=CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZES[0], strides=CONV_STRIDES[0],
                             padding='valid', data_format='channels_last', use_bias=False)(vae_x)
        vae_c1_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_c1_conv)
        vae_c1 = Activation(CONV_ACTIVATIONS[0])(vae_c1_bn)

        vae_c2_conv = Conv2D(filters=CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZES[1], strides=CONV_STRIDES[1],
                             padding='valid', data_format='channels_last', use_bias=False)(vae_c1)
        vae_c2_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_c2_conv)
        vae_c2 = Activation(CONV_ACTIVATIONS[1])(vae_c2_bn)

        vae_c3_conv = Conv2D(filters=CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZES[2], strides=CONV_STRIDES[2],
                             padding='valid', data_format='channels_last', use_bias=False)(vae_c2)
        vae_c3_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_c3_conv)
        vae_c3 = Activation(CONV_ACTIVATIONS[2])(vae_c3_bn)

        vae_c4_conv = Conv2D(filters=CONV_FILTERS[3], kernel_size=CONV_KERNEL_SIZES[3], strides=CONV_STRIDES[3],
                             padding='valid', data_format='channels_last', use_bias=False)(vae_c3)
        vae_c4_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_c4_conv)
        vae_c4 = Activation(CONV_ACTIVATIONS[3])(vae_c4_bn)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        # vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1, 1, DENSE_SIZE))(vae_dense_model)

        vae_d1_convT = Conv2DTranspose(filters=CONV_T_FILTERS[0], kernel_size=CONV_T_KERNEL_SIZES[0], padding='valid',
                                       data_format='channels_last', strides=CONV_T_STRIDES[0], use_bias=False)(vae_z_out)
        vae_d1_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_d1_convT)
        vae_d1 = Activation(CONV_T_ACTIVATIONS[0])(vae_d1_bn)
        # vae_d1_model = vae_d1(vae_z_out_model)

        vae_d2_convT = Conv2DTranspose(filters=CONV_T_FILTERS[1], kernel_size=CONV_T_KERNEL_SIZES[1], padding='valid',
                                       data_format='channels_last', strides=CONV_T_STRIDES[1], use_bias=False)(vae_d1)
        vae_d2_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_d2_convT)
        vae_d2 = Activation(CONV_T_ACTIVATIONS[1])(vae_d2_bn)
        # vae_d2_model = vae_d2(vae_d1_model)

        vae_d3_convT = Conv2DTranspose(filters=CONV_T_FILTERS[2], kernel_size=CONV_T_KERNEL_SIZES[2], padding='valid',
                                       data_format='channels_last', strides=CONV_T_STRIDES[2], use_bias=False)(vae_d2)
        vae_d3_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_d3_convT)
        vae_d3 = Activation(CONV_T_ACTIVATIONS[2])(vae_d3_bn)
        # vae_d3_model = vae_d3(vae_d2_model)

        vae_d4_convT = Conv2DTranspose(filters=CONV_T_FILTERS[3], kernel_size=CONV_T_KERNEL_SIZES[3], padding='valid',
                                       data_format='channels_last', strides=CONV_T_STRIDES[3], use_bias=False)(vae_d3)
        vae_d4_bn = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(vae_d4_convT)
        vae_d4 = Activation(CONV_T_ACTIVATIONS[3])(vae_d4_bn)
        # vae_d4_model = vae_d4(vae_d3_model)

        # MODELS
        vae = Model(vae_x, vae_d4)
        vae_encoder = Model(vae_x, vae_z)
        # vae_decoder = Model(vae_z_input, vae_d4)

        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        # optimizer = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
        optimizer = Adam(lr=0.0001)
        vae.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return (vae, vae_encoder, None) # vae_decoder)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data, validation_split=0.0):
        self.model.fit(data, data,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=validation_split)

        self.model.save_weights(config.vae_weight)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def get_vector(self, obs_data):
        z = self.encoder.predict(np.array(obs_data))
        return z

    def get_output(self, obs_data):
        # output = self.decoder.predict(np.array(z))
        output = self.model.predict(obs_data)
        return output

    def generate_rnn_data(self, obs_data, action_data):
        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x, y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)
