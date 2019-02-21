import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K   # https://keras.io/backend/
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD


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
EPOCHS = 2
BATCH_SIZE = 256  # 128  


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
        vae_c1 = Conv2D(filters=CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZES[0], strides=CONV_STRIDES[0],
                        padding='valid', data_format='channels_last', activation=CONV_ACTIVATIONS[0])(vae_x)   # , kernel_regularizer=l2(0.0001))(vae_x)
        vae_c2 = Conv2D(filters=CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZES[1], strides=CONV_STRIDES[1],
                        padding='valid', data_format='channels_last', activation=CONV_ACTIVATIONS[1])(vae_c1)  # , kernel_regularizer=l2(0.0001))(vae_c1)
        vae_c3 = Conv2D(filters=CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZES[2], strides=CONV_STRIDES[2],
                        padding='valid', data_format='channels_last', activation=CONV_ACTIVATIONS[2])(vae_c2)  # , kernel_regularizer=l2(0.0001))(vae_c2)
        vae_c4 = Conv2D(filters=CONV_FILTERS[3], kernel_size=CONV_KERNEL_SIZES[3], strides=CONV_STRIDES[3],
                        padding='valid', data_format='channels_last', activation=CONV_ACTIVATIONS[3])(vae_c3)  # , kernel_regularizer=l2(0.0001))(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)  # , kernel_regularizer=l2(0.0001))(vae_z_in) 
        vae_z_log_var = Dense(Z_DIM)(vae_z_in) # , kernel_regularizer=l2(0.0001))(vae_z_in) 

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)  # , kernel_regularizer=l2(0.0001))
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1, 1, DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters=CONV_T_FILTERS[0], kernel_size=CONV_T_KERNEL_SIZES[0], padding='valid',
                                 data_format='channels_last', strides=CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0]) # , kernel_regularizer=l2(0.0001))
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters=CONV_T_FILTERS[1], kernel_size=CONV_T_KERNEL_SIZES[1], padding='valid',
                                 data_format='channels_last', strides=CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1]) # , kernel_regularizer=l2(0.0001))
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters=CONV_T_FILTERS[2], kernel_size=CONV_T_KERNEL_SIZES[2], padding='valid',
                                 data_format='channels_last', strides=CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2]) # , kernel_regularizer=l2(0.0001))
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters=CONV_T_FILTERS[3], kernel_size=CONV_T_KERNEL_SIZES[3], padding='valid',
                                 data_format='channels_last', strides=CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3]) # , kernel_regularizer=l2(0.0001))
        vae_d4_model = vae_d4(vae_d3_model)

        # DECODER ONLY
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        # MODELS
        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        def vae_kl_loss(y_true, y_pred):
            kl_loss = - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)
            # stop optimizing for KL loss term once it is lower than some level, rather than letting it go to near zero.
            # So optimize for tf.max(KL, good_enough_kl_level) instead, to relax the information bottleneck of the VAE.
            kl_tolerance = 0.5
            kl_loss = K.maximum(kl_loss, kl_tolerance*Z_DIM)
            return kl_loss

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        optimizer = Adam(lr=1e-4)  # we train it separately, so we need decrese to 1e-5, 1e-6, 1e-7 by hand when the loss doesn't decrease even increase.
        # optimizer = SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=True)
        vae.compile(optimizer=optimizer, loss=vae_loss, metrics=[vae_r_loss, vae_kl_loss])

        return [vae, vae_encoder, vae_decoder]

    def set_weights(self, filepath):
        self.model.load_weights(filepath)
        print("load weights {} successfully".format(filepath))

    def train(self, data, j=0, i=0):
        earlystop = EarlyStopping(monitor='vae_r_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(data, data,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       callbacks=callbacks_list)

        self.model.save_weights('./models/vae_weights_' + str(j) + '_' + str(i) + '.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def get_vector(self, obs_data):
        z = self.encoder.predict(np.array(obs_data))
        return z

    def get_output(self, input_data):
        input_data = np.asarray(input_data)
        if input_data.shape == (1, Z_DIM):  # if input is latent vector
            output = self.decoder.predict(input_data)
            return output
        elif input_data.shape == (1, 48, 64, 3):  # if input is the raw image
            output = self.model.predict(input_data)
            return output
        else:
            raise Exception("Input shape {} is not right, it needs to be (1, {}) or (1, 48, 64, 3).".format(input.shape, Z_DIM))

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

        return [rnn_input, rnn_output]

