import math

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
import sys
import numpy as np
sys.path.append("..")
import config

Z_DIM = config.latent_vector_dim
ACTION_DIM = 2

HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5

BATCH_SIZE = 64
EPOCHS = 1


# separate coefficients from NN output
def get_mdn_coef(y_pred):
    d = GAUSSIAN_MIXTURES * Z_DIM

    rollout_length = K.shape(y_pred)[1]

    pi = y_pred[:, :, :d]
    mu = y_pred[:, :, d:(2 * d)]
    log_sigma = y_pred[:, :, (2 * d):(3 * d)]
    # discrete = y_pred[:,3*GAUSSIAN_MIXTURES:]

    pi = K.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)

    return pi, mu, sigma  # , discrete


# compute the distribution value under y_true
def tf_normal(y_true, mu, sigma, pi):
    rollout_length = K.shape(y_true)[1]
    y_true = K.tile(y_true, (1, 1, GAUSSIAN_MIXTURES))
    y_true = K.reshape(y_true, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

    result = y_true - mu
    result = result * (1.0 / (sigma + 1e-8))
    result = -K.square(result) / 2.0
    result = K.exp(result) * 1.0 / ((math.sqrt(2 * math.pi))*(sigma + 1e-8))
    result = result * pi
    result = K.sum(result, axis=2)  # sum over gaussian
    # constrain the max value of each element is no more than 1.0
    # result = K.expand_dims(result, -1)
    # max_value = K.ones_like(result)
    # result = K.concatenate([result, max_value], -1)
    # K.min(result, -1, keepdims=False)
    return result


# samples from a categorial distribution
def get_pi_idx(x, pdf):
    accumulate = 0
    for i in range(GAUSSIAN_MIXTURES):
        accumulate += K.get_value(pdf[i])
        # print(accumulate)
        if accumulate >= x:
            return i
    print('error with sampling ensemble')
    return -1


# generates a random sequence using the trained model
def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
    result = np.random.rand(Z_DIM, M)  # initially random [0, 1]
    rn = np.random.randn(Z_DIM, M)  # normal random matrix (0.0, 1.0)
    out_pi = K.reshape(out_pi, [GAUSSIAN_MIXTURES, Z_DIM])
    out_pi = K.permute_dimensions(out_pi, (1, 0))
    # print(out_pi.shape)
    out_mu = K.reshape(out_mu, [GAUSSIAN_MIXTURES, Z_DIM])
    out_sigma = K.reshape(out_sigma, [GAUSSIAN_MIXTURES, Z_DIM])

    # transforms result into random ensembles
    for j in range(M):
        for i in range(Z_DIM):
            idx = get_pi_idx(result[i, j], out_pi[i])
            mu = out_mu[idx, i]
            std = out_sigma[idx, i]
            result[i, j] = K.get_value(mu) + rn[i, j] * K.get_value(std)
            # print(result[i, j])
    return result


class RNN():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.hidden_units = HIDDEN_UNITS
        self.gaussian_mixtures = GAUSSIAN_MIXTURES

    def _build(self):
        # the model that will be trained
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output, _, _ = lstm(rnn_x)
        mdn = Dense(GAUSSIAN_MIXTURES * (3 * Z_DIM))(lstm_output)  # + discrete_dim

        rnn = Model(rnn_x, mdn)

        # the model used during prediction
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        state_inputs = [state_input_h, state_input_c]
        _, state_h, state_c = lstm(rnn_x, initial_state=[state_input_h, state_input_c])

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        # loss function
        def rnn_r_loss(y_true, y_pred):
            pi, mu, sigma = get_mdn_coef(y_pred)

            result = tf_normal(y_true, mu, sigma, pi)

            # result = -K.log(result + 1e-8)
            # result = K.mean(result, axis=(1, 2))  # mean over rollout length and z dim
            result = K.max(result, axis=(1, 2))
            return result

        def rnn_kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mdn_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2, 3])
            kl_tolerance = 0.5
            kl_loss = K.maximum(kl_loss, kl_tolerance * Z_DIM)
            return kl_loss

        def rnn_loss(y_true, y_pred):
            return rnn_r_loss(y_true, y_pred)  # + rnn_kl_loss(y_true, y_pred)

        # optimizer = Adam(lr=0.0001)
        optimizer = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
        rnn.compile(loss=rnn_loss, optimizer=optimizer, metrics=[rnn_r_loss, rnn_kl_loss])

        return [rnn, forward]

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output, i=0):
        earlystop = EarlyStopping(monitor='rnn_r_loss', min_delta=0.01, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]

        self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=0,
                       callbacks=callbacks_list)

        self.model.save_weights('./models/rnn_weights_{}.h5'.format(i))

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    # predict the output
    def get_output(self, input_data):
        output = self.model.predict(input_data)
        pi, mu, sigma = get_mdn_coef(output)
        result = generate_ensemble(pi, mu, sigma)
        return result
