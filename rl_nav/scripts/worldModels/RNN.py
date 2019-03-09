import math

from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import sys

import numpy as np
sys.path.append("..")
import config

Z_DIM = config.latent_vector_dim
ACTION_DIM = 2

HIDDEN_UNITS = 256

BATCH_SIZE = 64
EPOCHS = 2


# separate coefficients from NN output
def get_mdn_coef(y_pred):
    rollout_length = K.shape(y_pred)[1]

    mu = y_pred[:, :, 0:Z_DIM]
    log_sigma = y_pred[:, :, Z_DIM:(2 * Z_DIM)]

    mu = K.reshape(mu, [-1, rollout_length, Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, Z_DIM])
    sigma = K.exp(log_sigma)
    sigma = K.sigmoid(sigma)

    return mu, sigma


# compute the distribution value under y_true
def tf_normal(y_true, mu, sigma):
    result = y_true - mu
    result = result * (1.0 / (sigma + 1e-8))
    result = -K.square(result) / 2.0
    result = K.exp(result) * 1.0 / ((math.sqrt(2 * math.pi))*(sigma + 1e-8))
    # constrain the max value of each element is no more than 1.0
    # result = K.expand_dims(result, -1)
    # max_value = K.ones_like(result)
    # result = K.concatenate([result, max_value], -1)
    # K.min(result, -1, keepdims=False)
    return result


def sampling(mu, sigma):
    epsilon = K.random_normal(shape=(K.shape(mu)[0], Z_DIM), mean=0., stddev=1.)
    return mu + sigma * epsilon


# draw the training loss
class LossHistory(Callback):
    def __init__(self):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class RNN():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]
        self.history = LossHistory()
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.hidden_units = HIDDEN_UNITS

    def _build(self):
        # the model that will be trained
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output, _, _ = lstm(rnn_x)
        mdn = Dense(2 * Z_DIM)(lstm_output)  # + discrete_dim

        rnn = Model(rnn_x, mdn)

        # the model used during prediction
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        state_inputs = [state_input_h, state_input_c]
        _, state_h, state_c = lstm(rnn_x, initial_state=[state_input_h, state_input_c])

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        # loss function
        def rnn_r_loss(y_true, y_pred):
            mu, sigma = get_mdn_coef(y_pred)

            result = tf_normal(y_true, mu, sigma)
            result = -K.log(K.maximum(result, 1e-8))
            result = K.mean(result, axis=(1, 2))  # mean over rollout length and z dim
            return result

        def rnn_kl_loss(y_true, y_pred):
            mu, sigma = get_mdn_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2])
            kl_tolerance = 0.5
            kl_loss = K.maximum(kl_loss, kl_tolerance * Z_DIM)
            return kl_loss

        def rnn_loss(y_true, y_pred):
            return rnn_r_loss(y_true, y_pred)  # + rnn_kl_loss(y_true, y_pred)

        optimizer = Adam(lr=0.0001)
        # optimizer = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
        rnn.compile(loss=rnn_loss, optimizer=optimizer, metrics=[rnn_r_loss, rnn_kl_loss])

        return [rnn, forward]

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, rnn_input, rnn_output,):
        callbacks_list = [self.history]

        self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       callbacks=callbacks_list)

        self.model.save_weights('./models/rnn_weights.h5')

    def plot_loss(self):
        self.history.loss_plot('batch')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    # predict the output
    def get_output(self, input_data):
        output = self.model.predict(input_data)
        mu, sigma = get_mdn_coef(output)
        result = sampling(mu, sigma)
        return K.get_value(result)

