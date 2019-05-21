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
GAUSSIAN_MIXTURES = 5

HIDDEN_UNITS = 256

BATCH_SIZE = 32
EPOCHS = 20


# separate coefficients from NN output
def get_mixture_coef(y_pred):
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
    y_true = K.reshape(y_true, [-1, rollout_length, GAUSSIAN_MIXTURES,Z_DIM])

    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y_true - mu
#   result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = result * pi
    result = K.sum(result, axis=2)  # sum over gaussians
    # result = K.prod(result, axis=2) # multiply over latent dims
    return result


def sampling(pi, mu, sigma):
    pi = K.get_value(pi)
    pi = pi.reshape(GAUSSIAN_MIXTURES, Z_DIM)
    mu = K.get_value(mu)
    mu = mu.reshape(GAUSSIAN_MIXTURES, Z_DIM)
    sigma = K.get_value(sigma)
    sigma = sigma.reshape(GAUSSIAN_MIXTURES, Z_DIM)

    result = []
    for i in range(Z_DIM):
        p = [pi[0][i], pi[1][i], pi[2][i], pi[3][i], pi[4][i]]
        p = np.array(p)
        p /= p.sum()  # normalize
        # n = np.random.choice(5, 1, p=p)
        # n = n[0]  # make 1-dim list to a number
        n = np.argmax(p)
        print(p, n)

        epsilon = np.random.normal(loc=0., scale=1.)
        result.append(mu[n][i] + sigma[n][i] * epsilon)
    return result


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
        plt.ylabel('loss')
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
        # THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output, _, _ = lstm(rnn_x)
        mdn = Dense(GAUSSIAN_MIXTURES * (3 * Z_DIM))(lstm_output)  # + discrete_dim

        rnn = Model(rnn_x, mdn)

        # THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        state_inputs = [state_input_h, state_input_c]
        _, state_h, state_c = lstm(rnn_x, initial_state=[state_input_h, state_input_c])

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        # loss function
        def rnn_r_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)

            result = tf_normal(y_true, mu, sigma, pi)
            result = -K.log(K.maximum(result, 1e-8))
            result = K.mean(result, axis=(1, 2))  # mean over rollout length and z dim
            return result

        def rnn_kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2, 3])
            kl_tolerance = 0.5
            kl_loss = K.maximum(kl_loss, kl_tolerance * Z_DIM)
            return kl_loss

        def rnn_loss(y_true, y_pred):
            return rnn_r_loss(y_true, y_pred) + rnn_kl_loss(y_true, y_pred)

        optimizer = Adam(lr=0.0001)
        # optimizer = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
        rnn.compile(loss=rnn_loss, optimizer='rmsprop', metrics=[rnn_r_loss, rnn_kl_loss])

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
        pi, mu, sigma = get_mixture_coef(output)
        result = sampling(pi, mu, sigma)
        return result

