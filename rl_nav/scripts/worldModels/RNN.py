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

BATCH_SIZE = 32
EPOCHS = 5


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
        # the model that will be trained
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)

        lstm_output, _, _ = lstm(rnn_x)
        mdn = Dense(Z_DIM)(lstm_output)

        rnn = Model(rnn_x, mdn)

        # the model used during prediction
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        state_inputs = [state_input_h, state_input_c]
        
        _, state_h, state_c = lstm(rnn_x, initial_state=state_inputs)
        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        optimizer = Adam(lr=0.0001)
        # optimizer = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
        rnn.compile(loss='mean_squared_error', optimizer=optimizer)

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

        self.model.save_weights('./models/rnn_weights_RNN.h5')

    def plot_loss(self):
        self.history.loss_plot('batch')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    # predict the output
    def get_output(self, input_data):
        output = self.model.predict(input_data)
        return output

