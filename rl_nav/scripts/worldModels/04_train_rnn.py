# python 04_train_rnn.py --new_model

from RNN import RNN
import argparse
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append("..")
import config


def main(args):
    new_model = args.new_model

    rnn = RNN()

    if not new_model:
        try:
            rnn.set_weights(config.rnn_weight)
        except:
            print("Either set --new_model or ensure {} exists".format(config.rnn_weight))
            raise

    for i in range(100):
        print('Building {}th...'.format(i))
        rnn_input = np.load('./rnn_data/rnn_input_' + str(i) + '.npy')
        rnn_output = np.load('./rnn_data/rnn_output_' + str(i) + '.npy')
        # sequence pre-processing, for training LSTM the rnn_input must be (samples/epochs, time steps, features)
        rnn_input = pad_sequences(rnn_input, maxlen=40, dtype='float32', padding='post', truncating='post')
        rnn_output = pad_sequences(rnn_output, maxlen=40, dtype='float32', padding='post', truncating='post')
        print(rnn_input.shape)
        print(rnn_output.shape)
        rnn.train(rnn_input, rnn_output, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--new_model', type=bool, default=True, help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
