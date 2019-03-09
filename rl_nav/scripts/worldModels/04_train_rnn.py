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

    rnn_input = []
    rnn_output = []
    for i in range(130):
        # print('Building {}th...'.format(i))
        input = np.load('./rnn_data/rnn_input_' + str(i) + '.npy')
        output = np.load('./rnn_data/rnn_output_' + str(i) + '.npy')
        # sequence pre-processing, for training LSTM the rnn_input must be (samples/episodes, time steps, features)
        input = pad_sequences(input, maxlen=40, dtype='float32', padding='post', truncating='post')
        output = pad_sequences(output, maxlen=40, dtype='float32', padding='post', truncating='post')
        rnn_input.append(input)
        rnn_output.append(output)

    input = rnn_input[0]
    output = rnn_output[0]
    for i in range(len(rnn_input)-1):
        input = np.concatenate((input, rnn_input[i+1]), axis=0)
        output = np.concatenate((output, rnn_output[i+1]), axis=0)
        print(input.shape)
        print(output.shape)

    rnn.train(input, output)
    rnn.plot_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--new_model', type=bool, default=True, help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
