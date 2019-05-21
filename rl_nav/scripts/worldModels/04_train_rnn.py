# python 04_train_rnn.py --new_model

from RNN import RNN
import argparse
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append("..")
import config

dir_name = config.dir_name


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
    for i in range(65):
        # print('Building {}th...'.format(i))
        Input = np.load(dir_name + '/rnn_input_' + str(i) + '.npy')
        output = np.load(dir_name + '/rnn_output_' + str(i) + '.npy')
        # sequence pre-processing, for training LSTM the rnn_input must be (samples/episodes, time steps, features)
        Input = pad_sequences(Input, maxlen=30, dtype='float32', padding='pre', truncating='pre')
        output = pad_sequences(output, maxlen=30, dtype='float32', padding='pre', truncating='pre')
        rnn_input.append(Input)
        rnn_output.append(output)

    Input = rnn_input[0]
    output = rnn_output[0]
    for i in range(len(rnn_input)-1):
        Input = np.concatenate((Input, rnn_input[i+1]), axis=0)
        output = np.concatenate((output, rnn_output[i+1]), axis=0)
        print(Input.shape)
        print(output.shape)

    rnn.train(Input, output)
    rnn.plot_loss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train RNN'))
    parser.add_argument('--new_model', type=bool, default=True, help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
