import numpy as np
from RNN import RNN
from VAE import VAE
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
import config
dir_name = config.dir_name
episode = config.episode
frame = config.frame

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

vae = VAE()
vae.set_weights(config.vae_weight)

rnn = RNN()
rnn.set_weights(config.rnn_weight)


rnn_input = np.load(dir_name + '/rnn_input_0.npy')
rnn_output = np.load(dir_name + '/rnn_output_0.npy')
rnn_input = pad_sequences(rnn_input, maxlen=30, dtype='float32', padding='pre', truncating='pre')
rnn_output = pad_sequences(rnn_output, maxlen=30, dtype='float32', padding='pre', truncating='pre')

# print(rnn_input.shape)
# print(rnn_output.shape)
true = rnn_output[episode][frame]
print("The true value is: ")
print(true)

# print(rnn_input[episode][frame])
predict = rnn.get_output(np.expand_dims([rnn_input[episode][frame]], 0))
print("The prediction is: ")
print(predict)

plt.subplot(1, 2, 1)
true = true.reshape(1, 32)
true = vae.get_output(true)
true = true.reshape(48, 64, 3)[:, :, ::-1]  # convert bgr to rgb format
plt.imshow(true)  # The rgb values should be in the range [0 .. 1] for floats or [0 .. 255] for integers.
plt.title('Original')

plt.subplot(1, 2, 2)
predict = np.array(predict)
predict = predict.reshape(1, 32)
predict = vae.get_output(predict)
predict = predict.reshape(48, 64, 3)[:, :, ::-1]  # convert bgr to rgb format
plt.imshow(predict)  # The rgb values should be in the range [0 .. 1] for floats or [0 .. 255] for integers.
plt.title('Prediction')

plt.show()

# for i in range(rnn_input[indices].shape[0]):
#     output = rnn.get_output(rnn_input[:i+1])
#     print("the  real   value is {}".format(rnn_output[indices][i] - output))
#     # print("the predict value is {}".format(output))
