import numpy as np
import sys
import os
import VAE
sys.path.append("..")
import config


dir_name = 'rnn_data'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

vae = VAE.VAE()
vae.set_weights(config.vae_weight)

for i in range(100):
    obs_data = np.load('./record/observation{}'.format(i) + '.npy')
    action_data = np.load('./record/action{}'.format(i) + '.npy')

    rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
    np.save(dir_name+'/rnn_input_' + str(i), rnn_input)
    np.save(dir_name+'/rnn_output_' + str(i), rnn_output)
    print("Saving succeeds!")
