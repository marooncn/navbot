import numpy as np
import sys
import os
import VAE
sys.path.append("..")
import config

dir_name = config.dir_name
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

vae = VAE.VAE()
vae.set_weights(config.vae_weight)

for i in range(65):
    obs_data = np.load('./maze1_record/observation{}'.format(i) + '.npy')
    obs_data = np.array([np.asarray(episode) for episode in obs_data])
    obs_data = obs_data / 255.0

    action_data = np.load('./maze1_record/action{}'.format(i) + '.npy')

    rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
    np.save(dir_name+'/rnn_input_' + str(i), rnn_input)
    np.save(dir_name+'/rnn_output_' + str(i), rnn_output)
    print("Saving {}th succeeds!".format(i))
