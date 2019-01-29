import numpy as np
import argparse
import VAE
sys.path.append("..")
import config

vae = VAE.VAE()
vae.set_weights(config.vae_weight)

for i in range(95, 98):
    obs_data = np.load('./record/observation{}'.format(i) + '.npy')
    action_data = np.load('./record/action{}'.format(i) + '.npy')

    rnn_input, rnn_output = vae.generate_rnn_data(obs_data, action_data)
    np.save('./data/rnn_input_' + str(i), rnn_input)
    np.save('./data/rnn_output_' + str(i), rnn_output)
