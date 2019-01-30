# python 02_train_vae.py --new_model

import argparse
import numpy as np
import VAE
from keras import backend as K


import sys
sys.path.append("..")
import config


def main(args):
    new_model = args.new_model

    vae = VAE.VAE()
    vae.set_weights(config.vae_weight)

    if not new_model:
        try:
            vae.set_weights(config.vae_weight)
        except:
            print("Either set --new_model or ensure {} exists".format(config.vae_weight))
            raise

    for j in range(2):
        for i in range(100):
            data = np.load('./record/observation{}.npy'.format(i), encoding='latin1')
            data = np.array([item for episode in data for item in episode])
            # np.random.seed(0)
            # indices = np.random.choice(data.shape[0], 40000, replace=False)
            data = data.astype(np.float) / 255.0
            vae.train(data, j, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--new_model', type=bool, default=True, help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
