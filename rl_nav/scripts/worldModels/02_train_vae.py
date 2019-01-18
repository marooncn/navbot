# python 02_train_vae.py --new_model

import argparse
import numpy as np
import VAE

import sys
sys.path.append("..")
import config


def main(args):
    new_model = args.new_model

    vae = VAE.VAE()

    if not new_model:
        try:
            vae.set_weights(config.vae_weight)
        except:
            print("Either set --new_model or ensure" + config.vae_weight + " exists")
            raise

    data = np.load('./record/observation.npy')
    data = np.array([item for episode in data for item in episode])
    np.random.seed(0)
    for _ in range(10):
        indices = np.random.choice(data.shape[0], 10000, replace=False)
        data = data[indices] / 255.0
        # print(data.shape)

        vae.train(data, 0.0)

    indices = np.random.choice(data.shape[0], 10000, replace=False)
    data = data[indices] / 255.0
    # print(data.shape)
    vae.train(data, 0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--new_model', type=bool, default=False, help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
