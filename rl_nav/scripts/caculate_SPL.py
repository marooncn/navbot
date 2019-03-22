#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import config
import matplotlib.pyplot as plt


def main(args):
    recordpath = config.path + '/scripts/record'
    filepath = recordpath + args.filepath
    f = open(filepath, 'r')
    lines = f.readlines()
    last100 = lines[-100:]
    successes = []
    rewards = []
    timesteps = []
    shortest = 1000

    for i in range(100):
        line = last100[i]  # eg: '[432.1290540951935, 248, True]'
        data = line.split()
        rewards.append(float(data[0][1:-1]))  # 432.1290540951935
        timesteps.append(float(data[1][:-1]))  # 248
        # successes.append(bool(data[2][:-1]))  # bool('False') is True!
        successes.append(data[2][:-1] == str('True'))
        if successes[i]:
            if timesteps[i] < shortest:
                shortest = timesteps[i]
    print('success rate: {}%'.format(sum(successes)))
    print('the shortest path distance: {}'.format(shortest))

    SPL = 0
    for i in range(100):
        if successes[i]:
            SPL += 1.0/100*shortest/max(timesteps[i], shortest)
    print('SPL is {}'.format(SPL))

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.scatter(range(100), rewards, s=8, color='green')
    plt.legend(loc='best')
    plt.title('end-to-end PPO performance')
    plt.show()

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the performance.')
    parser.add_argument('--filepath', type=str, default='/PPO/nav1/PPO_episode.txt', help='the file path under record dir')
    args = parser.parse_args()
    main(args)

