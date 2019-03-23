#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def main(args):
    f = open('PPO_episode1000.txt', 'r')
    lines = f.readlines()
    reward = []    
    rewards = []

    for i in range(0, len(lines)-10, 10):
        for j in range(30):
            line = lines[i+j]  # eg: '[432.1290540951935, 248, True]'
            data = line.split()
            reward.append(float(data[0][1:-1]))  # 432.1290540951935
        avg_reward = np.mean(reward)
        rewards.append(avg_reward)
        reward = []
 
    f.close()

    f2 = open('PPO_remap_episode1000.txt', 'r')
    lines2 = f2.readlines()
    reward2 = []    
    rewards2 = []

    for i in range(0, len(lines2)-10, 10):
        for j in range(10):
            line2 = lines2[i+j]  # eg: '[432.1290540951935, 248, True]'
            data2 = line2.split()
            reward2.append(float(data2[0][1:-1]))  # 432.1290540951935
        avg_reward2 = np.mean(reward2)
        rewards2.append(avg_reward2)
        reward2 = []

    f2.close()

    plt.xlabel('10*episode')
    plt.ylabel('average reward')
    plt.plot(range(len(rewards)), rewards)
    plt.plot(range(len(rewards2)), rewards2)
    plt.legend(['PPO', 'PPO_remap']) 
    plt.title('performance')
    plt.show()
    plt.savefig('temp.png') 

