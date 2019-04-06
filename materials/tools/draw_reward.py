#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

# 创建画布
fig = plt.figure()
# 使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
# 将绘图区对象添加到画布中
fig.add_axes(ax)
# 通过set_axisline_style方法设置绘图区的底部及左侧坐标轴样式
# "-|>"代表实心箭头："->"代表空心箭头
ax.axis["bottom"].set_axisline_style("->", size = 1.5)
ax.axis["left"].set_axisline_style("->", size = 1.5)
# 通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
ax.axis["top"].set_visible(False)
ax.axis["right"].set_visible(False)

record_path = '../record/'


def main():
    f = open(record_path + 'PPO_nav1.txt', 'r')
    lines = f.readlines()
    reward = []    
    rewards = []

    for i in range(0, len(lines)-50, 50):
        for j in range(50):
            line = lines[i+j]  # eg: '[432.1290540951935, 248, True]'
            data = line.split()
            reward.append(float(data[0][1:-1]))  # 432.1290540951935
        avg_reward = np.mean(reward)
        rewards.append(avg_reward)
        reward = []
 
    f.close()

    f2 = open(record_path + 'E2E_PPO_nav1.txt', 'r')
    lines2 = f2.readlines()
    reward2 = []    
    rewards2 = []

    for i in range(0, len(lines2)-50, 50):
        for j in range(50):
            line2 = lines2[i+j]  # eg: '[432.1290540951935, 248, True]'
            data2 = line2.split()
            reward2.append(float(data2[0][1:-1]))  # 432.1290540951935
        avg_reward2 = np.mean(reward2)
        rewards2.append(avg_reward2)
        reward2 = []

    f2.close()

    fig = plt.gcf()
    plt.xlabel('50*episodes')
    plt.ylabel('average reward')
    plt.plot(range(len(rewards)), rewards)
    plt.plot(range(len(rewards2)), rewards2)
    plt.legend(['PPO', 'E2E_PPO']) 
    plt.title('maze1')
    fig.savefig('../result/maze1_dense_reward.png')
    plt.show()

main()
