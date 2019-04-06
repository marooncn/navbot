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
    success = []    
    successes = []

    for i in range(len(lines)):
        if i < 100:
            success.append(0)
        else:
            for j in range(100):
                line = lines[i-j]  # eg: '[432.1290540951935, 248, True]'
                data = line.split()
                # successes.append(bool(data[2][:-1]))  # bool('False') is True!
                success.append(data[2][:-1] == str('True'))
        success_rate = sum(success)
        successes.append(success_rate)
        success = []
 
    f.close()

    f2 = open(record_path + 'E2E_PPO_nav1.txt', 'r')
    lines2 = f2.readlines()
    success2 = []    
    successes2 = []

    for i in range(100, len(lines2)):
        if i < 100:
            success2.append(0)
        else:
            for j in range(100):
                line2 = lines2[i-j]  # eg: '[432.1290540951935, 248, True]'
                data2 = line2.split()
                # successes.append(bool(data[2][:-1]))  # bool('False') is True!
                success2.append(data2[2][:-1] == str('True'))
        success_rate2 = sum(success2)
        successes2.append(success_rate2)
        success2 = []

    f2.close()

    fig = plt.gcf()
    plt.xlabel('episode')
    plt.ylabel('success rate(%)')
    #plt.plot(range(len(successes)), successes)
    plt.plot(range(len(successes2)), successes2)
    #plt.legend(['PPO', 'E2E_PPO'])
    plt.title('maze1')
    fig.savefig('../result/maze1_dense_success.png')
    plt.show()


main()
