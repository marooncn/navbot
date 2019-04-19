#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.figure(figsize=(10, 6))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

record_path = '../materials/record/'


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

    plt.plot(range(len(successes)), successes, color="blue", label="Proposed", linewidth=1.5)
    plt.plot(range(len(successes2)), successes2, color="green", label="Benchmark", linewidth=1.5)

    size = 22
    plt.xticks(fontsize=size)  # 默认字体大小为10
    plt.yticks(fontsize=size)
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("episode", fontsize=size)
    plt.ylabel("success rate(%)", fontsize=size)

    plt.title('maze1', fontsize=size)
    # plt.legend()   # 显示各曲线的图例
    plt.legend(loc=4, numpoints=1)  # lower right
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=size)  # 设置图例字体的大小和粗细
    
    axes = plt.gca()
    # axes.set_xlim([None, None])  # 限定X轴的范围

    plt.savefig('../result/maze1_dense_success.png')
    plt.show()


main()
