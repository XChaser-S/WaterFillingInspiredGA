import matplotlib.pyplot as plt
import numpy as np

# # 获取SciPy默认颜色循环
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# # 生成示例数据
# x = np.arange(1, 11)
# y1_category1 = np.random.randint(50, 100, size=10)  # 折线图数据 - Category 1
# y1_category2 = np.random.randint(50, 100, size=10)  # 折线图数据 - Category 2
# y1_category3 = np.random.randint(50, 100, size=10)  # 折线图数据 - Category 3
#
# y2_category1 = np.random.randint(1, 10, size=10)  # 柱状图数据 - Category 1
# y2_category2 = np.random.randint(1, 10, size=10)  # 柱状图数据 - Category 2
# y2_category3 = np.random.randint(1, 10, size=10)  # 柱状图数据 - Category 3
#
# # 创建画布和子图对象
# fig, ax1 = plt.subplots()
#
# # 绘制折线图
# ax1.plot(x, y1_category1, color=colors[0], marker='o', label='Category 1')
# ax1.plot(x, y1_category2, color=colors[1], marker='s', label='Category 2')
# ax1.plot(x, y1_category3, color=colors[2], marker='^', label='Category 3')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Energy Consumption (J)', color=colors[0])
#
# # 创建第二个纵轴
# ax2 = ax1.twinx()
#
# # 绘制柱状图
# bar_width = 0.25
# bar1 = ax2.bar(x - bar_width, y2_category1, bar_width, color=colors[0], label='Category 1')
# bar2 = ax2.bar(x, y2_category2, bar_width, color=colors[1], label='Category 2')
# bar3 = ax2.bar(x + bar_width, y2_category3, bar_width, color=colors[2], label='Category 3')
# ax2.set_ylabel('Throughput (Gbps)')
#
# # 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# bars = [bar1, bar2, bar3]
# labels2 = ['Category 1', 'Category 2', 'Category 3']
# ax2.legend(bars + lines, labels2 + labels, loc='best')
#
# plt.title('Energy Consumption and Throughput')
# plt.show()

import matplotlib.patches as patches

# 渐变色柱状图
def gradient_bar(ax, x, y, width, height, color1, color2):
    for i in np.linspace(0, 1, 100):
        rect = patches.Rectangle((x, y + i * height), width, height / 100.0,
                                 color=(color1[0] * (1 - i) + color2[0] * i,
                                        color1[1] * (1 - i) + color2[1] * i,
                                        color1[2] * (1 - i) + color2[2] * i))
        ax.add_patch(rect)

# 数据
x = np.arange(5)
y = np.random.rand(5)

fig, ax = plt.subplots()
width = 0.5

# 绘制柱状图
for i in range(len(x)):
    gradient_bar(ax, x[i] - width / 2, 0, width, y[i], color1=(1, 0, 0), color2=(0, 0, 1))

ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim(0, 1)
plt.show()

