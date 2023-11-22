# import matplotlib.pyplot as plt
#
# # 模拟数据
# data = {
#     'DNN': [0.829, 0.846, 0.869, 0.844, 0.875],
#     'LSTM': [0.852, 0.875, 0.876, 0.889, 0.898],
#     'VDCNN': [0.808, 0.859, 0.872, 0.880, 0.889],
#     'Transformer': [0.785, 0.801, 0.817, 0.809, 0.823],
# }
#
# # 横坐标为不同数据量
# x = ['orIginal data', '+500', '+1000', '+1500', '+2000']
#
# # 创建一个新图形
# fig, ax = plt.subplots(figsize=(8,6)) # 调整图形大小
#
# # 添加每个模型的数据线
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'] # 每个模型对应一种颜色
# styles = ['-', '--', '-.', ':'] # 每个模型对应一种线条样式
# for i, (model, accuracies) in enumerate(data.items()):
#     ax.plot(x, accuracies, label=model, color=colors[i], linestyle=styles[i])
#
# # 添加图例
# ax.legend(loc='lower right', fontsize=10) # 调整图例的位置和字体大小
#
# # 添加轴标签和标题，并调整字体大小
# # ax.set_xlabel('数据量', fontsize=14)
# ax.set_ylabel('ACC', fontsize=10,fontweight='bold')
# # ax.set_title('不同模型的准确率', fontsize=16)
#
# # 调整坐标轴的范围和刻度，并添加网格线
# # ax.set_xlim([900, 5100])
# ax.set_xticks(x, fontsize=10,fontweight='bold')
# ax.set_ylim([0.75, 0.9])
# ax.set_yticks([0.3, 0.5, 0.7, 0.9])
# ax.grid(color='gray', linestyle=':', linewidth=0.5)
#
# # 添加背景色
# ax.set_facecolor('#f8f8f8')
#
# # 显示图形
# plt.show()
import matplotlib.pyplot as plt

# 模拟数据
data = {
    'DNN': [0.804,0.809,0.806,0.801,0.798],
    'TextCNN': [0.860, 0.868, 0.866, 0.861, 0.857],
    'VDCNN': [0.824, 0.828, 0.822, 0.819, 0.808],
    'Transformer': [0.742, 0.742, 0.746, 0.748, 0.744],
}

# 横坐标为不同数据量
x = ['Original', '+200', '+300', '+400', '+500']

# x = ['Original', '0', '0.05', '0.1', '0.2']
# 创建一个新图形
fig, ax = plt.subplots(figsize=(8, 6))

# 添加每个模型的数据线
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
styles = ['-', '--', '-.', ':']
for i, (model, accuracies) in enumerate(data.items()):
    ax.plot(x, accuracies, label=model, color=colors[i], linestyle=styles[i])

# 添加图例
ax.legend(loc='lower right', fontsize=10)

# 添加轴标签和标题，并调整字体大小
ax.set_xlabel('Data Size', fontsize=12, fontweight='bold') # 修改横轴标签
ax.set_ylabel('ACC', fontsize=12, fontweight='bold')
ax.set_title('ACC of Different Models', fontsize=16, fontweight='bold') # 修改图标题

# 调整坐标轴的范围和刻度，并添加网格线
ax.set_xlim([-0.5, 4.5]) # 修改x轴范围
ax.set_xticks(range(len(x))) # 修改x轴刻度
ax.set_xticklabels(x, fontsize=10, fontweight='bold') # 修改x轴标签字体
ax.set_ylim([0.74, 0.9])
ax.set_yticks([0.74, 0.78, 0.82, 0.86,0.9], fontsize=10, fontweight='bold')
ax.set_yticklabels([0.74, 0.78, 0.82, 0.86,0.9], fontsize=10, fontweight='bold')
ax.grid(color='gray', linestyle=':', linewidth=0.5)

# 添加背景色
ax.set_facecolor('#f8f8f8')
plt.savefig('a3.pdf', dpi=500)
# 显示图形
plt.show()

data = {
    'DNN': [0.882,
0.887,
0.885,
0.883,
0.878
],
    'TextCNN': [0.927,
0.931,
0.928,
0.921,
0.916,
],
    'VDCNN': [0.899,
0.904,
0.903,
0.893,
0.889,
],
    'Transformer': [0.776,
0.775,
0.802,
0.799,
0.782,
],
}

# 横坐标为不同数据量
x = ['Original', '+200', '+300', '+400', '+500']

# 创建一个新图形
fig, ax = plt.subplots(figsize=(8, 6))

# 添加每个模型的数据线
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
styles = ['-', '--', '-.', ':']
for i, (model, accuracies) in enumerate(data.items()):
    ax.plot(x, accuracies, label=model, color=colors[i], linestyle=styles[i])

# 添加图例
ax.legend(loc='lower right', fontsize=10)

# 添加轴标签和标题，并调整字体大小
ax.set_xlabel('Data Size', fontsize=12, fontweight='bold') # 修改横轴标签
ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
ax.set_title('AUC of Different Models', fontsize=16, fontweight='bold') # 修改图标题

# 调整坐标轴的范围和刻度，并添加网格线
ax.set_xlim([-0.5, 4.5]) # 修改x轴范围
ax.set_xticks(range(len(x))) # 修改x轴刻度
ax.set_xticklabels(x, fontsize=10, fontweight='bold') # 修改x轴标签字体
ax.set_ylim([0.75, 0.93])
ax.set_yticks([0.75, 0.8, 0.85, 0.9,0.93])
ax.set_yticklabels([0.75, 0.8, 0.85, 0.9,0.93], fontsize=10, fontweight='bold')
ax.grid(color='gray', linestyle=':', linewidth=0.5)

# 添加背景色
ax.set_facecolor('#f8f8f8')
plt.savefig('b3.pdf', dpi=500)
# 显示图形
plt.show()