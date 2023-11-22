import matplotlib.pyplot as plt
import numpy as np
# for patch in ax.artists:
#         r, g, b, a = patch.get_facecolor()
#         patch.set_facecolor((r, g, b, .3))
x = np.arange(3)
y = [0.884,0.848,0.821]
# y1=0.885
# y2 = 0.851
# y3 = 0.892
# y1 = [0.885,0.791,0.569]
# 多数据并列柱状图
bar_width = 0.35
# tick_label=[ 'T', 'S','A']
tick_label=[ 'A&T', 'A&S','T&S']
y1 = [0.904,0.888,0.855]
y2 = [0.909,0.861,0.878]
plt.style.use("default")
plt.bar(x, y, bar_width, align='center',  color=['#f1828d','#c9f29b','#9f5afd'])
# plt.bar(x, y1, bar_width, align='center', color=['#9f5afd','#9f5afd','#f1828d'])
# plt.bar(x+bar_width, y2, bar_width, align='center', color=['#f1828d','#c9f29b','#c9f29b'])
plt.xlabel('Attribute',fontsize=10,fontweight='bold')
plt.ylabel('ACC_binary',fontsize=10,fontweight='bold')
# plt.xticks(x, tick_label,fontsize=10,fontweight='bold')
plt.xticks(x+bar_width/2, tick_label,fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
# plt.figure(dpi=500)
# plt.legend()
plt.savefig('a.pdf',dpi=500)
plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(3)
# y = [0.884,0.848,0.821]
# # y1=0.885
# # y2 = 0.851
# # y3 = 0.892
# # y1 = [0.885,0.791,0.569]
# # 多数据并列柱状图
# bar_width = 0.35
# # tick_label=[ 'T', 'S','A']
# tick_label=[ 'A&T', 'A&S','T&S']
# y1 = [0.904,0.888,0.855]
# y2 = [0.909,0.861,0.878]
#
# fig, ax = plt.subplots() # 创建figure和ax对象
#
# ax.bar(x, y1, bar_width, align='center', color=['#9f5afd','#9f5afd','#f1828d'])
# ax.bar(x+bar_width, y2, bar_width, align='center', color=['#f1828d','#c9f29b','#c9f29b'])
#
# for patch in ax.artists: # 循环遍历ax对象中的所有图形元素并设置透明度
#     r, g, b, a = patch.get_facecolor()
#     patch.set_facecolor((r, g, b, .3))
#
# ax.set_xlabel('Attribute')
# ax.set_ylabel('ACC_binary')
# ax.set_xticks(x+bar_width/2)
# ax.set_xticklabels(tick_label)
#
# plt.show()
