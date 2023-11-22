

def create_dict(words):
    word_dict={}
    word_index=0
    for word in words:
        word_dict[word]=word_index
        word_index+=1
    return word_dict
def count20(data_path):
    file = open(data_path,'r')
    datas = file.readlines()
    words = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pep = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    res = []
    word_dict = create_dict(words)
    total = 0
    for data in datas:
        for char in data:
            if char in words:
                total += 1
                pep[word_dict[char]] += 1
    for p in pep:
        res.append(p/total)
    return res
pep =count20('../db/input.txt')
# pep =pep / total
pep2 = count20('../db/output.txt')
# pep2 = pep2 /total

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(20)
y = pep2
y1 = pep
# 多数据并列柱状图
bar_width = 0.35
tick_label=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# plt.figure(dpi= 500)
plt.bar(x, y, bar_width, align='center', color='#66c2a5', label='fake')
plt.bar(x+bar_width, y1, bar_width, align='center', color='#8da0cb', label='real')
plt.xlabel('Amino acids',fontsize=10,fontweight='bold')
plt.ylabel('Fraction',fontsize=10,fontweight='bold')
plt.xticks(x+bar_width/2, tick_label,fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
plt.legend()
# plt.figure(dpi=300)
plt.savefig('a.pdf',dpi=500)
plt.show()


# plt.rcParams['font.sans-serif']=['SimHei']
# plt.figure(figsize=(7.5,5),dpi=80) #调节画布的大小
# labels = tick_label #定义各个扇形的面积/标签
# sizes = pep2 #各个值，影响各个扇形的面积
# sizes2 = pep
# colors = ['#e52d5d', '#e01b77', '#d31b92', '#bb2cad', '#983fc5', '#7b60e0', '#547bf3', '#0091ff', '#00b6ff',
#                '#00d0da',
#                '#00df83', '#a4e312','#DE8787', '#DEB487', '#D4DE87', '#9FDE87', '#87DEBA', '#87B9DE', '#8C87DE', '#C587DE'] #每块扇形的颜色
# explode = (0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01)
# patches,text1,text2 = plt.pie(sizes,
#                       explode=explode,
#                       labels=labels,
#                       colors=colors,
#                       labeldistance = 1.2,#图例距圆心半径倍距离
#                       autopct = '%3.2f%%', #数值保留固定小数位
#                       shadow = False, #无阴影设置
#                       startangle =90, #逆时针起始角度设置
#                       pctdistance = 0.6) #数值距圆心半径倍数距离
# #patches饼图的返回值，texts1为饼图外label的文本，texts2为饼图内部文本
# plt.axis('equal')
# plt.legend()
# plt.show()
datas = []
for (p1,p2) in zip(pep,pep2):
    datas.append(p1-p2)
data_y1=[]
data_y2=[]
for i in range(20):
    if datas[i] > 0:
        data_y1.append(i+1)
    else:
        data_y2.append(i+1)
fig=plt.figure(figsize=(6.4,4.8))
print(fig.get_size_inches())
# test=[t[0] for t in oder]
plt.vlines(x=data_y1, ymin=0, ymax=[datas[t - 1] for t in data_y1],  color='#d91e18', alpha=0.4, linewidth=7)
plt.vlines(x=data_y2, ymin=[datas[t - 1] for t in data_y2], ymax=0,  color='#16a085', alpha=0.4, linewidth=7)

# Decorations
# plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
plt.xticks(np.arange(1, 21, 1),tick_label,fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold')
# plt.yticks(fontsize=15)
plt.xlabel('Amino acids',fontsize=10,fontweight='bold')
plt.ylabel('Difference in Fraction',fontsize=10,fontweight='bold')
# plt.xticks(np.arange(-90, 90, 10), fontsize=12)
# plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
# plt.grid(linestyle='--', alpha=0.5)
plt.savefig('b.pdf',dpi=500)
plt.show()