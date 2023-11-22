import numpy as np
import matplotlib.pyplot as plt
from pylab import *
# matplotlib.use('Agg')
def kmer_total(k,data_path):
    f = open(data_path,'r')
    seqs = f.readlines()[:1000]
    kmer = {}
    for seq in seqs:
        seq = seq.strip('\n')
        seq = seq.strip()
        if len(seq) > k - 1:
            for i in range(len(seq) - k):
                s = seq[i:i+k]
                if s not in kmer:
                    kmer[s] = 1
                else:
                    kmer[s] += 1
    return kmer
def count_num(kmer):
    total = 0
    for value in kmer.values():
        total += value
    return total
def count_type(kmer):
    total = 0
    for value in kmer.keys():
        total += 1
    return total


def topk(k,kmer):
    oder = sorted(kmer.items(),key=lambda x:x[1],reverse=True)
    res = []
    for i in range(k):
        res.append(oder[i])
    return res
def average_kemr(kmer):
    sum_kmer = 0
    total = count_type(kmer)
    # for i in range(k):
    #     total *= 20
    # num = count_num(kmer)
    for i in kmer.values():
        if i == 1:
            sum_kmer += 1

    return sum_kmer/total
kmer = kmer_total(4,'../db/input.txt')
# kmer = kmer_total(3,'../data/amp.txt')
num = count_num(kmer)
# type = count_type(kmer)
top = topk(10,kmer)
for r in top:
    print(r[0],r[1]/num)
# print(top)
# print(num)
# print(type)
# print([x[1] for x in top])

# print(average_kemr(kmer))
# print(kmer)
# print('...............')
kmer2 = kmer_total(4,'../db/output.txt')
num2= count_num(kmer2)
# type2 = count_type(kmer2)
top2 = topk(10,kmer2)
for r in top2:
    print(r[0], r[1] / num2)
##########
# print(num2)
# print(type2)
# print(top2)

# print(average_kemr(kmer2))
# mpl.rcParams['font.sans-serif'] = ['SimHei']

# x = np.array([x[1] for x in top])  #2015年
# y = np.array([x[1] for x in top2])  #2017年

#
# plt.figure(figsize=(12,8))
# plt.barh(range(len(y)), -x,color='darkorange',label='gen')
# plt.barh(range(len(x)), y,color='limegreen',label='real')
#
#
# # plt.xlim((-15000,15000))
# # plt.xticks((-15000,-10000,-5000,0,5000,10000,15000),('15000','10000','5000','0','5000','10000','15000'))
# # plt.yticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
# #            ('0-4岁', '5-9岁', '10-14岁', '15-19岁', '20-24岁', '25-29岁', '30-34岁', '35-39岁',
# #             '40-44岁', '45-49岁', '50-54岁', '55-59岁', '60-64岁', '65-69岁', '70-74岁', '75-79岁',
# #             '80-84岁', '85-89岁', '90-94岁', '95以上'))
# # plt.xlabel('人口数量（万）')
#
# plt.legend()
# plt.show()
# total =top
# total.extend(top2)
# print(total)
t1 = [m for m in top]
t2 = [m for m in top2]
# oder = sorted(total,key=lambda x:x[1],reverse=True)
# df = {}
# x = np.array([y[1] for y in top])
# df['mpg_z'] = (x - x.mean())/x.std()
# c = ['red' if x[1] in t2 else 'green' for x in oder]
# print(c)
total = []
for i in t1:
    total.append(i)
for i in t2:
    total.append(i)
total = sorted(total,key=lambda x:x[1],reverse=True)
y1=[]
y2=[]
for i in range(20):
    if total[i] in t1:
        y1.append(i+1)
        print(total[i][1]/num)
    else:
        y2.append(i+1)
        print(0-total[i][1] / num2)
# Draw plot
plt.figure(figsize=(14,10))
# test=[t[0] for t in oder]
plt.hlines(y=y1, xmin=0, xmax=[t[1]*100/num for t in t1],  color='#d91e18', alpha=0.4, linewidth=7)
plt.hlines(y=y2, xmin=[0-t[1]*100/num2 for t in t2], xmax=0,  color='#16a085', alpha=0.4, linewidth=7)

# Decorations
# plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
plt.yticks(np.arange(1, 21, 1),[t[0] for t in total], fontsize=10,fontweight='bold')
# plt.xticks(np.linspace(-1, 1, 21), fontsize=15,fontweight='bold')
plt.xticks(np.linspace(-0.4, 0.4, 11), fontsize=10,fontweight='bold')

# plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
# plt.grid(linestyle='--', alpha=0.5)
plt.savefig('b.pdf',dpi=500)
plt.show()