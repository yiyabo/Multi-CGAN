file = open('newamp3.txt', 'r')

lines = file.readlines()
print(len(lines))
count = 0
res=[]
# two = []
for l in lines:
    l = l.strip()
    # if l.split('\t')[1] == 'AMP':
    #     two.append(count - 1)
    # count += (float)(l.split('\t')[-1])
    res.append((float)(l.split('\t')[-1]))
# print(count / len(lines))
import numpy as np
np.save('newamp3.npy',res)