import numpy as np
scores=np.load('plots.npy')
index = []
for i in range(len(scores)):
    if scores[i] >= 0.6:
        index.append(i)
origin = open('exp5b.txt','r')
lines = origin.readlines()
with open('sspos.txt','w')as f:
    for i in index:

        f.write(lines[index].strip() + '\n')