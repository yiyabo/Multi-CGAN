# //H G I E B
import numpy as np
# f=open('SecondaryStructurePrediction82.fasta')
f=open('ss.fasta')
txt=[]

score = []
for line in f:
    txt.append(line.strip())
for i in range(len(txt)):
    if i % 2 == 0:
        continue
    else:
        print(txt[i])
        print(len(txt[i]))
        count = 0
        for c in txt[i]:
            if c == 'H' or c == 'E' or c =='B' or c =='I' or c =='G':
                count += 1
        score.append(count/len(txt[i]))
print(score)
# np.save('cwgan_struct_score_NEW2.npy',score)

np.save('plots.npy',score)