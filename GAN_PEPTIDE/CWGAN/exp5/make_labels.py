import numpy as np
scores = np.load('cwgan_struct_score_NEW2.npy',allow_pickle=True)
print(scores)
labels = []
# for score in scores:
#     if score < 0.6:
#         labels.append(0)
#     else:
#         labels.append(1)
# np.save('cwgan_struct_label_new.npy',labels)
total = 0
for score in scores:
    total += score
print(total / len(scores))