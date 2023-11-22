import  numpy as np
def amp_label(data_path,npy_path):
    file = open(data_path, 'r')
    lines = file.readlines()
    count = 0
    score = []
    for l in lines:
        # if l.split('\t')[1] == 'AMP':
        score.append((float)(l.split('\t')[-1].strip()))
        count += 1
    np.save(npy_path, score)
#
# def toxicity_label(data_path,label_path):
#     file = open(data_path, 'r')
#     lines = file.readlines()
#     count = 0
#     two = []
#     for l in lines:
#         if l.split('\t')[1] == '0':
#             two.append(count)
#         count += 1
#     np.save(label_path, two)
# def toxicity_label(data_path,label_path):
#     file = open(data_path, 'r')
#     lines = file.readlines()
#     count = 0
#     two = []
#     for l in lines:
#         if l.split('\t')[1] == '0':
#             two.append(count)
#         count += 1
#     np.save(label_path, two)

# toxicity_label('labels/toxicity_res.txt','labels/toxicity_res.npy')
amp_label('res3.txt','amp_score3.npy')
# toxicity_label('labels/exp4.txt','labels/exp4.npy')