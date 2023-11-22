# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# @author: yhq
# @contact:1318231697@qq.com
# @version: 1.0.0
# @file: boundary.py
# @time: 2022/6/14 10:23
# """
# import pickle
# import numpy as np
# # f = open(r'laten.pickle','rb')
# # data = pickle.load(f)
# # print(data)
# from sklearn import svm
#
# chosen_num_or_ratio=0.4
# split_ratio=0.8
# latent_codes = np.load('laten_codes.npy')
#
# scores = np.load('scores_n.npy', allow_pickle=True)
# count = 0
# # for i in range(scores.shape[0]):
# #     if scores[i][0] > 0.5:
# #         count = count + 1
# # radio = count / scores.shape[0]
# # print(radio)
# # print(a[0].shape)
# sorted_idx = np.argsort(scores, axis=0)[::-1][:22000]
# # print(scores[sorted_idx])
# latent_codes = latent_codes[sorted_idx]
# scores = scores[sorted_idx]
# print(latent_codes.shape)
# print(scores.shape)
# num_samples = latent_codes.shape[0]
# if 0 < chosen_num_or_ratio <= 1:
#     chosen_num = int(num_samples * chosen_num_or_ratio)
# else:
#     chosen_num = int(chosen_num_or_ratio)
#
# # chosen_num = min(chosen_num, num_samples // 2)
# num2 = int(num_samples * 0.4)
#
# train_num = int(chosen_num * split_ratio)
# val_num = chosen_num - train_num
# # Positive samples.
# positive_idx = np.arange(chosen_num)
# np.random.shuffle(positive_idx)
# positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
# positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
# # Negative samples.
# train_num2 = int(num2 * split_ratio)
# val_num2 = num2 - train_num2
# negative_idx = np.arange(num2)
# np.random.shuffle(negative_idx)
# negative_train = latent_codes[-num2:][negative_idx[:train_num2]]
# print(scores[-num2:][negative_idx[:train_num2]])
# negative_val = latent_codes[-num2:][negative_idx[train_num2:]]
# # Training set.
# train_data = np.concatenate([positive_train, negative_train], axis=0)
# train_label = np.concatenate([np.ones(train_num, dtype=np.int),
#                               np.zeros(train_num2, dtype=np.int)], axis=0)
#
# # Validation set.
# val_data = np.concatenate([positive_val, negative_val], axis=0)
# val_label = np.concatenate([np.ones(val_num, dtype=np.int),
#                             np.zeros(val_num2, dtype=np.int)], axis=0)
#
# # Remaining set.
# remaining_num = num_samples - chosen_num -num2
# remaining_data = latent_codes[chosen_num:-num2]
# remaining_scores = scores[chosen_num:-num2]
# decision_value = (scores[0] + scores[-1]) / 2
# remaining_label = np.ones(remaining_num, dtype=np.int)
# # remaining_label[remaining_scores.ravel() < decision_value] = 0
# for i in range((remaining_num)):
#     if remaining_scores[i] < 0.5:
#         remaining_label[i] = 0
# remaining_positive_num = np.sum(remaining_label == 1)
# remaining_negative_num = np.sum(remaining_label == 0)
#
# clf = svm.SVC(kernel='linear')
# classifier = clf.fit(train_data, train_label)
#
#
# if val_num:
#     val_prediction = classifier.predict(val_data)
#     correct_num = np.sum(val_label == val_prediction)
#     print(correct_num, '/', val_num)
#
# if remaining_num:
#     remaining_prediction = classifier.predict(remaining_data)
#     correct_num = np.sum(remaining_label == remaining_prediction)
#     print(correct_num,'/',remaining_num)
#
#
# a = classifier.coef_.reshape(1, 100).astype(np.float32)
# boundary = a / np.linalg.norm(a)
# print(boundary.shape)
# np.save('toxicity_boundary.npy', boundary)


import numpy as np
from sklearn.svm import LinearSVC

# 从磁盘加载latent_codes和scores
latent_codes = np.load('laten_codes.npy')
scores = np.load('amp_score.npy')

# 按照得分从大到小对scores进行排序
sorted_indices = np.argsort(scores)[::-1]
sorted_scores = scores[sorted_indices]

# 获取得分大于0.5的数量
x = len(np.where(sorted_scores > 0.5)[0])


selected_indices = sorted_indices

# 构造正样本和负样本
X_train = latent_codes[selected_indices, :]
y_train = np.zeros(len(selected_indices))
y_train[:x] = 1

# 训练线性SVM模型
# svm = LinearSVC()
svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)

# svm.fit(X_train, y_train)

# 保存分类边界
boundary = svm.coef_.reshape(1, 100).astype(np.float32)
np.save('boundary_a.npy', boundary)
