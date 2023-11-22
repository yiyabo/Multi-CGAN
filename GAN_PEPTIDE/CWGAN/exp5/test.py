from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 假设y_true为真实标签，y_score为预测概率

data = pd.read_csv('test_stack.csv')


y_true =  np.array(data.real_label).tolist()
y_score =  np.array(data.sore_4).tolist()


# y_true = [0, 0, 1, 1]
# y_score = [0.1, 0.4, 0.35, 0.8]

# 计算fpr, tpr和阈值
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr,tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr,tpr,color='blue',label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

# 找到阈值为0.1时对应的fpr和tpr的索引
idx = np.argmin(np.abs(thresholds - 0.1))

# 在ROC曲线上标记该点
plt.scatter(fpr[idx], tpr[idx], marker='o', color='black', label='Threshold=0.10')

# 显示图像
plt.show()