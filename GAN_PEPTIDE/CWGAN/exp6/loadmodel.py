import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# class PeptideDataset(Dataset):
#     def __init__(self, pos_file, neg_file):
#         self.sequences = []
#         self.labels = []
#         self._load_data(pos_file, 1)
#         self._load_data(neg_file, 0)
#
#     # def __getitem__(self, index):
#     #     sequence = self.sequences[index]
#     #     label = self.labels[index]
#     #     # 将序列转换为张量
#     #     sequence_tensor = torch.zeros((26, len(sequence)), dtype=torch.float32)
#     #     for i, aa in enumerate(sequence):
#     #         sequence_tensor[ord(aa) - ord('A'), i] = 1
#     #     return sequence_tensor, label
#     def __getitem__(self, index, max_len=30):
#         sequence = self.sequences[index]
#         label = self.labels[index]
#         if max_len is not None and len(sequence) > max_len:
#             sequence = sequence[:max_len]
#         else:
#             sequence += 'B' * (max_len - len(sequence))
#         # 将序列转换为张量
#         sequence_tensor = torch.zeros((26, max_len), dtype=torch.float32)
#         for i, aa in enumerate(sequence):
#             sequence_tensor[ord(aa) - ord('A'), i] = 1
#         return sequence_tensor, label
#
#     def __len__(self):
#         return len(self.labels)
#
#     def _load_data(self, file_path, label):
#         with open(file_path, "r") as f:
#             for line in f:
#                 line = line.strip()
#                 if len(line) > 0:
#                     self.sequences.append(line)
#                     self.labels.append(label)
class PeptideDataset(Dataset):
    def __init__(self, pos_file, neg_file, max_seq_len=30):
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.labels = []
        self._load_data(pos_file, 1)
        self._load_data(neg_file, 0)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        # 将序列转换为张量
        sequence_tensor = torch.zeros((26, self.max_seq_len), dtype=torch.float32)
        for i, aa in enumerate(sequence):
            if i >= self.max_seq_len:
                break
            # print(i)
            sequence_tensor[ord(aa) - ord('A'), i] = 1
        return sequence_tensor, label

    def __len__(self):
        return len(self.labels)

    def _load_data(self, file_path, label):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    if len(line) > self.max_seq_len:
                        line = line[:self.max_seq_len]
                    else:
                        line = line + 'B' * (self.max_seq_len - len(line))
                    self.sequences.append(line)
                    self.labels.append(label)
class PeptideDataset2(Dataset):
    def __init__(self, pos_file, max_seq_len=30):
        self.max_seq_len = max_seq_len
        self.sequences = []
        self.labels = []
        self._load_data(pos_file, 1)
        # self._load_data(neg_file, 0)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        # 将序列转换为张量
        sequence_tensor = torch.zeros((26, self.max_seq_len), dtype=torch.float32)
        for i, aa in enumerate(sequence):
            if i >= self.max_seq_len:
                break
            # print(i)
            sequence_tensor[ord(aa) - ord('A'), i] = 1
        return sequence_tensor, label

    def __len__(self):
        return len(self.labels)

    def _load_data(self, file_path, label):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    if len(line) > self.max_seq_len:
                        line = line[:self.max_seq_len]
                    else:
                        line = line + 'B' * (self.max_seq_len - len(line))
                    self.sequences.append(line)
                    self.labels.append(label)

# class PeptideCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(26, 64, kernel_size=7, padding=3)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(896, 256)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         scores = smooth_sigmoid(x)[:,1]
#         # print(scores)
#         return x,scores
# class PeptideCNN(nn.Module):
#     def __init__(self, weight_decay=0.01):
#         super().__init__()
#         self.conv1 = nn.Conv1d(26, 64, kernel_size=7, padding=3)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(896, 256)
#         self.fc2 = nn.Linear(256, 2)
#         self.weight_decay = weight_decay
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         scores = smooth_sigmoid(x)[:,1]
#         # Add L2 regularization to the weights of the fully connected layers
#         l2_reg = self.weight_decay * (torch.sum(self.fc1.weight ** 2) + torch.sum(self.fc2.weight ** 2))
#         return x,scores - l2_reg
import torch.nn.functional as F
# from torch.utils.early_stopping import EarlyStopping

class PeptideCNN(nn.Module):
    def __init__(self, dropout_prob=0.5, l2_reg=0.01):
        super().__init__()
        self.conv1 = nn.Conv1d(26, 64, kernel_size=7, padding=3)
        self.drop1 = nn.Dropout(p=dropout_prob)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.drop2 = nn.Dropout(p=dropout_prob)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(896, 256)
        self.fc2 = nn.Linear(256, 2)
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        l2_loss = self.l2_reg * (torch.sum(torch.pow(self.fc1.weight, 2)) + torch.sum(torch.pow(self.fc2.weight, 2)))
        return x, F.sigmoid(x)[:, 1], l2_loss
# class PeptideCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(26, 64, kernel_size=7, padding=3)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 25, 256)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = self.pool2(x)
#         x = x.view(-1, 128 * 25)
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         return x
def smooth_sigmoid(x, alpha=0.1):
    return 1 / (1 + torch.exp(-alpha * x))


def use(model, test_loader, device):
    # model.eval()
    # total_loss = 0
    # total_correct = 0
    scores = []
    with torch.no_grad():
        for data, _ in test_loader:
            data= data.to(device)
            output,score,_ = model(data)
            scores.append(score)
    scores = [item for tensor in scores for item in torch.flatten(tensor).tolist()]
            # loss = criterion(output, target)
            # pred = output.argmax(dim=1, keepdim=True)
            # total_correct += pred.eq(target.view_as(pred)).sum().item()
            # total_loss += loss.item()
    return scores

# 加载模型
# 定义AA_TO_IDX和IDX_TO_AA
# AA_TO_IDX = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
#              'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20,
#              'U': 21, 'B': 22, 'Z': 23, 'O': 24, 'J': 25}
#
# IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
batch_size=32
# 加载模型
model = PeptideCNN()
model.load_state_dict(torch.load('single_amp_best_model0.pt',map_location={'cuda:3':'cuda:2'}))
model.to(device)
# 读取待预测序列txt
# s = []
test_dataset=PeptideDataset2('gen_wgan_edit.txt',30)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
s = use(model,test_loader,device)
# 读取未标注序列
# with open('sequence.txt', 'r') as f:
#     seq = f.read().strip()
#
# # 将序列转换为模型输入张量
# x = torch.zeros((len(seq), 25), dtype=torch.float32)
# for i, aa in enumerate(seq):
#     if aa in AA_TO_IDX:
#         x[i][AA_TO_IDX[aa]] = 1
#
# # 在第0维增加一维表示batch_size
# x = x.unsqueeze(0)
#
# # 模型预测
# with torch.no_grad():
#     output,score = model(x)
#     s.append(s)
#     # scores = F.sigmoid(output)
#     # pred = scores[0][0].item()
#
# # 打印预测结果
# print(f"预测分数为: {score:.2f}")
# with open('sequence.txt') as f:
#     sequence = [line.strip() for line in f]
# # with open('sequence.txt', 'r') as f:
# #     sequence = f.read().strip()
# # 将序列转换为数字编码
# for seq in sequence:
#     x = [AA_TO_IDX[aa] for aa in seq]
#     x = torch.LongTensor(x).unsqueeze(0)  # 增加一维batch_size=1
#
#     # 预测标签
#     model.eval()
#     with torch.no_grad():
#         y_pred, scores = model(x).cpu().numpy()
#         # scores = nn.functional.sigmoid(y_pred[0]).cpu().numpy()  # 将第一维的分数保存在scores列表中
#
#     # 输出每个位置的得分
#
#     for i, score in enumerate(scores):
#         s.append(score)
#         print(f'Position {i + 1}: {score}')
# s = torch.tensor(s).to(device)
import numpy as np
np.save('scores_e.npy',s)
# def main():
    # 设置参数
# batch_size = 32
# epochs = 200
# learning_rate = 0.00001
# pos_file = 'spos.txt'
# neg_file = 'sneg.txt'
#
# # 实例化数据集和数据加载器
# dataset = PeptideDataset(pos_file, neg_file)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型、优化器和损失函数
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# model = PeptideCNN().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
# # 记录训练过程中的准确率和损失
# train_losses = []
# train_accuracies = []
# test_losses = []
# test_accuracies = []
# best_test_accuracy = 0
# 训练和测试模型
# for epoch in range(epochs):
#     # print(epoch)
#     train_loss, train_acc,s1 = train(model, train_loader, optimizer, criterion, device)
#     test_loss, test_acc,s2 = test(model, test_loader, criterion, device)
#     train_losses.append(train_loss)
#     train_accuracies.append(train_acc)
#     test_losses.append(test_loss)
#     test_accuracies.append(test_acc)
#     if test_acc > best_test_accuracy:
#         best_test_accuracy = test_acc
#         torch.save(model.state_dict(), 'best_model.pt')
#     print(
#         f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - test_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}")
#     # print('train:',s1)
#     # print('test:', s2)
#  # 画出训练集和测试集准确率变化曲线
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(range(epochs), train_losses, label='Train')
# plt.plot(range(epochs), test_losses, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(epochs), train_accuracies, label='Train')
# plt.plot(range(epochs), test_accuracies, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()