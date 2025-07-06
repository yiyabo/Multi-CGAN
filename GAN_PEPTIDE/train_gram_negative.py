#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用Gram-.fasta数据集训练CWGAN的脚本
基于原始CWGAN.py修改，适配新的数据集
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.optim as optim
import os

# 从原始文件导入必要的函数和类
def to_onehot(file_path):
    pep = np.load(file_path)
    pep = torch.LongTensor(pep)
    one_hot_pep = F.one_hot(pep, 21).reshape(-1, 30, 21)
    return one_hot_pep

def return_index(one_hot_coding):
    index = np.argwhere(one_hot_coding == 1)
    return index[:, -1].reshape(-1, 30)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes=2):  # 修改为支持可配置的类别数
        super(Discriminator, self).__init__()
        self.emb = self._emb(num_classes, 20)  # 使用可配置的类别数
        self.linear = nn.Sequential(
            nn.Linear(30*21+20, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 30 * 21),
        )
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=(10,1), stride=1, padding=0),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 6, 1, 0),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def _emb(self, label, dim):
        return nn.Embedding(label, dim)
    
    def forward(self, x, label):
        label = label.squeeze().squeeze()
        label = self.emb(label)
        x = x.view(-1, 30*21)
        x = torch.cat([x, label], 1)
        x = self.linear(x)
        x = x.reshape(-1, 1, 30, 21)
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes=2):  # 修改为支持可配置的类别数
        super(Generator, self).__init__()
        self.emb = self._emb(num_classes, 20)  # 使用可配置的类别数
        self.net = nn.Sequential(
            self._block(channels_noise+20, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  #  8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  #  16x16
            self._block(features_g * 4, features_g * 2, 6, 1, 0),  #  21*21
            self._block(features_g * 2, channels_img, (10,1), 1, 0),  # 30*21
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def _emb(self, label, dim):
        return nn.Embedding(label, dim)

    def forward(self, x, label):
        label = label.squeeze().squeeze()
        label = self.emb(label)
        if len(label.shape) == 1:  # 处理单个样本的情况
            label = label.unsqueeze(0)
        label = label.unsqueeze(2)
        label = label.unsqueeze(3)
        x = torch.cat([x, label], 1)
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, label, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    mixed_scores = critic(interpolated_images, label)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def save_checkpoint(state, filename):
    print(f"=> 保存检查点到 {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint, gen, disc):
    print("=> 加载检查点")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

def main():
    # 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 训练参数 - 针对小数据集优化
    LEARNING_RATE = 5e-5  # 降低学习率
    BATCH_SIZE = 16       # 减小批次大小
    IMAGE_SIZE = (30, 21)
    CHANNELS_IMG = 1
    Z_DIM = 90
    NUM_EPOCHS = 500      # 增加训练轮数
    FEATURES_CRITIC = 16
    FEATURES_GEN = 16
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10
    NUM_CLASSES = 1       # 只有一个类别：革兰氏阴性菌抗菌肽
    
    print(f"训练参数:")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  类别数: {NUM_CLASSES}")
    
    # 加载数据
    print("正在加载训练数据...")
    train_data = to_onehot('data/gram_negative_train_pos.npy')
    train_labels = np.load('data/gram_negative_train_labels.npy')
    train_labels = torch.LongTensor(train_labels)
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    
    # 创建数据加载器
    loader = load_array((train_data, train_labels), BATCH_SIZE)
    
    # 创建模型
    print("正在创建模型...")
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES).to(device)
    critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES).to(device)
    
    # 初始化权重
    initialize_weights(gen)
    initialize_weights(critic)
    
    # 优化器
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
    fixed_labels = torch.zeros(BATCH_SIZE, 1, dtype=torch.long).to(device)  # 全部使用标签0
    
    # 创建保存目录
    save_dir = "gram_negative_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    gen.train()
    critic.train()
    
    loss_g_history = []
    loss_d_history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss_g = []
        epoch_loss_d = []
        
        for batch_idx, (real, label) in enumerate(loader):
            real = real.unsqueeze(1).to(device)
            label = label.to(device)
            cur_batch_size = real.shape[0]
            
            # 训练判别器
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
                fake = gen(noise, label)
                
                critic_real = critic(real, label).reshape(-1)
                critic_fake = critic(fake, label).reshape(-1)
                gp = gradient_penalty(critic, label, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
            
            # 训练生成器
            gen_fake = critic(fake, label).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            epoch_loss_g.append(loss_gen.item())
            epoch_loss_d.append(loss_critic.item())
            
            if batch_idx % 20 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
                )
        
        # 记录每个epoch的平均损失
        avg_loss_g = np.mean(epoch_loss_g)
        avg_loss_d = np.mean(epoch_loss_d)
        loss_g_history.append(avg_loss_g)
        loss_d_history.append(avg_loss_d)
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] 平均损失 - D: {avg_loss_d:.4f}, G: {avg_loss_g:.4f}")
        
        # 定期保存模型和生成样本
        if (epoch + 1) % 50 == 0:
            # 保存模型
            checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint({
                'epoch': epoch + 1,
                'gen': gen.state_dict(),
                'critic': critic.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'loss_g': avg_loss_g,
                'loss_d': avg_loss_d,
            }, checkpoint_path)
            
            # 生成样本
            gen.eval()
            with torch.no_grad():
                fake_samples = gen(fixed_noise, fixed_labels)
                # 转换为序列索引
                fake_indices = return_index(fake_samples.cpu())
                # 保存生成的序列
                np.save(f"{save_dir}/generated_epoch_{epoch+1}.npy", fake_indices)
            gen.train()
    
    print("训练完成!")
    
    # 保存最终模型
    final_checkpoint_path = f"{save_dir}/final_model.pth"
    save_checkpoint({
        'epoch': NUM_EPOCHS,
        'gen': gen.state_dict(),
        'critic': critic.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_critic': opt_critic.state_dict(),
        'loss_g_history': loss_g_history,
        'loss_d_history': loss_d_history,
    }, final_checkpoint_path)
    
    # 保存损失历史
    np.save(f"{save_dir}/loss_history.npy", {
        'generator': loss_g_history,
        'discriminator': loss_d_history
    })
    
    print(f"所有结果已保存到 {save_dir} 目录")

if __name__ == "__main__":
    main()
