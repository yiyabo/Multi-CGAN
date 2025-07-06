#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
服务器版本的Gram-.fasta数据集CWGAN训练脚本
支持GPU、后台运行、日志记录、断点续训等功能
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.optim as optim
import os
import time
import logging
import argparse
import json
from datetime import datetime

# 设置日志
def setup_logging(log_dir):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 从原始文件导入必要的函数和类
def to_onehot(file_path):
    pep = np.load(file_path)
    pep = torch.LongTensor(pep)
    one_hot_pep = F.one_hot(pep, 21).reshape(-1, 30, 21)
    return one_hot_pep

def return_index(one_hot_coding):
    """将one-hot编码或softmax输出转换为索引"""
    # 如果输入是torch张量，转换为numpy
    if hasattr(one_hot_coding, 'numpy'):
        one_hot_coding = one_hot_coding.numpy()

    # 如果是连续值（如tanh输出），使用argmax
    if one_hot_coding.dtype == np.float32 or one_hot_coding.dtype == np.float64:
        # 对最后一个维度取argmax
        indices = np.argmax(one_hot_coding, axis=-1)
        return indices.reshape(-1, 30)
    else:
        # 原始的one-hot处理方式
        index = np.argwhere(one_hot_coding == 1)
        if len(index) == 0:
            # 如果没有找到1，使用argmax作为备选
            indices = np.argmax(one_hot_coding, axis=-1)
            return indices.reshape(-1, 30)
        return index[:, -1].reshape(-1, 30)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes=2):
        super(Discriminator, self).__init__()
        self.emb = self._emb(num_classes, 20)
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
    def __init__(self, channels_noise, channels_img, features_g, num_classes=2):
        super(Generator, self).__init__()
        self.emb = self._emb(num_classes, 20)
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
        if len(label.shape) == 1:
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

def save_checkpoint(state, filename, logger):
    logger.info(f"保存检查点到 {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, gen, critic, opt_gen, opt_critic, logger):
    logger.info(f"从 {checkpoint_path} 加载检查点")
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_critic.load_state_dict(checkpoint['opt_critic'])
    return checkpoint['epoch'], checkpoint.get('loss_g_history', []), checkpoint.get('loss_d_history', [])

def save_config(config, save_dir):
    """保存训练配置"""
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def generate_samples(gen, device, num_samples, z_dim, save_path, logger):
    """生成样本并保存"""
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
        labels = torch.zeros(num_samples, 1, dtype=torch.long).to(device)
        fake_samples = gen(noise, labels)

        # 将tanh输出转换为概率分布，然后转换为索引
        # tanh输出范围是[-1,1]，转换为[0,1]
        fake_samples_normalized = (fake_samples + 1) / 2

        # 应用softmax使其成为概率分布
        fake_samples_prob = torch.softmax(fake_samples_normalized.view(-1, 30, 21), dim=-1)

        # 转换为索引
        fake_indices = return_index(fake_samples_prob.cpu())
        np.save(save_path, fake_indices)
        logger.info(f"生成 {num_samples} 个样本保存到 {save_path}")
    gen.train()

def main():
    parser = argparse.ArgumentParser(description='训练Gram-负性菌抗菌肽生成模型')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='gram_negative_results_server', help='结果保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--save_interval', type=int, default=50, help='保存间隔')
    parser.add_argument('--sample_interval', type=int, default=100, help='生成样本间隔')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("开始训练Gram-负性菌抗菌肽生成模型")
    
    # 设备配置
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"使用GPU: {device}")
    else:
        device = "cpu"
        logger.info("使用CPU")
    
    # 训练参数
    config = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'z_dim': 90,
        'features_critic': 16,
        'features_gen': 16,
        'critic_iterations': 5,
        'lambda_gp': 10,
        'num_classes': 1,
        'device': device
    }
    
    logger.info(f"训练配置: {config}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    save_config(config, args.save_dir)
    
    # 加载数据
    logger.info("加载训练数据...")
    train_data = to_onehot(os.path.join(args.data_dir, 'gram_negative_train_pos.npy'))
    train_labels = np.load(os.path.join(args.data_dir, 'gram_negative_train_labels.npy'))
    train_labels = torch.LongTensor(train_labels)
    
    logger.info(f"训练数据形状: {train_data.shape}")
    logger.info(f"训练标签形状: {train_labels.shape}")
    
    # 创建数据加载器
    loader = load_array((train_data, train_labels), config['batch_size'])
    
    # 创建模型
    logger.info("创建模型...")
    gen = Generator(config['z_dim'], 1, config['features_gen'], config['num_classes']).to(device)
    critic = Discriminator(1, config['features_critic'], config['num_classes']).to(device)
    
    # 初始化权重
    initialize_weights(gen)
    initialize_weights(critic)
    
    # 优化器
    opt_gen = optim.Adam(gen.parameters(), lr=config['learning_rate'], betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=config['learning_rate'], betas=(0.0, 0.9))
    
    # 恢复训练或从头开始
    start_epoch = 0
    loss_g_history = []
    loss_d_history = []
    
    if args.resume:
        start_epoch, loss_g_history, loss_d_history = load_checkpoint(
            args.resume, gen, critic, opt_gen, opt_critic, logger)
    
    # 固定噪声用于生成样本
    fixed_noise = torch.randn(config['batch_size'], config['z_dim'], 1, 1).to(device)
    fixed_labels = torch.zeros(config['batch_size'], 1, dtype=torch.long).to(device)
    
    # 训练循环
    logger.info(f"开始训练，从第 {start_epoch} 轮开始...")
    gen.train()
    critic.train()
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        epoch_loss_g = []
        epoch_loss_d = []
        
        for batch_idx, (real, label) in enumerate(loader):
            real = real.unsqueeze(1).to(device)
            label = label.to(device)
            cur_batch_size = real.shape[0]
            
            # 训练判别器
            for _ in range(config['critic_iterations']):
                noise = torch.randn(cur_batch_size, config['z_dim'], 1, 1).to(device)
                fake = gen(noise, label)
                
                critic_real = critic(real, label).reshape(-1)
                critic_fake = critic(fake, label).reshape(-1)
                gp = gradient_penalty(critic, label, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + config['lambda_gp'] * gp
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
            
            if batch_idx % 20 == 0:
                logger.info(
                    f"Epoch [{epoch}/{config['num_epochs']}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
                )
        
        # 记录每个epoch的平均损失
        avg_loss_g = np.mean(epoch_loss_g)
        avg_loss_d = np.mean(epoch_loss_d)
        loss_g_history.append(avg_loss_g)
        loss_d_history.append(avg_loss_d)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch [{epoch}/{config['num_epochs']}] 完成 "
            f"平均损失 - D: {avg_loss_d:.4f}, G: {avg_loss_g:.4f} "
            f"用时: {epoch_time:.2f}s"
        )
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'gen': gen.state_dict(),
                'critic': critic.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'loss_g': avg_loss_g,
                'loss_d': avg_loss_d,
                'loss_g_history': loss_g_history,
                'loss_d_history': loss_d_history,
                'config': config
            }, checkpoint_path, logger)
        
        # 定期生成样本
        if (epoch + 1) % args.sample_interval == 0:
            sample_path = os.path.join(args.save_dir, f"generated_epoch_{epoch+1}.npy")
            generate_samples(gen, device, config['batch_size'], config['z_dim'], sample_path, logger)
    
    logger.info("训练完成!")
    
    # 保存最终模型
    final_checkpoint_path = os.path.join(args.save_dir, "final_model.pth")
    save_checkpoint({
        'epoch': config['num_epochs'],
        'gen': gen.state_dict(),
        'critic': critic.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_critic': opt_critic.state_dict(),
        'loss_g_history': loss_g_history,
        'loss_d_history': loss_d_history,
        'config': config
    }, final_checkpoint_path, logger)
    
    # 保存损失历史
    np.save(os.path.join(args.save_dir, 'loss_history.npy'), {
        'generator': loss_g_history,
        'discriminator': loss_d_history
    })
    
    logger.info(f"所有结果已保存到 {args.save_dir} 目录")

if __name__ == "__main__":
    main()
