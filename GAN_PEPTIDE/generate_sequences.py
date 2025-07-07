#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用训练好的模型生成肽段序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os

# 复制必要的模型类定义
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

def return_index(one_hot_coding):
    """将one-hot编码或softmax输出转换为索引"""
    if hasattr(one_hot_coding, 'numpy'):
        one_hot_coding = one_hot_coding.numpy()
    
    if one_hot_coding.dtype == np.float32 or one_hot_coding.dtype == np.float64:
        indices = np.argmax(one_hot_coding, axis=-1)
        return indices.reshape(-1, 30)
    else:
        index = np.argwhere(one_hot_coding == 1)
        if len(index) == 0:
            indices = np.argmax(one_hot_coding, axis=-1)
            return indices.reshape(-1, 30)
        return index[:, -1].reshape(-1, 30)

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        z_dim = config.get('z_dim', 90)
        features_gen = config.get('features_gen', 16)
        num_classes = config.get('num_classes', 1)
    else:
        # 使用默认配置
        z_dim = 90
        features_gen = 16
        num_classes = 1
    
    # 创建生成器
    generator = Generator(z_dim, 1, features_gen, num_classes).to(device)
    
    # 加载权重
    generator.load_state_dict(checkpoint['gen'])
    generator.eval()
    
    print(f"模型加载成功!")
    print(f"配置: z_dim={z_dim}, features_gen={features_gen}, num_classes={num_classes}")
    
    return generator, z_dim

def generate_peptide_sequences(generator, z_dim, num_sequences, device, batch_size=64):
    """生成肽段序列"""
    print(f"生成 {num_sequences} 条肽段序列...")
    
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    all_sequences = []
    
    # 分批生成
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # 计算当前批次大小
            current_batch_size = min(batch_size, num_sequences - batch_idx * batch_size)
            
            # 生成随机噪声
            noise = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
            
            # 生成标签（全部使用标签0：革兰氏阴性菌抗菌肽）
            labels = torch.zeros(current_batch_size, 1, dtype=torch.long).to(device)
            
            # 生成样本
            fake_samples = generator(noise, labels)
            
            # 转换为概率分布
            fake_samples_normalized = (fake_samples + 1) / 2
            fake_samples_prob = torch.softmax(fake_samples_normalized.view(-1, 30, 21), dim=-1)
            
            # 转换为氨基酸索引
            fake_indices = return_index(fake_samples_prob.cpu())
            
            # 转换为氨基酸序列
            for seq_indices in fake_indices:
                seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
                seq_str = seq_str.rstrip(' ')  # 移除padding
                if len(seq_str) > 0:  # 只保存非空序列
                    all_sequences.append(seq_str)
            
            if batch_idx % 5 == 0:
                print(f"  已生成 {len(all_sequences)} 条序列...")
    
    print(f"生成完成! 总共 {len(all_sequences)} 条有效序列")
    return all_sequences

def save_sequences(sequences, output_file, format='fasta'):
    """保存序列到文件"""
    print(f"保存序列到: {output_file}")
    
    if format.lower() == 'fasta':
        with open(output_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">generated_peptide_{i+1}\n")
                f.write(f"{seq}\n")
    elif format.lower() == 'txt':
        with open(output_file, 'w') as f:
            for seq in sequences:
                f.write(f"{seq}\n")
    
    print(f"保存完成! {len(sequences)} 条序列已保存")

def main():
    parser = argparse.ArgumentParser(description='生成肽段序列')
    parser.add_argument('--model', type=str, default='gram_negative_results_server/final_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--num_sequences', type=int, default=300, 
                       help='生成序列数量')
    parser.add_argument('--output', type=str, default='generated_300_peptides.fasta', 
                       help='输出文件名')
    parser.add_argument('--format', type=str, default='fasta', choices=['fasta', 'txt'],
                       help='输出格式')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='批次大小')
    parser.add_argument('--gpu_id', type=int, default=0, 
                       help='GPU ID')
    
    args = parser.parse_args()
    
    # 设备配置
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_id}"
        print(f"使用GPU: {device}")
    else:
        device = "cpu"
        print("使用CPU")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("可用的模型文件:")
        model_dir = os.path.dirname(args.model)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    print(f"  {model_dir}/{f}")
        return
    
    try:
        # 加载模型
        generator, z_dim = load_model(args.model, device)
        
        # 生成序列
        sequences = generate_peptide_sequences(
            generator, z_dim, args.num_sequences, device, args.batch_size
        )
        
        # 确保有足够的序列
        if len(sequences) < args.num_sequences:
            print(f"警告: 只生成了 {len(sequences)} 条有效序列，少于请求的 {args.num_sequences} 条")
        else:
            # 截取到请求的数量
            sequences = sequences[:args.num_sequences]
        
        # 保存序列
        save_sequences(sequences, args.output, args.format)
        
        # 显示一些样本
        print(f"\n样本序列 (前10条):")
        for i, seq in enumerate(sequences[:10]):
            print(f"  {i+1:2d}: {seq}")
        
        # 统计信息
        lengths = [len(seq) for seq in sequences]
        print(f"\n序列统计:")
        print(f"  数量: {len(sequences)}")
        print(f"  平均长度: {np.mean(lengths):.2f}")
        print(f"  长度范围: {min(lengths)}-{max(lengths)}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
