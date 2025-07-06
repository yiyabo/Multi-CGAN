#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证预处理后的数据质量
"""

import numpy as np
import torch
import torch.nn.functional as F

def validate_data():
    """验证数据质量"""
    print("=== 数据质量验证 ===\n")
    
    # 加载数据
    train_seq = np.load('data/gram_negative_train_pos.npy')
    train_labels = np.load('data/gram_negative_train_labels.npy')
    val_seq = np.load('data/gram_negative_val_pos.npy')
    val_labels = np.load('data/gram_negative_val_labels.npy')
    
    print(f"训练集序列形状: {train_seq.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"验证集序列形状: {val_seq.shape}")
    print(f"验证集标签形状: {val_labels.shape}")
    
    # 检查数据范围
    print(f"\n数据范围检查:")
    print(f"训练序列最小值: {train_seq.min()}, 最大值: {train_seq.max()}")
    print(f"验证序列最小值: {val_seq.min()}, 最大值: {val_seq.max()}")
    print(f"训练标签最小值: {train_labels.min()}, 最大值: {train_labels.max()}")
    print(f"验证标签最小值: {val_labels.min()}, 最大值: {val_labels.max()}")
    
    # 检查标签分布
    print(f"\n标签分布:")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    print(f"训练集标签分布: {dict(zip(unique_train, counts_train))}")
    print(f"验证集标签分布: {dict(zip(unique_val, counts_val))}")
    
    # 检查序列中的氨基酸分布
    print(f"\n氨基酸分布检查:")
    unique_aa_train, counts_aa_train = np.unique(train_seq, return_counts=True)
    print(f"训练集氨基酸种类: {len(unique_aa_train)}")
    print(f"氨基酸索引范围: {unique_aa_train.min()} - {unique_aa_train.max()}")
    
    # 检查padding情况
    padding_idx = 20  # 空格的索引
    train_padding = np.sum(train_seq == padding_idx, axis=1)
    val_padding = np.sum(val_seq == padding_idx, axis=1)
    
    print(f"\nPadding统计:")
    print(f"训练集平均padding长度: {train_padding.mean():.2f}")
    print(f"验证集平均padding长度: {val_padding.mean():.2f}")
    print(f"训练集最大padding长度: {train_padding.max()}")
    print(f"验证集最大padding长度: {val_padding.max()}")
    
    # 测试one-hot转换
    print(f"\nOne-hot转换测试:")
    sample_seq = torch.LongTensor(train_seq[:5])
    one_hot_seq = F.one_hot(sample_seq, 21).reshape(-1, 30, 21)
    print(f"原始序列形状: {sample_seq.shape}")
    print(f"One-hot形状: {one_hot_seq.shape}")
    
    # 显示几个样本序列
    print(f"\n样本序列 (前5个):")
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    
    for i in range(min(5, len(train_seq))):
        seq_indices = train_seq[i]
        seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
        # 移除trailing padding
        seq_str = seq_str.rstrip(' ')
        print(f"序列 {i+1}: {seq_str} (长度: {len(seq_str)})")
    
    print(f"\n=== 验证完成 ===")
    return True

if __name__ == "__main__":
    validate_data()
