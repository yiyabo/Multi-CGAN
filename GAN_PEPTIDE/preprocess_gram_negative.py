#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gram-.fasta数据预处理脚本
将FASTA格式转换为CWGAN训练所需的.npy格式
"""

import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn.functional as F

def create_amino_acid_dict():
    """创建氨基酸字典，与原项目保持一致"""
    words = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    word_dict = {}
    for word_index, word in enumerate(words):
        word_dict[word] = word_index
    return word_dict

def parse_fasta(fasta_file):
    """解析FASTA文件"""
    sequences = []
    sequence_ids = []
    
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    current_seq = ""
    current_id = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_seq and current_id:
                sequences.append(current_seq)
                sequence_ids.append(current_id)
            current_id = line[1:]  # 去掉>符号
            current_seq = ""
        elif line and not line.startswith('>'):
            current_seq += line
    
    # 添加最后一个序列
    if current_seq and current_id:
        sequences.append(current_seq)
        sequence_ids.append(current_id)
    
    return sequences, sequence_ids

def clean_sequences(sequences, word_dict):
    """清理序列，处理非标准字符"""
    cleaned_sequences = []
    removed_count = 0
    
    standard_aa = set(word_dict.keys()) - {' '}  # 去掉padding字符
    
    for seq in sequences:
        # 转换为大写
        seq = seq.upper()
        
        # 检查是否包含非标准字符
        seq_aa = set(seq)
        if seq_aa.issubset(standard_aa):
            cleaned_sequences.append(seq)
        else:
            # 尝试替换常见的非标准字符
            # 小写l通常是大写L的错误
            seq_fixed = seq.replace('l', 'L')
            seq_aa_fixed = set(seq_fixed)
            
            if seq_aa_fixed.issubset(standard_aa):
                cleaned_sequences.append(seq_fixed)
                print(f"修复序列: {seq} -> {seq_fixed}")
            else:
                removed_count += 1
                non_standard = seq_aa_fixed - standard_aa
                print(f"移除序列 (包含非标准字符 {non_standard}): {seq}")
    
    print(f"清理完成: 保留 {len(cleaned_sequences)} 个序列, 移除 {removed_count} 个序列")
    return cleaned_sequences

def process_sequences(sequences, word_dict, max_length=30, strategy='truncate'):
    """处理序列长度和编码"""
    processed_sequences = []
    truncated_count = 0
    
    for seq in sequences:
        if len(seq) > max_length:
            if strategy == 'truncate':
                # 截断到max_length
                seq = seq[:max_length]
                truncated_count += 1
            elif strategy == 'skip':
                # 跳过过长的序列
                continue
        
        # 转换为数字编码
        encoded_seq = []
        for char in seq:
            if char in word_dict:
                encoded_seq.append(word_dict[char])
            else:
                print(f"警告: 未知字符 '{char}' 在序列中")
                continue
        
        # 填充到固定长度
        while len(encoded_seq) < max_length:
            encoded_seq.append(word_dict[' '])  # 使用空格作为padding
        
        processed_sequences.append(encoded_seq)
    
    if strategy == 'truncate' and truncated_count > 0:
        print(f"截断了 {truncated_count} 个序列到 {max_length} 个氨基酸")
    
    return np.array(processed_sequences, dtype=int)

def create_labels(num_sequences, label_type='single'):
    """创建标签"""
    if label_type == 'single':
        # 单一标签：革兰氏阴性菌抗菌肽
        labels = np.zeros((num_sequences, 1), dtype=int)
        labels.fill(0)  # 标签0表示革兰氏阴性菌抗菌肽
    elif label_type == 'length_based':
        # 基于长度的标签
        # 这里可以根据需要实现
        pass
    
    return labels

def augment_data(sequences, augment_factor=2):
    """数据增强"""
    augmented_sequences = []
    original_sequences = sequences.copy()
    
    # 原始数据
    augmented_sequences.extend(original_sequences)
    
    # 生成增强数据
    for _ in range(augment_factor - 1):
        for seq in original_sequences:
            # 随机突变：随机替换1-2个氨基酸
            augmented_seq = seq.copy()
            
            # 找到非padding位置
            non_padding_indices = [i for i, aa in enumerate(seq) if aa != 20]  # 20是padding
            
            if len(non_padding_indices) > 2:
                # 随机选择1-2个位置进行突变
                num_mutations = np.random.choice([1, 2])
                mutation_indices = np.random.choice(non_padding_indices, 
                                                  size=min(num_mutations, len(non_padding_indices)), 
                                                  replace=False)
                
                for idx in mutation_indices:
                    # 随机选择一个新的氨基酸（0-19，排除padding）
                    new_aa = np.random.randint(0, 20)
                    augmented_seq[idx] = new_aa
            
            augmented_sequences.append(augmented_seq)
    
    return np.array(augmented_sequences)

def save_processed_data(sequences, labels, save_prefix='gram_negative'):
    """保存处理后的数据"""
    # 保存序列数据
    seq_path = f'data/{save_prefix}_pos.npy'
    np.save(seq_path, sequences)
    print(f"序列数据已保存到: {seq_path}")
    
    # 保存标签数据
    label_path = f'data/{save_prefix}_labels.npy'
    np.save(label_path, labels)
    print(f"标签数据已保存到: {label_path}")
    
    # 验证保存的数据
    loaded_seq = np.load(seq_path)
    loaded_labels = np.load(label_path)
    
    print(f"验证: 序列形状 {loaded_seq.shape}, 标签形状 {loaded_labels.shape}")
    
    return seq_path, label_path

def create_validation_split(sequences, labels, val_ratio=0.2):
    """创建训练/验证集分割"""
    total_samples = len(sequences)
    val_size = int(total_samples * val_ratio)
    
    # 随机打乱
    indices = np.random.permutation(total_samples)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_sequences = sequences[train_indices]
    train_labels = labels[train_indices]
    val_sequences = sequences[val_indices]
    val_labels = labels[val_indices]
    
    return (train_sequences, train_labels), (val_sequences, val_labels)

def main():
    """主处理流程"""
    print("=== Gram-.fasta 数据预处理 ===\n")
    
    # 1. 创建氨基酸字典
    word_dict = create_amino_acid_dict()
    print(f"氨基酸字典: {word_dict}")
    
    # 2. 解析FASTA文件
    print("\n正在解析FASTA文件...")
    sequences, sequence_ids = parse_fasta('Gram-.fasta')
    print(f"解析完成: {len(sequences)} 个序列")
    
    # 3. 清理序列
    print("\n正在清理序列...")
    cleaned_sequences = clean_sequences(sequences, word_dict)
    
    # 4. 处理序列长度和编码
    print("\n正在处理序列...")
    processed_sequences = process_sequences(cleaned_sequences, word_dict, 
                                          max_length=30, strategy='truncate')
    print(f"处理完成: {processed_sequences.shape}")
    
    # 5. 创建标签
    print("\n正在创建标签...")
    labels = create_labels(len(processed_sequences), label_type='single')
    print(f"标签创建完成: {labels.shape}")
    
    # 6. 数据增强（可选）
    print("\n是否进行数据增强? (y/n): ", end="")
    # 为了自动化，这里默认进行数据增强
    do_augment = True
    
    if do_augment:
        print("正在进行数据增强...")
        augmented_sequences = augment_data(processed_sequences, augment_factor=3)
        augmented_labels = np.tile(labels, (3, 1))
        print(f"数据增强完成: {augmented_sequences.shape}")
        
        final_sequences = augmented_sequences
        final_labels = augmented_labels
    else:
        final_sequences = processed_sequences
        final_labels = labels
    
    # 7. 创建训练/验证集分割
    print("\n正在创建训练/验证集分割...")
    (train_seq, train_labels), (val_seq, val_labels) = create_validation_split(
        final_sequences, final_labels, val_ratio=0.2)
    
    print(f"训练集: {train_seq.shape}, 验证集: {val_seq.shape}")
    
    # 8. 保存数据
    print("\n正在保存数据...")
    train_seq_path, train_label_path = save_processed_data(
        train_seq, train_labels, 'gram_negative_train')
    val_seq_path, val_label_path = save_processed_data(
        val_seq, val_labels, 'gram_negative_val')
    
    print(f"\n=== 预处理完成 ===")
    print(f"训练数据: {train_seq_path}, {train_label_path}")
    print(f"验证数据: {val_seq_path}, {val_label_path}")
    
    # 9. 生成统计报告
    print(f"\n=== 最终统计 ===")
    print(f"原始序列数: {len(sequences)}")
    print(f"清理后序列数: {len(cleaned_sequences)}")
    print(f"最终训练样本数: {len(train_seq)}")
    print(f"最终验证样本数: {len(val_seq)}")

if __name__ == "__main__":
    main()
