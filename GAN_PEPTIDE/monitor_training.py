#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
监控训练进度的脚本
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def monitor_training():
    """监控训练进度"""
    save_dir = "gram_negative_results"
    
    print("=== 训练监控 ===")
    print("监控训练进度，按 Ctrl+C 退出监控")
    
    try:
        while True:
            # 检查保存目录
            if os.path.exists(save_dir):
                # 列出所有检查点文件
                checkpoints = glob(f"{save_dir}/checkpoint_epoch_*.pth")
                generated_files = glob(f"{save_dir}/generated_epoch_*.npy")
                
                print(f"\n当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"已保存的检查点: {len(checkpoints)}")
                print(f"已生成的样本文件: {len(generated_files)}")
                
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=os.path.getctime)
                    print(f"最新检查点: {os.path.basename(latest_checkpoint)}")
                
                # 检查是否有损失历史文件
                loss_file = f"{save_dir}/loss_history.npy"
                if os.path.exists(loss_file):
                    print("训练已完成！")
                    break
            else:
                print("训练结果目录尚未创建...")
            
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def analyze_generated_samples(epoch=None):
    """分析生成的样本"""
    save_dir = "gram_negative_results"
    
    if epoch is None:
        # 找到最新的生成文件
        generated_files = glob(f"{save_dir}/generated_epoch_*.npy")
        if not generated_files:
            print("没有找到生成的样本文件")
            return
        
        latest_file = max(generated_files, key=os.path.getctime)
        epoch = int(latest_file.split('_')[-1].split('.')[0])
    else:
        latest_file = f"{save_dir}/generated_epoch_{epoch}.npy"
        if not os.path.exists(latest_file):
            print(f"文件不存在: {latest_file}")
            return
    
    print(f"\n=== 分析第 {epoch} 轮生成的样本 ===")
    
    # 加载生成的样本
    generated_indices = np.load(latest_file)
    print(f"生成样本形状: {generated_indices.shape}")
    
    # 转换为氨基酸序列
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    
    print(f"\n生成的样本序列 (前10个):")
    for i in range(min(10, len(generated_indices))):
        seq_indices = generated_indices[i]
        seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
        # 移除trailing padding
        seq_str = seq_str.rstrip(' ')
        print(f"样本 {i+1}: {seq_str} (长度: {len(seq_str)})")
    
    # 分析序列长度分布
    lengths = []
    for seq_indices in generated_indices:
        seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
        seq_str = seq_str.rstrip(' ')
        lengths.append(len(seq_str))
    
    print(f"\n长度统计:")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"最短长度: {min(lengths)}")
    print(f"最长长度: {max(lengths)}")
    print(f"标准差: {np.std(lengths):.2f}")
    
    # 分析氨基酸组成
    all_aa = ''.join([''.join([amino_acids[idx] for idx in seq]).rstrip(' ') 
                      for seq in generated_indices])
    
    from collections import Counter
    aa_counter = Counter(all_aa)
    
    print(f"\n氨基酸频率 (前10个):")
    for aa, count in aa_counter.most_common(10):
        print(f"  {aa}: {count} ({count/len(all_aa)*100:.2f}%)")

def save_sequences_to_fasta(epoch=None, output_file=None):
    """将生成的序列保存为FASTA格式"""
    save_dir = "gram_negative_results"
    
    if epoch is None:
        # 找到最新的生成文件
        generated_files = glob(f"{save_dir}/generated_epoch_*.npy")
        if not generated_files:
            print("没有找到生成的样本文件")
            return
        
        latest_file = max(generated_files, key=os.path.getctime)
        epoch = int(latest_file.split('_')[-1].split('.')[0])
    else:
        latest_file = f"{save_dir}/generated_epoch_{epoch}.npy"
        if not os.path.exists(latest_file):
            print(f"文件不存在: {latest_file}")
            return
    
    if output_file is None:
        output_file = f"{save_dir}/generated_epoch_{epoch}.fasta"
    
    # 加载生成的样本
    generated_indices = np.load(latest_file)
    
    # 转换为氨基酸序列并保存
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    
    with open(output_file, 'w') as f:
        for i, seq_indices in enumerate(generated_indices):
            seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
            seq_str = seq_str.rstrip(' ')  # 移除padding
            
            if len(seq_str) > 0:  # 只保存非空序列
                f.write(f">generated_peptide_{i+1}\n")
                f.write(f"{seq_str}\n")
    
    print(f"生成的序列已保存到: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            monitor_training()
        elif command == "analyze":
            epoch = int(sys.argv[2]) if len(sys.argv) > 2 else None
            analyze_generated_samples(epoch)
        elif command == "fasta":
            epoch = int(sys.argv[2]) if len(sys.argv) > 2 else None
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            save_sequences_to_fasta(epoch, output_file)
        else:
            print("用法:")
            print("  python monitor_training.py monitor     # 监控训练进度")
            print("  python monitor_training.py analyze [epoch]  # 分析生成样本")
            print("  python monitor_training.py fasta [epoch] [output_file]  # 保存为FASTA")
    else:
        print("用法:")
        print("  python monitor_training.py monitor     # 监控训练进度")
        print("  python monitor_training.py analyze [epoch]  # 分析生成样本")
        print("  python monitor_training.py fasta [epoch] [output_file]  # 保存为FASTA")
