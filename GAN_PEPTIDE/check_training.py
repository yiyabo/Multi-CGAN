#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练状态检查脚本
检查训练进度、生成样本质量等
"""

import os
import sys
import numpy as np
import torch
import json
from glob import glob
from datetime import datetime
import argparse

def check_training_status():
    """检查训练状态"""
    print("=== 训练状态检查 ===\n")
    
    # 检查进程
    if os.path.exists("training.pid"):
        with open("training.pid", "r") as f:
            pid = f.read().strip()
        
        # 检查进程是否还在运行
        try:
            os.kill(int(pid), 0)
            print(f"✓ 训练进程正在运行 (PID: {pid})")
        except OSError:
            print(f"✗ 训练进程已停止 (PID: {pid})")
    else:
        print("✗ 未找到训练进程信息")
    
    # 检查结果目录
    results_dir = "gram_negative_results_server"
    if os.path.exists(results_dir):
        print(f"✓ 结果目录存在: {results_dir}")
        
        # 检查检查点文件
        checkpoints = glob(f"{results_dir}/checkpoint_epoch_*.pth")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            print(f"✓ 最新检查点: Epoch {epoch}")
            
            # 加载检查点信息
            try:
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    total_epochs = config.get('num_epochs', 'Unknown')
                    progress = (epoch / total_epochs * 100) if total_epochs != 'Unknown' else 'Unknown'
                    print(f"✓ 训练进度: {epoch}/{total_epochs} ({progress:.1f}%)")
                
                if 'loss_g' in checkpoint and 'loss_d' in checkpoint:
                    print(f"✓ 最新损失 - Generator: {checkpoint['loss_g']:.4f}, Discriminator: {checkpoint['loss_d']:.4f}")
            except Exception as e:
                print(f"✗ 无法读取检查点信息: {e}")
        else:
            print("✗ 未找到检查点文件")
        
        # 检查生成样本
        generated_files = glob(f"{results_dir}/generated_epoch_*.npy")
        if generated_files:
            latest_generated = max(generated_files, key=os.path.getctime)
            epoch = int(latest_generated.split('_')[-1].split('.')[0])
            print(f"✓ 最新生成样本: Epoch {epoch}")
        else:
            print("✗ 未找到生成样本文件")
    else:
        print(f"✗ 结果目录不存在: {results_dir}")
    
    # 检查日志
    log_files = glob("logs/training_*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"✓ 最新日志: {latest_log}")
        
        # 显示最后几行日志
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("最新日志内容:")
                    for line in lines[-5:]:
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"✗ 无法读取日志: {e}")
    else:
        print("✗ 未找到日志文件")

def analyze_loss_history():
    """分析损失历史"""
    print("\n=== 损失历史分析 ===\n")
    
    results_dir = "gram_negative_results_server"
    loss_file = f"{results_dir}/loss_history.npy"
    
    if os.path.exists(loss_file):
        try:
            loss_data = np.load(loss_file, allow_pickle=True).item()
            
            if 'generator' in loss_data and 'discriminator' in loss_data:
                gen_losses = loss_data['generator']
                disc_losses = loss_data['discriminator']
                
                print(f"训练轮数: {len(gen_losses)}")
                print(f"Generator损失:")
                print(f"  最新: {gen_losses[-1]:.4f}")
                print(f"  平均: {np.mean(gen_losses):.4f}")
                print(f"  最小: {np.min(gen_losses):.4f}")
                
                print(f"Discriminator损失:")
                print(f"  最新: {disc_losses[-1]:.4f}")
                print(f"  平均: {np.mean(disc_losses):.4f}")
                print(f"  最小: {np.min(disc_losses):.4f}")
                
                # 简单的收敛分析
                if len(gen_losses) > 10:
                    recent_gen = np.mean(gen_losses[-10:])
                    early_gen = np.mean(gen_losses[:10])
                    if recent_gen < early_gen:
                        print("✓ Generator损失呈下降趋势")
                    else:
                        print("⚠ Generator损失可能未收敛")
            else:
                print("✗ 损失数据格式不正确")
        except Exception as e:
            print(f"✗ 无法读取损失历史: {e}")
    else:
        print("✗ 损失历史文件不存在")

def analyze_generated_samples(epoch=None):
    """分析生成样本"""
    print("\n=== 生成样本分析 ===\n")
    
    results_dir = "gram_negative_results_server"
    
    if epoch is None:
        # 找到最新的生成文件
        generated_files = glob(f"{results_dir}/generated_epoch_*.npy")
        if not generated_files:
            print("✗ 未找到生成样本文件")
            return
        
        latest_file = max(generated_files, key=os.path.getctime)
        epoch = int(latest_file.split('_')[-1].split('.')[0])
    else:
        latest_file = f"{results_dir}/generated_epoch_{epoch}.npy"
        if not os.path.exists(latest_file):
            print(f"✗ 文件不存在: {latest_file}")
            return
    
    print(f"分析第 {epoch} 轮生成的样本")
    
    try:
        generated_indices = np.load(latest_file)
        print(f"样本数量: {len(generated_indices)}")
        
        # 转换为氨基酸序列
        amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
        
        # 分析序列长度
        lengths = []
        valid_sequences = 0
        
        for seq_indices in generated_indices:
            seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
            seq_str = seq_str.rstrip(' ')  # 移除padding
            if len(seq_str) > 0:
                lengths.append(len(seq_str))
                valid_sequences += 1
        
        if lengths:
            print(f"有效序列数: {valid_sequences}/{len(generated_indices)} ({valid_sequences/len(generated_indices)*100:.1f}%)")
            print(f"序列长度统计:")
            print(f"  平均: {np.mean(lengths):.2f}")
            print(f"  最短: {min(lengths)}")
            print(f"  最长: {max(lengths)}")
            print(f"  标准差: {np.std(lengths):.2f}")
            
            # 显示几个样本
            print(f"\n样本序列 (前5个):")
            count = 0
            for i, seq_indices in enumerate(generated_indices):
                if count >= 5:
                    break
                seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
                seq_str = seq_str.rstrip(' ')
                if len(seq_str) > 0:
                    print(f"  {count+1}: {seq_str}")
                    count += 1
        else:
            print("✗ 未找到有效序列")
            
    except Exception as e:
        print(f"✗ 无法分析生成样本: {e}")

def main():
    parser = argparse.ArgumentParser(description='检查训练状态和分析结果')
    parser.add_argument('--status', action='store_true', help='检查训练状态')
    parser.add_argument('--loss', action='store_true', help='分析损失历史')
    parser.add_argument('--samples', type=int, nargs='?', const=-1, help='分析生成样本 (可指定epoch)')
    parser.add_argument('--all', action='store_true', help='执行所有检查')
    
    args = parser.parse_args()
    
    if args.all or (not args.status and not args.loss and args.samples is None):
        # 默认执行所有检查
        check_training_status()
        analyze_loss_history()
        analyze_generated_samples()
    else:
        if args.status:
            check_training_status()
        if args.loss:
            analyze_loss_history()
        if args.samples is not None:
            epoch = args.samples if args.samples > 0 else None
            analyze_generated_samples(epoch)

if __name__ == "__main__":
    main()
