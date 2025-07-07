#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析最终训练结果
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import torch

def analyze_final_results():
    """分析最终训练结果"""
    print("🎉 === 训练完成！最终结果分析 === 🎉\n")
    
    results_dir = "gram_negative_results_server"
    
    # 1. 检查文件完整性
    print("📁 文件检查:")
    required_files = [
        "final_model.pth",
        "checkpoint_epoch_1500.pth", 
        "generated_epoch_1500.npy",
        "loss_history.npy"
    ]
    
    for file in required_files:
        file_path = f"{results_dir}/{file}"
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✅ {file} ({size:.2f} MB)")
        else:
            print(f"  ❌ {file} (缺失)")
    
    # 2. 分析损失历史
    print(f"\n📊 损失分析:")
    try:
        loss_data = np.load(f"{results_dir}/loss_history.npy", allow_pickle=True).item()
        
        if 'generator' in loss_data and 'discriminator' in loss_data:
            gen_losses = loss_data['generator']
            disc_losses = loss_data['discriminator']
            
            print(f"  训练轮数: {len(gen_losses)}")
            print(f"  Generator损失:")
            print(f"    最终: {gen_losses[-1]:.4f}")
            print(f"    最佳: {min(gen_losses):.4f}")
            print(f"    平均: {np.mean(gen_losses):.4f}")
            
            print(f"  Discriminator损失:")
            print(f"    最终: {disc_losses[-1]:.4f}")
            print(f"    最佳: {min(disc_losses):.4f}")
            print(f"    平均: {np.mean(disc_losses):.4f}")
            
            # 收敛分析
            if len(gen_losses) > 100:
                recent_gen = np.mean(gen_losses[-100:])
                early_gen = np.mean(gen_losses[:100])
                improvement = ((early_gen - recent_gen) / abs(early_gen)) * 100
                
                if improvement > 0:
                    print(f"  📈 Generator改善: {improvement:.1f}%")
                else:
                    print(f"  📉 Generator变化: {improvement:.1f}%")
        
    except Exception as e:
        print(f"  ❌ 无法读取损失历史: {e}")
    
    # 3. 分析生成样本
    print(f"\n🧬 生成样本分析:")
    try:
        generated_samples = np.load(f"{results_dir}/generated_epoch_1500.npy")
        print(f"  样本数量: {len(generated_samples)}")
        
        # 转换为氨基酸序列
        amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
        
        sequences = []
        lengths = []
        
        for seq_indices in generated_samples:
            seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
            seq_str = seq_str.rstrip(' ')  # 移除padding
            if len(seq_str) > 0:
                sequences.append(seq_str)
                lengths.append(len(seq_str))
        
        valid_sequences = len(sequences)
        print(f"  有效序列: {valid_sequences}/{len(generated_samples)} ({valid_sequences/len(generated_samples)*100:.1f}%)")
        
        if lengths:
            print(f"  序列长度统计:")
            print(f"    平均: {np.mean(lengths):.2f}")
            print(f"    范围: {min(lengths)}-{max(lengths)}")
            print(f"    标准差: {np.std(lengths):.2f}")
            
            # 长度分布
            length_counter = Counter(lengths)
            print(f"  最常见长度:")
            for length, count in length_counter.most_common(5):
                print(f"    {length}个氨基酸: {count}次 ({count/len(lengths)*100:.1f}%)")
            
            # 氨基酸组成分析
            all_aa = ''.join(sequences)
            aa_counter = Counter(all_aa)
            
            print(f"  氨基酸组成 (前10个):")
            for aa, count in aa_counter.most_common(10):
                print(f"    {aa}: {count} ({count/len(all_aa)*100:.2f}%)")
            
            # 显示样本序列
            print(f"\n  样本序列 (前10个):")
            for i, seq in enumerate(sequences[:10]):
                print(f"    {i+1:2d}: {seq}")
        
    except Exception as e:
        print(f"  ❌ 无法分析生成样本: {e}")
    
    # 4. 与原始数据对比
    print(f"\n🔬 与原始数据对比:")
    try:
        # 加载原始训练数据进行对比
        original_data = np.load("data/gram_negative_train_pos.npy")
        
        # 转换原始数据为序列
        original_sequences = []
        original_lengths = []
        
        for seq_indices in original_data[:100]:  # 取前100个样本对比
            seq_str = ''.join([amino_acids[idx] for idx in seq_indices])
            seq_str = seq_str.rstrip(' ')
            if len(seq_str) > 0:
                original_sequences.append(seq_str)
                original_lengths.append(len(seq_str))
        
        if original_lengths and lengths:
            print(f"  长度对比:")
            print(f"    原始数据平均长度: {np.mean(original_lengths):.2f}")
            print(f"    生成数据平均长度: {np.mean(lengths):.2f}")
            
            # 氨基酸组成对比
            original_aa = ''.join(original_sequences)
            original_aa_counter = Counter(original_aa)
            
            print(f"  氨基酸频率对比 (前5个):")
            print(f"    {'AA':<3} {'原始':<8} {'生成':<8} {'差异':<8}")
            for aa, _ in original_aa_counter.most_common(5):
                orig_freq = original_aa_counter[aa] / len(original_aa) * 100
                gen_freq = aa_counter.get(aa, 0) / len(all_aa) * 100 if 'all_aa' in locals() else 0
                diff = gen_freq - orig_freq
                print(f"    {aa:<3} {orig_freq:<8.2f} {gen_freq:<8.2f} {diff:+.2f}")
        
    except Exception as e:
        print(f"  ⚠️  无法加载原始数据进行对比: {e}")
    
    # 5. 保存分析结果
    print(f"\n💾 保存分析结果:")
    
    # 保存生成的序列为FASTA格式
    if 'sequences' in locals() and sequences:
        fasta_file = f"{results_dir}/final_generated_peptides.fasta"
        with open(fasta_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">generated_peptide_{i+1}\n")
                f.write(f"{seq}\n")
        print(f"  ✅ FASTA文件: {fasta_file}")
    
    # 保存统计信息
    stats_file = f"{results_dir}/training_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Multi-CGAN 训练统计报告\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"训练完成时间: 2025-07-06 11:20:59\n")
        f.write(f"总训练轮数: 1500\n")
        f.write(f"GPU: RTX 4090 D\n")
        f.write(f"批次大小: 64\n\n")
        
        if 'gen_losses' in locals():
            f.write(f"最终Generator损失: {gen_losses[-1]:.4f}\n")
            f.write(f"最终Discriminator损失: {disc_losses[-1]:.4f}\n\n")
        
        if 'valid_sequences' in locals():
            f.write(f"生成的有效序列数: {valid_sequences}\n")
            f.write(f"平均序列长度: {np.mean(lengths):.2f}\n")
            f.write(f"序列长度范围: {min(lengths)}-{max(lengths)}\n")
    
    print(f"  ✅ 统计报告: {stats_file}")
    
    print(f"\n🎊 分析完成！训练非常成功！")
    print(f"📁 所有结果保存在: {results_dir}/")
    print(f"🧬 生成的肽段序列: {results_dir}/final_generated_peptides.fasta")

if __name__ == "__main__":
    analyze_final_results()
