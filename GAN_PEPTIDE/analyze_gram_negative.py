#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析Gram-.fasta数据集的脚本
分析序列特征，为后续训练做准备
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

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

def analyze_sequences(sequences):
    """分析序列特征"""
    print("=== Gram-.fasta 数据集分析报告 ===\n")
    
    # 基本统计
    total_sequences = len(sequences)
    print(f"总序列数量: {total_sequences}")
    
    # 序列长度分析
    lengths = [len(seq) for seq in sequences]
    print(f"\n序列长度统计:")
    print(f"  最短长度: {min(lengths)}")
    print(f"  最长长度: {max(lengths)}")
    print(f"  平均长度: {np.mean(lengths):.2f}")
    print(f"  中位数长度: {np.median(lengths):.2f}")
    print(f"  标准差: {np.std(lengths):.2f}")
    
    # 长度分布
    length_counter = Counter(lengths)
    print(f"\n长度分布 (前10个最常见的长度):")
    for length, count in length_counter.most_common(10):
        print(f"  长度 {length}: {count} 个序列 ({count/total_sequences*100:.1f}%)")
    
    # 长度范围分析
    length_ranges = {
        "≤10": sum(1 for l in lengths if l <= 10),
        "11-20": sum(1 for l in lengths if 11 <= l <= 20),
        "21-30": sum(1 for l in lengths if 21 <= l <= 30),
        "31-40": sum(1 for l in lengths if 31 <= l <= 40),
        "41-50": sum(1 for l in lengths if 41 <= l <= 50),
        ">50": sum(1 for l in lengths if l > 50)
    }
    
    print(f"\n长度范围分布:")
    for range_name, count in length_ranges.items():
        print(f"  {range_name}: {count} 个序列 ({count/total_sequences*100:.1f}%)")
    
    # 氨基酸组成分析
    all_amino_acids = ''.join(sequences)
    aa_counter = Counter(all_amino_acids)
    
    print(f"\n氨基酸组成分析:")
    print(f"  总氨基酸数量: {len(all_amino_acids)}")
    print(f"  不同氨基酸种类: {len(aa_counter)}")
    
    # 标准20种氨基酸
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    found_aa = set(aa_counter.keys())
    
    print(f"\n氨基酸种类检查:")
    print(f"  标准氨基酸: {sorted(found_aa & standard_aa)}")
    non_standard = found_aa - standard_aa
    if non_standard:
        print(f"  非标准字符: {sorted(non_standard)}")
        for char in sorted(non_standard):
            print(f"    '{char}': {aa_counter[char]} 次")
    else:
        print(f"  ✓ 所有字符都是标准氨基酸")
    
    # 氨基酸频率
    print(f"\n氨基酸频率 (前10个最常见):")
    for aa, count in aa_counter.most_common(10):
        if aa in standard_aa:
            print(f"  {aa}: {count} ({count/len(all_amino_acids)*100:.2f}%)")
    
    return {
        'total_sequences': total_sequences,
        'lengths': lengths,
        'aa_counter': aa_counter,
        'non_standard_aa': non_standard,
        'length_ranges': length_ranges
    }

def check_compatibility_with_original(sequences):
    """检查与原项目的兼容性"""
    print(f"\n=== 与原项目兼容性分析 ===")
    
    # 原项目使用的氨基酸字典
    original_aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
    original_aa_set = set(original_aa[:-1])  # 去掉空格
    
    # 检查序列中的氨基酸
    all_amino_acids = ''.join(sequences)
    found_aa = set(all_amino_acids)
    
    compatible_aa = found_aa & original_aa_set
    incompatible_aa = found_aa - original_aa_set
    
    print(f"兼容的氨基酸: {sorted(compatible_aa)}")
    if incompatible_aa:
        print(f"不兼容的字符: {sorted(incompatible_aa)}")
        print(f"需要处理的字符数量: {len(incompatible_aa)}")
    else:
        print(f"✓ 所有氨基酸都与原项目兼容")
    
    # 长度兼容性 (原项目使用30作为固定长度)
    lengths = [len(seq) for seq in sequences]
    over_30 = sum(1 for l in lengths if l > 30)
    under_30 = sum(1 for l in lengths if l <= 30)
    
    print(f"\n长度兼容性 (原项目固定长度30):")
    print(f"  ≤30的序列: {under_30} ({under_30/len(sequences)*100:.1f}%)")
    print(f"  >30的序列: {over_30} ({over_30/len(sequences)*100:.1f}%)")
    
    if over_30 > 0:
        print(f"  需要截断的序列: {over_30}")
        print(f"  建议: 截断到30个氨基酸或调整模型输入维度")
    
    return {
        'compatible_aa': compatible_aa,
        'incompatible_aa': incompatible_aa,
        'sequences_over_30': over_30,
        'sequences_under_30': under_30
    }

def suggest_preprocessing_strategy(analysis_result, compatibility_result):
    """建议预处理策略"""
    print(f"\n=== 预处理策略建议 ===")
    
    total_sequences = analysis_result['total_sequences']
    over_30 = compatibility_result['sequences_over_30']
    incompatible_aa = compatibility_result['incompatible_aa']
    
    print(f"1. 数据清理:")
    if incompatible_aa:
        print(f"   - 需要处理 {len(incompatible_aa)} 种非标准字符")
        print(f"   - 建议: 删除包含非标准字符的序列或进行字符替换")
    else:
        print(f"   ✓ 无需字符清理")
    
    print(f"\n2. 长度处理:")
    if over_30 > 0:
        print(f"   - {over_30} 个序列超过30个氨基酸")
        print(f"   - 选项1: 截断到30个氨基酸 (可能丢失信息)")
        print(f"   - 选项2: 调整模型输入维度到更大值")
        print(f"   - 建议: 先尝试截断，如果效果不好再调整模型")
    else:
        print(f"   ✓ 所有序列都≤30个氨基酸")
    
    print(f"\n3. 数据增强:")
    print(f"   - 当前数据量: {total_sequences} (相对较小)")
    print(f"   - 建议: 考虑数据增强技术")
    print(f"     * 序列的随机突变")
    print(f"     * 滑动窗口截取")
    print(f"     * 添加轻微噪声")
    
    print(f"\n4. 训练策略:")
    print(f"   - 建议减小batch size (从32到16或8)")
    print(f"   - 增加训练轮数")
    print(f"   - 使用较小的学习率")
    print(f"   - 考虑使用预训练模型")

if __name__ == "__main__":
    # 分析数据集
    fasta_file = "Gram-.fasta"
    
    print("正在解析FASTA文件...")
    sequences, sequence_ids = parse_fasta(fasta_file)
    
    print("正在分析序列特征...")
    analysis_result = analyze_sequences(sequences)
    
    print("正在检查兼容性...")
    compatibility_result = check_compatibility_with_original(sequences)
    
    print("正在生成建议...")
    suggest_preprocessing_strategy(analysis_result, compatibility_result)
    
    print(f"\n=== 分析完成 ===")
    print(f"详细结果已保存到变量中，可用于后续处理")
