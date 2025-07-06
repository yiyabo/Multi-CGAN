#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速检查训练进度
"""

import os
import time
from glob import glob

def quick_check():
    """快速检查训练状态"""
    print("=== 快速训练状态检查 ===\n")
    
    # 检查进程
    if os.path.exists("training.pid"):
        with open("training.pid", "r") as f:
            pid = f.read().strip()
        
        try:
            os.kill(int(pid), 0)
            print(f"✓ 训练进程运行中 (PID: {pid})")
        except OSError:
            print(f"✗ 训练进程已停止 (PID: {pid})")
            return
    else:
        print("✗ 未找到训练进程")
        return
    
    # 检查最新日志
    log_files = glob("logs/training_*.log")
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"✓ 日志文件: {latest_log}")
        
        # 读取最后几行
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("\n最新日志:")
                    for line in lines[-3:]:
                        if line.strip():
                            print(f"  {line.strip()}")
        except Exception as e:
            print(f"✗ 读取日志失败: {e}")
    
    # 检查结果目录
    results_dir = "gram_negative_results_server"
    if os.path.exists(results_dir):
        checkpoints = glob(f"{results_dir}/checkpoint_epoch_*.pth")
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
            print(f"✓ 最新检查点: Epoch {epoch}")
        else:
            print("⏳ 还未生成检查点文件")
    
    print(f"\n检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    quick_check()
