#!/bin/bash

# 重启训练脚本 - 修复错误后使用
# 使用方法: ./restart_training.sh [gpu_id] [batch_size] [epochs]

set -e

# 默认参数
GPU_ID=${1:-0}
BATCH_SIZE=${2:-64}
NUM_EPOCHS=${3:-1500}

echo "=== 重启训练脚本 ==="
echo "GPU ID: $GPU_ID"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "====================="

# 停止现有训练进程
if [ -f "training.pid" ]; then
    OLD_PID=$(cat training.pid)
    echo "停止现有训练进程 (PID: $OLD_PID)..."
    
    if ps -p $OLD_PID > /dev/null 2>&1; then
        kill $OLD_PID
        echo "已停止进程 $OLD_PID"
        sleep 2
    else
        echo "进程 $OLD_PID 已经停止"
    fi
    
    rm -f training.pid
fi

# 检查是否有现有的检查点
RESULTS_DIR="gram_negative_results_server"
RESUME_ARG=""

if [ -d "$RESULTS_DIR" ]; then
    # 查找最新的检查点
    LATEST_CHECKPOINT=$(ls -t $RESULTS_DIR/checkpoint_epoch_*.pth 2>/dev/null | head -1 || echo "")
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        EPOCH=$(echo $LATEST_CHECKPOINT | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
        echo "找到检查点: $LATEST_CHECKPOINT (Epoch $EPOCH)"
        
        read -p "是否从检查点恢复训练? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            RESUME_ARG="--resume $LATEST_CHECKPOINT"
            echo "将从 Epoch $EPOCH 恢复训练"
        else
            echo "将从头开始训练"
        fi
    fi
fi

# 生成新的时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_restart_${TIMESTAMP}.log"

echo "重启训练..."
echo "日志文件: $LOG_FILE"

# 启动训练
nohup python train_gram_negative_server.py \
    --gpu_id $GPU_ID \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --save_interval 50 \
    --sample_interval 100 \
    --log_dir logs \
    --save_dir gram_negative_results_server \
    $RESUME_ARG > $LOG_FILE 2>&1 &

NEW_PID=$!
echo "训练已重启!"
echo "新进程ID: $NEW_PID"
echo "日志文件: $LOG_FILE"

# 保存新的PID
echo $NEW_PID > training.pid
echo "进程ID已保存到 training.pid"

echo ""
echo "监控命令:"
echo "  查看日志: tail -f $LOG_FILE"
echo "  GPU监控: ./monitor_gpu.sh"
echo "  检查状态: python check_training.py --status"
