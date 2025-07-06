#!/bin/bash

# 快速重启脚本 - 修复PyTorch加载问题后使用

echo "=== 快速重启训练 ==="

# 停止当前进程
if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    echo "停止进程 $PID..."
    kill $PID 2>/dev/null || echo "进程已停止"
    sleep 2
    rm -f training.pid
fi

# 直接重启，从检查点恢复
LATEST_CHECKPOINT=$(ls -t gram_negative_results_server/checkpoint_epoch_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CHECKPOINT" ]; then
    EPOCH=$(echo $LATEST_CHECKPOINT | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
    echo "从检查点恢复: Epoch $EPOCH"
    
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="logs/training_fixed_${TIMESTAMP}.log"
    
    nohup python train_gram_negative_server.py \
        --gpu_id 0 \
        --batch_size 64 \
        --num_epochs 1500 \
        --save_interval 50 \
        --sample_interval 100 \
        --resume "$LATEST_CHECKPOINT" > $LOG_FILE 2>&1 &
    
    NEW_PID=$!
    echo $NEW_PID > training.pid
    echo "训练已重启! PID: $NEW_PID"
    echo "日志: $LOG_FILE"
    echo "监控: tail -f $LOG_FILE"
else
    echo "未找到检查点，从头开始训练"
    ./start_training.sh 0 64 1500
fi
