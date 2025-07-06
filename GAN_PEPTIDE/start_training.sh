#!/bin/bash

# Multi-CGAN 训练启动脚本
# 使用方法: ./start_training.sh [gpu_id] [batch_size] [epochs]

set -e  # 遇到错误立即退出

# 默认参数
GPU_ID=${1:-0}
BATCH_SIZE=${2:-32}
NUM_EPOCHS=${3:-1000}
PROJECT_DIR=$(pwd)

echo "=== Multi-CGAN 训练启动脚本 ==="
echo "项目目录: $PROJECT_DIR"
echo "GPU ID: $GPU_ID"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "================================"

# 检查环境
echo "检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "错误: Python未找到"
    exit 1
fi

echo "Python版本: $(python --version)"

# 检查GPU
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
else
    echo "警告: nvidia-smi未找到，可能没有GPU或驱动未安装"
fi

# 检查数据文件
echo "检查数据文件..."
DATA_FILES=(
    "data/gram_negative_train_pos.npy"
    "data/gram_negative_train_labels.npy"
    "data/gram_negative_val_pos.npy" 
    "data/gram_negative_val_labels.npy"
)

for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 数据文件 $file 不存在"
        echo "请先运行数据预处理: python preprocess_gram_negative.py"
        exit 1
    fi
done

echo "数据文件检查完成 ✓"

# 创建必要目录
mkdir -p logs
mkdir -p gram_negative_results_server

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "开始训练..."
echo "日志文件: $LOG_FILE"

# 启动训练
nohup python train_gram_negative_server.py \
    --gpu_id $GPU_ID \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --save_interval 50 \
    --sample_interval 100 \
    --log_dir logs \
    --save_dir gram_negative_results_server > $LOG_FILE 2>&1 &

TRAIN_PID=$!
echo "训练已启动!"
echo "进程ID: $TRAIN_PID"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f $LOG_FILE"
echo "  查看进程: ps aux | grep $TRAIN_PID"
echo "  杀死进程: kill $TRAIN_PID"
echo "  GPU监控: watch -n 1 nvidia-smi"
echo ""
echo "训练完成后结果将保存在: gram_negative_results_server/"

# 保存PID到文件
echo $TRAIN_PID > training.pid
echo "进程ID已保存到 training.pid"
