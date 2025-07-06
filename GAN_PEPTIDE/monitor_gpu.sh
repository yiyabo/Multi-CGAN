#!/bin/bash

# GPU监控脚本
# 使用方法: ./monitor_gpu.sh [interval]

INTERVAL=${1:-2}  # 默认2秒刷新一次

echo "=== GPU监控 (每${INTERVAL}秒刷新) ==="
echo "按 Ctrl+C 退出"
echo ""

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi未找到"
    echo "请确保已安装NVIDIA驱动"
    exit 1
fi

# 显示GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits

echo ""
echo "开始监控..."
echo ""

# 实时监控
while true; do
    clear
    echo "=== GPU状态监控 $(date) ==="
    echo ""
    
    # 显示GPU使用情况
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    while IFS=', ' read -r index name gpu_util mem_used mem_total temp power; do
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "N/A")
        printf "GPU %s (%s):\n" "$index" "$name"
        printf "  GPU使用率: %s%%\n" "$gpu_util"
        printf "  显存使用: %s MB / %s MB (%.1f%%)\n" "$mem_used" "$mem_total" "$mem_percent"
        printf "  温度: %s°C\n" "$temp"
        printf "  功耗: %s W\n" "$power"
        echo ""
    done
    
    # 显示运行的进程
    echo "GPU进程:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | \
    while IFS=', ' read -r pid process_name used_mem; do
        printf "  PID %s: %s (显存: %s MB)\n" "$pid" "$process_name" "$used_mem"
    done
    
    # 检查训练进程
    if [ -f "training.pid" ]; then
        TRAIN_PID=$(cat training.pid)
        if ps -p $TRAIN_PID > /dev/null 2>&1; then
            echo ""
            echo "训练进程状态: 运行中 (PID: $TRAIN_PID)"
            
            # 显示最新日志
            if ls logs/training_*.log 1> /dev/null 2>&1; then
                LATEST_LOG=$(ls -t logs/training_*.log | head -1)
                echo "最新日志 ($LATEST_LOG):"
                tail -3 "$LATEST_LOG" | sed 's/^/  /'
            fi
        else
            echo ""
            echo "训练进程状态: 已停止"
        fi
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    
    sleep $INTERVAL
done
