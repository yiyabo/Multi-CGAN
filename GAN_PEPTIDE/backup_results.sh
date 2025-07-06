#!/bin/bash

# 结果备份脚本
# 使用方法: ./backup_results.sh [backup_dir]

set -e

BACKUP_DIR=${1:-"backups"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="multi_cgan_backup_${TIMESTAMP}"
FULL_BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

echo "=== Multi-CGAN 结果备份脚本 ==="
echo "备份目录: $FULL_BACKUP_PATH"
echo "时间戳: $TIMESTAMP"
echo "================================"

# 创建备份目录
mkdir -p "$FULL_BACKUP_PATH"

# 要备份的目录和文件
ITEMS_TO_BACKUP=(
    "gram_negative_results_server"
    "logs"
    "data/gram_negative_*.npy"
    "*.py"
    "*.md"
    "*.txt"
    "*.sh"
)

echo "开始备份..."

# 备份文件
for item in "${ITEMS_TO_BACKUP[@]}"; do
    if ls $item 1> /dev/null 2>&1; then
        echo "备份: $item"
        cp -r $item "$FULL_BACKUP_PATH/" 2>/dev/null || true
    else
        echo "跳过: $item (不存在)"
    fi
done

# 创建备份信息文件
cat > "$FULL_BACKUP_PATH/backup_info.txt" << EOF
备份信息
========
备份时间: $(date)
备份目录: $FULL_BACKUP_PATH
主机名: $(hostname)
用户: $(whoami)
工作目录: $(pwd)

备份内容:
EOF

# 列出备份内容
find "$FULL_BACKUP_PATH" -type f | sort >> "$FULL_BACKUP_PATH/backup_info.txt"

# 计算备份大小
BACKUP_SIZE=$(du -sh "$FULL_BACKUP_PATH" | cut -f1)
echo "备份大小: $BACKUP_SIZE" >> "$FULL_BACKUP_PATH/backup_info.txt"

echo ""
echo "备份完成!"
echo "备份位置: $FULL_BACKUP_PATH"
echo "备份大小: $BACKUP_SIZE"

# 可选：创建压缩包
read -p "是否创建压缩包? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "创建压缩包..."
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    echo "压缩包已创建: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    
    # 询问是否删除原始备份目录
    read -p "是否删除原始备份目录? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BACKUP_NAME"
        echo "原始备份目录已删除"
    fi
fi

echo "备份脚本执行完成!"
