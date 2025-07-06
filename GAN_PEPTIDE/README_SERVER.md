# Multi-CGAN 服务器部署版本

基于Gram-负性菌抗菌肽数据集的条件生成对抗网络训练系统。

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n multi_cgan python=3.9
conda activate multi_cgan

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 如果数据文件不存在，运行预处理
python preprocess_gram_negative.py
```

### 3. 启动训练

```bash
# 方法1: 使用启动脚本 (推荐)
chmod +x start_training.sh
./start_training.sh

# 方法2: 直接运行
python train_gram_negative_server.py --gpu_id 0 --batch_size 32 --num_epochs 1000
```

### 4. 监控训练

```bash
# GPU监控
./monitor_gpu.sh

# 训练状态检查
python check_training.py --all

# 查看日志
tail -f logs/training_*.log
```

## 文件说明

### 核心文件
- `train_gram_negative_server.py`: 服务器版训练脚本
- `preprocess_gram_negative.py`: 数据预处理脚本
- `Gram-.fasta`: 原始数据集

### 辅助脚本
- `start_training.sh`: 训练启动脚本
- `monitor_gpu.sh`: GPU监控脚本
- `check_training.py`: 训练状态检查
- `backup_results.sh`: 结果备份脚本

### 配置文件
- `requirements.txt`: Python依赖包
- `environment_setup.md`: 环境配置指南
- `SERVER_DEPLOYMENT_GUIDE.md`: 详细部署指南

## 训练参数

### 默认配置
- 批次大小: 32
- 学习率: 5e-5
- 训练轮数: 1000
- GPU: 自动检测

### 自定义参数
```bash
python train_gram_negative_server.py \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --gpu_id 0
```

## 结果文件

训练完成后，结果保存在 `gram_negative_results_server/` 目录：

- `checkpoint_epoch_*.pth`: 训练检查点
- `generated_epoch_*.npy`: 生成的肽段序列
- `final_model.pth`: 最终模型
- `loss_history.npy`: 损失历史
- `config.json`: 训练配置

## 常用命令

### 训练管理
```bash
# 启动训练
./start_training.sh 0 32 1000  # GPU_ID BATCH_SIZE EPOCHS

# 恢复训练
python train_gram_negative_server.py --resume gram_negative_results_server/checkpoint_epoch_500.pth

# 停止训练
kill $(cat training.pid)
```

### 监控和分析
```bash
# 检查训练状态
python check_training.py --status

# 分析损失
python check_training.py --loss

# 分析生成样本
python check_training.py --samples 100

# GPU监控
./monitor_gpu.sh 2  # 2秒刷新间隔
```

### 结果处理
```bash
# 分析生成样本
python monitor_training.py analyze 100

# 转换为FASTA格式
python monitor_training.py fasta 100 generated_peptides.fasta

# 备份结果
./backup_results.sh
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   python train_gram_negative_server.py --batch_size 16
   ```

2. **训练中断恢复**
   ```bash
   # 找到最新检查点
   ls -t gram_negative_results_server/checkpoint_*.pth | head -1
   
   # 恢复训练
   python train_gram_negative_server.py --resume [checkpoint_path]
   ```

3. **权限问题**
   ```bash
   chmod +x *.sh
   chmod -R 755 .
   ```

### 性能优化

1. **GPU优化**
   - 使用合适的batch_size
   - 启用cudnn.benchmark
   - 监控GPU使用率

2. **数据优化**
   - 将数据存储在SSD上
   - 增加DataLoader的num_workers

3. **训练优化**
   - 调整学习率
   - 使用学习率调度器
   - 考虑混合精度训练

## 技术支持

### 日志分析
- 训练日志: `logs/training_*.log`
- 错误信息: 检查日志文件末尾
- 性能指标: 使用 `check_training.py`

### 联系方式
如有问题，请检查：
1. 环境配置是否正确
2. 数据文件是否完整
3. GPU驱动是否正常
4. 日志文件中的错误信息

## 版本信息

- 版本: 1.0
- 更新日期: 2025-01-06
- 支持的Python版本: 3.7-3.10
- 支持的PyTorch版本: 1.9+
