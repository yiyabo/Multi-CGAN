# 服务器部署和运行指南

## 快速开始

### 1. 上传项目到服务器

```bash
# 方法1: 使用scp
scp -r /path/to/Multi-CGAN username@server_ip:/path/to/destination/

# 方法2: 使用rsync (推荐)
rsync -avz --progress /path/to/Multi-CGAN/ username@server_ip:/path/to/destination/

# 方法3: 使用git
git clone https://github.com/your-repo/Multi-CGAN.git
cd Multi-CGAN/GAN_PEPTIDE
```

### 2. 环境配置

```bash
# 登录服务器
ssh username@server_ip

# 进入项目目录
cd /path/to/Multi-CGAN/GAN_PEPTIDE

# 创建虚拟环境
conda create -n multi_cgan python=3.9
conda activate multi_cgan

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 验证环境
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 3. 数据准备

确保数据文件已正确上传：
```bash
ls -la data/
# 应该看到:
# gram_negative_train_pos.npy
# gram_negative_train_labels.npy
# gram_negative_val_pos.npy
# gram_negative_val_labels.npy
```

如果数据文件不存在，运行预处理：
```bash
python preprocess_gram_negative.py
```

## 训练启动方式

### 方法1: 直接运行 (前台)

```bash
# 基本运行
python train_gram_negative_server.py

# 指定参数运行
python train_gram_negative_server.py \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 1000 \
    --gpu_id 0 \
    --save_interval 50
```

### 方法2: 后台运行 (推荐)

```bash
# 使用nohup
nohup python train_gram_negative_server.py \
    --batch_size 32 \
    --num_epochs 1000 \
    --gpu_id 0 > training.log 2>&1 &

# 查看进程
ps aux | grep train_gram_negative_server

# 查看日志
tail -f training.log
```

### 方法3: 使用screen (推荐)

```bash
# 创建screen会话
screen -S multi_cgan_training

# 在screen中运行训练
python train_gram_negative_server.py --num_epochs 1000 --gpu_id 0

# 分离screen (Ctrl+A, D)
# 重新连接: screen -r multi_cgan_training
# 查看所有会话: screen -ls
```

### 方法4: 使用tmux

```bash
# 创建tmux会话
tmux new-session -d -s multi_cgan_training

# 在tmux中运行
tmux send-keys -t multi_cgan_training "python train_gram_negative_server.py --num_epochs 1000" Enter

# 查看会话
tmux list-sessions

# 连接会话
tmux attach-session -t multi_cgan_training
```

## 训练参数说明

### 基本参数
- `--data_dir`: 数据目录 (默认: data)
- `--save_dir`: 结果保存目录 (默认: gram_negative_results_server)
- `--log_dir`: 日志目录 (默认: logs)
- `--gpu_id`: GPU ID (默认: 0)

### 训练参数
- `--batch_size`: 批次大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 5e-5)
- `--num_epochs`: 训练轮数 (默认: 1000)
- `--save_interval`: 保存间隔 (默认: 50)
- `--sample_interval`: 生成样本间隔 (默认: 100)

### 恢复训练
- `--resume`: 检查点路径，用于恢复训练

### 示例命令

```bash
# 高性能GPU训练
python train_gram_negative_server.py \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --num_epochs 2000 \
    --gpu_id 0 \
    --save_interval 25 \
    --sample_interval 50

# 低内存GPU训练
python train_gram_negative_server.py \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_epochs 1500 \
    --gpu_id 0

# 恢复训练
python train_gram_negative_server.py \
    --resume gram_negative_results_server/checkpoint_epoch_500.pth \
    --num_epochs 1000
```

## 监控和管理

### 1. 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 2. 监控训练进度

```bash
# 查看日志
tail -f logs/training_*.log

# 查看保存的文件
ls -la gram_negative_results_server/

# 使用监控脚本
python monitor_training.py monitor
```

### 3. 分析生成样本

```bash
# 分析最新生成的样本
python monitor_training.py analyze

# 分析特定轮次的样本
python monitor_training.py analyze 100

# 转换为FASTA格式
python monitor_training.py fasta 100 generated_peptides.fasta
```

## 故障排除

### 1. 内存不足 (OOM)

```bash
# 减小批次大小
python train_gram_negative_server.py --batch_size 16

# 或者使用梯度累积
# (需要修改代码实现)
```

### 2. 训练中断恢复

```bash
# 找到最新的检查点
ls -t gram_negative_results_server/checkpoint_*.pth | head -1

# 恢复训练
python train_gram_negative_server.py \
    --resume gram_negative_results_server/checkpoint_epoch_XXX.pth
```

### 3. 多GPU使用

```bash
# 指定特定GPU
export CUDA_VISIBLE_DEVICES=1
python train_gram_negative_server.py --gpu_id 0

# 或者直接指定
CUDA_VISIBLE_DEVICES=1 python train_gram_negative_server.py --gpu_id 0
```

### 4. 权限问题

```bash
# 确保目录权限
chmod -R 755 /path/to/Multi-CGAN
chown -R username:usergroup /path/to/Multi-CGAN
```

## 结果文件说明

训练完成后，`gram_negative_results_server/` 目录包含：

- `checkpoint_epoch_*.pth`: 训练检查点
- `generated_epoch_*.npy`: 生成的样本 (数字编码)
- `final_model.pth`: 最终模型
- `loss_history.npy`: 损失历史
- `config.json`: 训练配置

日志文件在 `logs/` 目录中。

## 性能优化建议

### 1. 数据加载优化
- 将数据存储在SSD上
- 增加DataLoader的num_workers

### 2. 训练优化
- 使用合适的batch_size (GPU内存允许的最大值)
- 启用cudnn.benchmark
- 考虑使用混合精度训练

### 3. 监控优化
- 定期清理日志文件
- 监控磁盘空间使用
- 设置自动备份重要检查点

## 自动化脚本

我已经为您创建了以下辅助脚本，详见项目目录：

- `start_training.sh`: 启动训练脚本
- `monitor_gpu.sh`: GPU监控脚本
- `backup_results.sh`: 结果备份脚本
- `check_training.py`: 训练状态检查脚本

使用方法：
```bash
chmod +x *.sh
./start_training.sh
```
