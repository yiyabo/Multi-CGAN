# 服务器环境配置指南

## 系统要求

### 硬件要求
- **GPU**: 推荐NVIDIA GPU (GTX 1080Ti/RTX 2080/RTX 3080或更高)
- **内存**: 至少8GB RAM，推荐16GB+
- **存储**: 至少10GB可用空间
- **CPU**: 多核CPU，推荐8核+

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+/CentOS 7+) 或 Windows 10+
- **Python**: 3.7-3.10
- **CUDA**: 11.0+ (如果使用GPU)
- **cuDNN**: 对应CUDA版本的cuDNN

## 环境安装步骤

### 1. 创建Python虚拟环境

```bash
# 使用conda (推荐)
conda create -n multi_cgan python=3.9
conda activate multi_cgan

# 或使用venv
python -m venv multi_cgan_env
source multi_cgan_env/bin/activate  # Linux/Mac
# multi_cgan_env\Scripts\activate  # Windows
```

### 2. 安装PyTorch

根据您的CUDA版本选择合适的PyTorch版本：

```bash
# CUDA 11.8 (推荐)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# CPU版本 (不推荐用于训练)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
```

## GPU配置

### 检查GPU状态
```bash
# 查看GPU信息
nvidia-smi

# 查看CUDA版本
nvcc --version
```

### 多GPU配置
如果服务器有多个GPU，可以指定使用的GPU：

```bash
# 使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 使用多个GPU
export CUDA_VISIBLE_DEVICES=0,1
```

## 常见问题解决

### 1. CUDA版本不匹配
```bash
# 卸载现有PyTorch
pip uninstall torch torchvision

# 重新安装匹配的版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足
- 减小batch_size参数
- 使用梯度累积
- 启用混合精度训练

### 3. 权限问题
```bash
# 确保有写入权限
chmod +w /path/to/project
```

### 4. 网络问题
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 性能优化建议

### 1. 数据加载优化
- 设置适当的`num_workers`参数
- 使用SSD存储数据
- 预加载数据到内存

### 2. 训练优化
- 使用混合精度训练 (AMP)
- 启用cudnn benchmark
- 合理设置batch size

### 3. 监控工具
- 使用`nvidia-smi`监控GPU使用率
- 使用`htop`监控CPU和内存
- 使用`tensorboard`或`wandb`跟踪训练进度

## 环境变量配置

创建`.env`文件：
```bash
# GPU配置
CUDA_VISIBLE_DEVICES=0

# 数据路径
DATA_DIR=/path/to/data
RESULTS_DIR=/path/to/results
LOG_DIR=/path/to/logs

# 训练参数
BATCH_SIZE=32
LEARNING_RATE=5e-5
NUM_EPOCHS=1000
```
