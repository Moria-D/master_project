#!/bin/bash
# 最小化依赖安装脚本

set -e

echo "=== 安装最小化依赖 ==="

cd ~/Projects/cultural-heritage-3dgs/gaussian-splatting

# 检查CUDA版本
echo "检查CUDA版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
    echo "CUDA版本: $CUDA_VERSION"
else
    echo "nvcc不可用，使用默认CUDA版本"
    CUDA_VERSION="11.8"
fi

# 安装PyTorch（根据CUDA版本）
echo "安装PyTorch..."
if [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
fi

# 安装基本依赖
echo "安装基本依赖..."
pip install numpy scipy scikit-image opencv-python matplotlib seaborn pandas tqdm lpips
pip install plyfile imageio imageio-ffmpeg
pip install ninja

# 尝试安装可选依赖（失败则跳过）
echo "安装可选依赖..."
pip install pytorch3d || echo "pytorch3d安装失败，跳过"
pip install trimesh || echo "trimesh安装失败，跳过"
pip install open3d || echo "open3d安装失败，跳过"

# 尝试安装tiny-cuda-nn
echo "安装tiny-cuda-nn..."
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || echo "tiny-cuda-nn安装失败，跳过"

# 验证安装
echo "验证安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== 最小化依赖安装完成 ==="
echo "现在可以尝试运行实验了"

