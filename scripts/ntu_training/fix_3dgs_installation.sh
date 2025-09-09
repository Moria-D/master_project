#!/bin/bash
# 修复3DGS安装问题

set -e

echo "=== 修复3DGS安装 ==="

cd ~/Projects/cultural-heritage-3dgs

# 检查gaussian-splatting目录
if [ ! -d "gaussian-splatting" ]; then
    echo "错误: gaussian-splatting目录不存在"
    exit 1
fi

cd gaussian-splatting

echo "检查gaussian-splatting目录结构..."
ls -la

# 查找requirements文件
echo "查找requirements文件..."
find . -name "*requirements*" -type f

# 检查是否有setup.py或pyproject.toml
echo "检查安装文件..."
ls -la | grep -E "(setup\.py|pyproject\.toml|requirements)"

# 尝试不同的安装方法
echo "尝试安装3DGS..."

# 方法1: 如果有setup.py，直接安装
if [ -f "setup.py" ]; then
    echo "使用setup.py安装..."
    pip install -e .
elif [ -f "pyproject.toml" ]; then
    echo "使用pyproject.toml安装..."
    pip install -e .
else
    echo "手动安装依赖..."
    # 手动安装3DGS的依赖
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy scipy scikit-image opencv-python matplotlib seaborn pandas tqdm lpips
    pip install plyfile imageio imageio-ffmpeg
    pip install pytorch3d
    pip install trimesh
    pip install open3d
    pip install ninja
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
fi

echo "3DGS安装完成！"

# 验证安装
echo "验证安装..."
python -c "
try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU数量: {torch.cuda.device_count()}')
        print(f'当前GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'PyTorch导入错误: {e}')
"

echo "=== 修复完成 ==="
