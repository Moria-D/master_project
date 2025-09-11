#!/bin/bash

# NTU 计算节点快速设置和运行脚本
echo "=== NTU 计算节点 3DGS 实验设置 ==="
echo "时间: $(date)"
echo "用户: $(whoami)"

PROJECT_DIR="$HOME/Projects/cultural-heritage-3dgs"

# 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 项目目录不存在: $PROJECT_DIR"
    echo "请先上传项目文件"
    exit 1
fi

cd "$PROJECT_DIR"
echo "进入项目目录: $(pwd)"

# 1. 加载 CUDA 模块
echo ""
echo "=== 1. 加载 CUDA 模块 ==="
if command -v module &> /dev/null; then
    module load cuda/12.8.0 2>/dev/null && echo "✅ CUDA 12.8.0 加载成功" || echo "⚠️  CUDA 模块加载失败"
else
    echo "⚠️  module 命令不可用"
fi

# 2. 设置 CUDA 环境变量
echo ""
echo "=== 2. 设置 CUDA 环境变量 ==="
export CUDA_HOME=/apps/cuda_12.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1

echo "CUDA_HOME: $CUDA_HOME"
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc 可用: $(nvcc --version | grep 'release' | awk '{print $5}')"
else
    echo "❌ nvcc 不可用"
fi

# 3. 安装/更新 Python 包
echo ""
echo "=== 3. 安装 Python 依赖 ==="
python3 -m pip install --upgrade pip

# 安装核心依赖
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install simple-knn plyfile tqdm numpy pillow

# 4. 编译 CUDA 扩展
echo ""
echo "=== 4. 编译 CUDA 扩展 ==="
cd gaussian-splatting

# 尝试编译 diff-gaussian-rasterization
echo "编译 diff-gaussian-rasterization..."
cd submodules/diff-gaussian-rasterization
python3 setup.py build_ext --inplace 2>/dev/null && python3 setup.py develop 2>/dev/null && echo "✅ diff-gaussian-rasterization 编译成功" || echo "⚠️  diff-gaussian-rasterization 编译失败，使用 CPU 版本"

# 编译 simple-knn
echo "编译 simple-knn..."
cd ../simple-knn
python3 setup.py build_ext --inplace 2>/dev/null && python3 setup.py develop 2>/dev/null && echo "✅ simple-knn 编译成功" || echo "❌ simple-knn 编译失败"

cd ../..
echo "返回项目根目录: $(pwd)"

# 5. 验证安装
echo ""
echo "=== 5. 验证安装 ==="
python3 -c "
import sys
print('Python 版本:', sys.version.split()[0])

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} 可用')
    print(f'   CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU 数量: {torch.cuda.device_count()}')
        print(f'   GPU 名称: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ PyTorch 不可用: {e}')

try:
    import simple_knn
    print('✅ simple_knn 可用')
except ImportError as e:
    print(f'❌ simple_knn 不可用: {e}')

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    print('✅ diff_gaussian_rasterization 可用')
except ImportError as e:
    print(f'⚠️  diff_gaussian_rasterization 不可用，使用 CPU 版本')
    print('   这是正常的，继续运行实验')
"

# 6. 检查数据
echo ""
echo "=== 6. 检查实验数据 ==="
if [ -d "data/mipnerf360/room" ] && [ -f "data/mipnerf360/room/transforms_train.json" ]; then
    echo "✅ 实验数据准备就绪"
else
    echo "❌ 实验数据不完整"
    echo "请确保 data/mipnerf360/room/ 目录和 transforms_train.json 文件存在"
fi

echo ""
echo "=== 设置完成 ==="
echo ""
echo "🎯 现在可以运行实验了："
echo ""
echo "快速测试（推荐先运行）:"
echo "  ./quick_test.sh room 0"
echo ""
echo "完整实验:"
echo "  ./run_ntu_experiment.sh room 0"
echo ""
echo "其他可用场景: bicycle, garden"
echo ""
echo "环境检查:"
echo "  ./check_ntu_env.sh"
echo ""
echo "祝实验顺利！🚀"
