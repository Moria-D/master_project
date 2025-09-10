#!/bin/bash

# NTU 计算节点环境检查脚本
echo "=== NTU 计算节点环境检查 ==="
echo "时间: $(date)"
echo "用户: $(whoami)"
echo "主机名: $(hostname)"
echo "当前目录: $(pwd)"

echo ""
echo "=== Python 环境 ==="
python --version
which python
echo "Python 路径:"
python -c "import sys; print('\n'.join(sys.path))"

echo ""
echo "=== CUDA 环境 ==="
nvidia-smi || echo "nvidia-smi 不可用"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"

echo ""
echo "=== 检查必要的 Python 包 ==="
python -c "
try:
    import torch
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA 设备数量: {torch.cuda.device_count()}')
        print(f'当前 CUDA 设备: {torch.cuda.current_device()}')
        print(f'设备名称: {torch.cuda.get_device_name()}')
except ImportError:
    print('PyTorch 未安装')

try:
    import numpy
    print(f'NumPy 版本: {numpy.__version__}')
except ImportError:
    print('NumPy 未安装')

try:
    import PIL
    print(f'PIL 版本: {PIL.__version__}')
except ImportError:
    print('PIL 未安装')
"

echo ""
echo "=== 检查项目文件 ==="
echo "数据目录:"
ls -la data/mipnerf360/ 2>/dev/null || echo "mipnerf360 数据不存在"

echo ""
echo "gaussian-splatting 目录:"
ls -la gaussian-splatting/ 2>/dev/null || echo "gaussian-splatting 目录不存在"

echo ""
echo "=== 检查训练脚本 ==="
if [ -f "gaussian-splatting/train.py" ]; then
    echo "✓ train.py 存在"
else
    echo "✗ train.py 不存在"
fi

if [ -f "gaussian-splatting/render.py" ]; then
    echo "✓ render.py 存在"
else
    echo "✗ render.py 不存在"
fi

echo ""
echo "=== 环境检查完成 ==="
echo "如果所有检查都通过，可以运行:"
echo "  ./quick_test.sh room 0    # 快速测试"
echo "  ./run_ntu_experiment.sh room 0    # 完整实验"
