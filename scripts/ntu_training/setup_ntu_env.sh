#!/bin/bash
# NTU GPU服务器环境设置脚本

set -e

echo "=== 设置NTU GPU环境 ==="

# 加载模块（如果集群使用module）
if command -v module >/dev/null 2>&1; then
    module load gcc/14.2.0 || true
fi

# 显示CUDA信息（若可用）
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi
if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA版本: $(nvcc --version | grep release | awk '{print $6}')" || true
fi

# 准备conda环境
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 创建并激活环境
conda create -n ch-3dgs-gpu python=3.10 -y || echo "环境可能已存在"
conda activate ch-3dgs-gpu || source activate ch-3dgs-gpu

# 根据CUDA版本安装PyTorch
CUDA_VERSION="$( (nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1) || echo '')"
echo "检测到CUDA版本: ${CUDA_VERSION}"
if [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到匹配CUDA，安装CPU版PyTorch（如需要可手动改为GPU版）"
    pip install torch torchvision torchaudio
fi

# 安装项目依赖
cd "$HOME/Projects/cultural-heritage-3dgs"
pip install -r requirements.txt

# 克隆并安装gaussian-splatting
if [ ! -d "$HOME/Projects/cultural-heritage-3dgs/gaussian-splatting" ]; then
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$HOME/Projects/cultural-heritage-3dgs/gaussian-splatting"
    cd "$HOME/Projects/cultural-heritage-3dgs/gaussian-splatting"
    pip install -r requirements.txt || true
    pip install -e .
else
    echo "gaussian-splatting已存在"
fi

echo "环境设置完成！"
echo "下一步: 运行 train_all.sh 或 run_complete_experiment.sh 开始训练"
