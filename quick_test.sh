#!/bin/bash

# 快速测试脚本 - 验证环境和运行短时间训练
# 使用方法: ./quick_test.sh <scene_name> <gpu_id>
# 例如: ./quick_test.sh room 0

set -e

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <scene_name> <gpu_id>"
    echo "例如: $0 room 0"
    exit 1
fi

SCENE_NAME=$1
GPU_ID=$2
DATA_DIR="data/mipnerf360/${SCENE_NAME}"
EXP_DIR="exp/quick_test/${SCENE_NAME}"

echo "=== 快速测试开始 ==="
echo "场景: ${SCENE_NAME}"
echo "GPU ID: ${GPU_ID}"
echo "时间: $(date)"

# 检查环境
echo "=== 环境检查 ==="
echo "Python 版本:"
python --version
echo "CUDA 版本:"
nvidia-smi | grep "CUDA Version" || echo "无法获取 CUDA 版本"
echo "GPU 状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# 检查数据
if [ ! -f "${DATA_DIR}/transforms_train.json" ]; then
    echo "错误: 数据不存在: ${DATA_DIR}/transforms_train.json"
    exit 1
fi

# 创建测试目录
mkdir -p "${EXP_DIR}"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 进入 gaussian-splatting 目录
cd gaussian-splatting

echo "=== 开始快速训练 (1000 次迭代) ==="
echo "开始时间: $(date)"

# 快速训练 - 只运行 1000 次迭代用于测试
python train.py \
    -s "../${DATA_DIR}" \
    -m "../${EXP_DIR}" \
    --iterations 1000 \
    --eval \
    --save_iterations 1000 \
    --test_iterations 1000 \
    --quiet

echo "快速训练完成时间: $(date)"

# 返回项目根目录
cd ..

echo "=== 快速测试完成 ==="
echo "测试结果保存在: ${EXP_DIR}"
echo "如果看到此消息，说明环境配置正确！"
