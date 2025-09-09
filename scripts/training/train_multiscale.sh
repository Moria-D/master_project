#!/bin/bash

# 多尺度3DGS训练脚本
# 使用方法: ./train_multiscale.sh <artifact_name> <gpu_id>
# 例如: ./train_multiscale.sh artifact_1 0

set -e

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <artifact_name> <gpu_id>"
    echo "例如: $0 artifact_1 0"
    exit 1
fi

ARTIFACT_NAME=$1
GPU_ID=$2
DATA_DIR="../data/${ARTIFACT_NAME}"
EXP_DIR="../exp/multiscale_3dgs/${ARTIFACT_NAME}"

echo "开始训练多尺度模型: ${ARTIFACT_NAME}"
echo "数据目录: ${DATA_DIR}"
echo "实验目录: ${EXP_DIR}"
echo "GPU ID: ${GPU_ID}"

# 创建实验目录
mkdir -p "${EXP_DIR}"

# 检查数据目录是否存在
if [ ! -f "${DATA_DIR}/transforms_train.json" ]; then
    echo "错误: 训练数据不存在: ${DATA_DIR}/transforms_train.json"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "=== 开始训练 ==="
# 这里需要根据你的多尺度实现调整命令
python train_multiscale.py \
    -s "${DATA_DIR}" \
    -m "${EXP_DIR}" \
    --iterations 30000 \
    --use_multiscale \
    --density_from_grad \
    --lod_levels 4 \
    --eval \
    --save_iterations 30000 \
    --test_iterations 30000 \
    --quiet

echo "训练完成!"
echo "模型保存在: ${EXP_DIR}"
echo "下一步: 运行评估脚本"
