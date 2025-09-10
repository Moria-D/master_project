#!/bin/bash

# NTU 计算节点上的完整实验脚本
# 使用方法: ./run_ntu_experiment.sh <scene_name> <gpu_id>
# 例如: ./run_ntu_experiment.sh room 0

set -e

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <scene_name> <gpu_id>"
    echo "可用场景: room, bicycle, garden"
    echo "例如: $0 room 0"
    exit 1
fi

SCENE_NAME=$1
GPU_ID=$2
DATA_DIR="data/mipnerf360/${SCENE_NAME}"
EXP_DIR="exp/baseline_3dgs/mipnerf360/${SCENE_NAME}"

echo "=== 开始 3DGS 实验 ==="
echo "场景: ${SCENE_NAME}"
echo "数据目录: ${DATA_DIR}"
echo "实验目录: ${EXP_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "时间: $(date)"

# 检查数据目录
if [ ! -f "${DATA_DIR}/transforms_train.json" ]; then
    echo "错误: 训练数据不存在: ${DATA_DIR}/transforms_train.json"
    exit 1
fi

# 创建实验目录
mkdir -p "${EXP_DIR}"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 检查 GPU 可用性
echo "=== 检查 GPU 状态 ==="
nvidia-smi

# 进入 gaussian-splatting 目录
cd gaussian-splatting

echo "=== 开始训练 Baseline 3DGS ==="
echo "开始时间: $(date)"

# 训练命令
python train.py \
    -s "../${DATA_DIR}" \
    -m "../${EXP_DIR}" \
    --iterations 30000 \
    --eval \
    --save_iterations 30000 \
    --test_iterations 30000 \
    --quiet

echo "训练完成时间: $(date)"

# 渲染测试图像
echo "=== 开始渲染测试图像 ==="
RENDER_DIR="../${EXP_DIR}/test/ours_30000/renders"
mkdir -p "${RENDER_DIR}"

python render.py \
    -m "../${EXP_DIR}" \
    --skip_train \
    --skip_test

echo "渲染完成时间: $(date)"

# 返回项目根目录
cd ..

# 计算质量指标
echo "=== 计算质量指标 ==="
python scripts/evaluation/evaluate_quality.py \
    --gt_dir "${DATA_DIR}/images" \
    --baseline_dir "${EXP_DIR}/test/ours_30000/renders" \
    --output_dir "eval/metrics" \
    --scene_name "${SCENE_NAME}"

echo "=== 实验完成 ==="
echo "完成时间: $(date)"
echo "结果保存在: ${EXP_DIR}"
echo "质量指标保存在: eval/metrics/"

# 显示结果摘要
if [ -f "eval/metrics/${SCENE_NAME}_metrics.json" ]; then
    echo "=== 质量指标摘要 ==="
    cat "eval/metrics/${SCENE_NAME}_metrics.json"
fi
