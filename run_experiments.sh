#!/bin/bash

# 主实验控制脚本
# 使用方法: ./run_experiments.sh <artifact_name> <image_dir>
# 例如: ./run_experiments.sh artifact_1 /path/to/images

set -e

if [ $# -ne 2 ]; then
    echo "使用方法: $0 <artifact_name> <image_dir>"
    echo "例如: $0 artifact_1 /path/to/images"
    exit 1
fi

ARTIFACT_NAME=$1
IMAGE_DIR=$2

echo "=== 文物数字化实验开始 ==="
echo "文物名称: ${ARTIFACT_NAME}"
echo "图像目录: ${IMAGE_DIR}"

# 步骤1: COLMAP重建
echo "=== 步骤1: COLMAP重建 ==="
cd scripts/colmap
./run_colmap.sh "${ARTIFACT_NAME}" "${IMAGE_DIR}"
cd ../..

# 步骤2: 生成transforms.json (需要手动完成)
echo "=== 步骤2: 生成训练数据 ==="
echo "请手动生成transforms_train.json和transforms_test.json"
echo "可以使用nerfstudio或其他工具进行转换"
echo "完成后按回车继续..."
read

# 步骤3: 训练Baseline模型 (在GPU上执行)
echo "=== 步骤3: 训练Baseline模型 ==="
echo "请在GPU服务器上执行:"
echo "cd scripts/training"
echo "./train_baseline.sh ${ARTIFACT_NAME} 0"
echo "完成后按回车继续..."
read

# 步骤4: 训练多尺度模型 (在GPU上执行)
echo "=== 步骤4: 训练多尺度模型 ==="
echo "请在GPU服务器上执行:"
echo "cd scripts/training"
echo "./train_multiscale.sh ${ARTIFACT_NAME} 0"
echo "完成后按回车继续..."
read

# 步骤5: 渲染测试图像 (在GPU上执行)
echo "=== 步骤5: 渲染测试图像 ==="
echo "请在GPU服务器上执行渲染命令"
echo "完成后按回车继续..."
read

# 步骤6: 评估质量
echo "=== 步骤6: 评估重建质量 ==="
python scripts/evaluation/evaluate_quality.py \
    --gt_dir "data/${ARTIFACT_NAME}/images_test" \
    --baseline_dir "eval/renders/baseline" \
    --multiscale_dir "eval/renders/multiscale" \
    --output_dir "eval/metrics"

# 步骤7: 评估效率
echo "=== 步骤7: 评估渲染效率 ==="
python scripts/evaluation/evaluate_efficiency.py \
    --baseline_csv "eval/metrics/baseline_fps.csv" \
    --multiscale_csv "eval/metrics/multiscale_fps.csv" \
    --output_dir "eval/plots"

# 步骤8: 生成报告
echo "=== 步骤8: 生成实验报告 ==="
echo "实验完成!"
echo "结果保存在:"
echo "  - 质量指标: eval/metrics/"
echo "  - 效率分析: eval/plots/"
echo "  - 渲染图像: eval/renders/"

