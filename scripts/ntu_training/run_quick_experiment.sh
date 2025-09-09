#!/bin/bash
# 快速实验脚本 - 用于快速验证多尺度方法效果

set -e

echo "=== 快速实验验证 ==="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ch-3dgs-gpu

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 快速实验参数
SCENE_NAME="room"  # 使用room场景
ITERATIONS=5000    # 减少迭代次数，快速验证
SAVE_ITERATIONS=1000
TEST_ITERATIONS=1000

echo "实验场景: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"

# 数据检查
DATA_DIR="~/Projects/cultural-heritage-3dgs/data/mipnerf360/$SCENE_NAME"
if [ ! -f "$DATA_DIR/transforms_train.json" ]; then
    echo "错误: transforms_train.json 不存在"
    exit 1
fi

echo "数据准备完成 ✓"

# 快速Baseline训练
echo "=== 快速Baseline训练 ==="
BASELINE_DIR="~/Projects/cultural-heritage-3dgs/exp/baseline_3dgs/$SCENE_NAME"
mkdir -p "$BASELINE_DIR"

cd ~/Projects/cultural-heritage-3dgs/gaussian-splatting

echo "开始Baseline训练..."
python train.py \
    -s "$DATA_DIR" \
    -m "$BASELINE_DIR" \
    --iterations $ITERATIONS \
    --eval \
    --save_iterations $SAVE_ITERATIONS \
    --test_iterations $TEST_ITERATIONS \
    --quiet

echo "Baseline训练完成 ✓"

# 快速多尺度训练
echo "=== 快速多尺度训练 ==="
MULTISCALE_DIR="~/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs/$SCENE_NAME"
mkdir -p "$MULTISCALE_DIR"

cd ~/Projects/cultural-heritage-3dgs

echo "开始多尺度训练..."
python scripts/training/train_multiscale.py \
    -s "$DATA_DIR" \
    -m "$MULTISCALE_DIR" \
    --iterations $ITERATIONS \
    --use_multiscale \
    --density_from_grad \
    --lod_levels 4 \
    --gradient_threshold 0.1 \
    --eval \
    --save_iterations $SAVE_ITERATIONS \
    --test_iterations $TEST_ITERATIONS \
    --quiet

echo "多尺度训练完成 ✓"

# 快速渲染测试
echo "=== 快速渲染测试 ==="
RENDER_DIR="~/Projects/cultural-heritage-3dgs/eval/renders"
mkdir -p "$RENDER_DIR/baseline"
mkdir -p "$RENDER_DIR/multiscale"

cd ~/Projects/cultural-heritage-3dgs/gaussian-splatting

# 只渲染几张测试图像
echo "渲染Baseline测试图像..."
python render.py \
    -m "$BASELINE_DIR" \
    --test_transforms "$DATA_DIR/transforms_test.json" \
    --outdir "$RENDER_DIR/baseline" \
    --quiet

echo "渲染多尺度测试图像..."
python render.py \
    -m "$MULTISCALE_DIR" \
    --test_transforms "$DATA_DIR/transforms_test.json" \
    --outdir "$RENDER_DIR/multiscale" \
    --quiet

echo "渲染完成 ✓"

# 快速评估
echo "=== 快速评估 ==="
cd ~/Projects/cultural-heritage-3dgs

METRICS_DIR="eval/metrics"
mkdir -p "$METRICS_DIR"

echo "计算质量指标..."
python scripts/evaluation/evaluate_quality.py \
    --gt_dir "$DATA_DIR/images" \
    --baseline_dir "$RENDER_DIR/baseline" \
    --multiscale_dir "$RENDER_DIR/multiscale" \
    --output_dir "$METRICS_DIR"

# 模型大小对比
echo "模型大小对比..."
BASELINE_SIZE=$(du -sh "$BASELINE_DIR" | cut -f1)
MULTISCALE_SIZE=$(du -sh "$MULTISCALE_DIR" | cut -f1)

echo "快速实验结果:" > "$METRICS_DIR/quick_experiment_results.txt"
echo "Baseline模型大小: $BASELINE_SIZE" >> "$METRICS_DIR/quick_experiment_results.txt"
echo "Multiscale模型大小: $MULTISCALE_SIZE" >> "$METRICS_DIR/quick_experiment_results.txt"
echo "训练迭代: $ITERATIONS" >> "$METRICS_DIR/quick_experiment_results.txt"

echo "=== 快速实验完成 ==="
echo "结果保存在: $METRICS_DIR/quick_experiment_results.txt"
echo ""
echo "下一步: 如果快速实验成功，可以运行完整实验"
echo "chmod +x scripts/ntu_training/run_complete_experiment.sh"
echo "./scripts/ntu_training/run_complete_experiment.sh"
