#!/bin/bash
# 完整实验执行脚本
# 在NTU GPU上运行完整的文物数字化实验

set -e

echo "=== 文物数字化多尺度高斯泼溅完整实验 ==="

# 激活环境（自动探测conda，不再硬编码路径）
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || true
    conda activate ch-3dgs-gpu || true
fi

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 实验参数
SCENE_NAME="mipnerf360/room"  # 默认使用room场景
ITERATIONS=30000
SAVE_ITERATIONS=5000
TEST_ITERATIONS=5000

echo "实验场景: $SCENE_NAME"
echo "训练迭代: $ITERATIONS"

# 步骤1: 数据准备检查
echo "=== 步骤1: 检查数据准备 ==="
DATA_DIR="$HOME/Projects/cultural-heritage-3dgs/data/$SCENE_NAME"
if [ ! -f "$DATA_DIR/transforms_train.json" ]; then
    echo "错误: transforms_train.json 不存在"
    echo "请先运行数据准备脚本"
    exit 1
fi

echo "数据准备完成 ✓"

# 步骤2: 训练Baseline模型
echo "=== 步骤2: 训练Baseline模型 ==="
BASELINE_DIR="$HOME/Projects/cultural-heritage-3dgs/exp/baseline_3dgs/$SCENE_NAME"
mkdir -p "$BASELINE_DIR"

cd "$HOME/Projects/cultural-heritage-3dgs/gaussian-splatting"

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

# 步骤3: 训练多尺度改进模型
echo "=== 步骤3: 训练多尺度改进模型 ==="
MULTISCALE_DIR="$HOME/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs/$SCENE_NAME"
mkdir -p "$MULTISCALE_DIR"

cd "$HOME/Projects/cultural-heritage-3dgs"

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

# 步骤4: 渲染测试图像
echo "=== 步骤4: 渲染测试图像 ==="
RENDER_DIR="$HOME/Projects/cultural-heritage-3dgs/eval/renders"
mkdir -p "$RENDER_DIR/baseline"
mkdir -p "$RENDER_DIR/multiscale"

cd "$HOME/Projects/cultural-heritage-3dgs/gaussian-splatting"

# Baseline渲染
echo "渲染Baseline测试图像..."
python render.py \
    -m "$BASELINE_DIR" \
    --test_transforms "$DATA_DIR/transforms_test.json" \
    --outdir "$RENDER_DIR/baseline" \
    --quiet

# 多尺度渲染
echo "渲染多尺度测试图像..."
python render.py \
    -m "$MULTISCALE_DIR" \
    --test_transforms "$DATA_DIR/transforms_test.json" \
    --outdir "$RENDER_DIR/multiscale" \
    --quiet

echo "渲染完成 ✓"

# 步骤5: 质量评估
echo "=== 步骤5: 质量评估 ==="
cd "$HOME/Projects/cultural-heritage-3dgs"

METRICS_DIR="eval/metrics"
mkdir -p "$METRICS_DIR"

echo "计算质量指标..."
python scripts/evaluation/evaluate_quality.py \
    --gt_dir "$DATA_DIR/images" \
    --baseline_dir "$RENDER_DIR/baseline" \
    --multiscale_dir "$RENDER_DIR/multiscale" \
    --output_dir "$METRICS_DIR"

echo "质量评估完成 ✓"

# 步骤6: 效率评估
echo "=== 步骤6: 效率评估 ==="
PLOTS_DIR="eval/plots"
mkdir -p "$PLOTS_DIR"

echo "计算效率指标..."
python scripts/evaluation/evaluate_efficiency.py \
    --baseline_model "$BASELINE_DIR" \
    --multiscale_model "$MULTISCALE_DIR" \
    --output_dir "$PLOTS_DIR"

echo "效率评估完成 ✓"

# 步骤7: 模型大小对比
echo "=== 步骤7: 模型大小对比 ==="
echo "计算模型文件大小..."

BASELINE_SIZE=$(du -sh "$BASELINE_DIR" | cut -f1)
MULTISCALE_SIZE=$(du -sh "$MULTISCALE_DIR" | cut -f1)

echo "模型大小对比:" > "$METRICS_DIR/model_size_comparison.txt"
echo "Baseline: $BASELINE_SIZE" >> "$METRICS_DIR/model_size_comparison.txt"
echo "Multiscale: $MULTISCALE_SIZE" >> "$METRICS_DIR/model_size_comparison.txt"

echo "模型大小对比完成 ✓"

# 步骤8: 生成实验报告
echo "=== 步骤8: 生成实验报告 ==="
REPORT_FILE="eval/experiment_report.txt"

cat > "$REPORT_FILE" << EOF
文物数字化多尺度高斯泼溅实验报告
====================================

实验场景: $SCENE_NAME
训练迭代: $ITERATIONS
实验时间: $(date)

实验结果:
---------
1. 模型大小对比:
   - Baseline: $BASELINE_SIZE
   - Multiscale: $MULTISCALE_SIZE

2. 质量指标:
   - 详细结果请查看: $METRICS_DIR/quality_metrics.csv

3. 效率指标:
   - 详细结果请查看: $PLOTS_DIR/efficiency_analysis.png

4. 渲染图像:
   - Baseline: $RENDER_DIR/baseline/
   - Multiscale: $RENDER_DIR/multiscale/

实验完成！
EOF

echo "实验报告已生成: $REPORT_FILE"

# 显示关键结果
echo ""
echo "=== 实验完成 ==="
echo "关键结果文件:"
echo "  - 质量指标: $METRICS_DIR/"
echo "  - 效率分析: $PLOTS_DIR/"
echo "  - 渲染图像: $RENDER_DIR/"
echo "  - 实验报告: $REPORT_FILE"

# 显示模型大小对比
echo ""
echo "模型大小对比:"
echo "  Baseline: $BASELINE_SIZE"
echo "  Multiscale: $MULTISCALE_SIZE"

echo ""
echo "实验完成！请查看上述文件进行详细分析。"
