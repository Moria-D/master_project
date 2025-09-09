#!/bin/bash
# 在NTU GPU上训练所有场景

set -e

echo "=== 开始3DGS训练 ==="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ch-3dgs-gpu

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 训练场景列表
SCENES=("mipnerf360/garden" "mipnerf360/room" "mipnerf360/bicycle")

# 训练Baseline
echo "=== 训练Baseline模型 ==="
for scene in "${SCENES[@]}"; do
    echo "训练Baseline: $scene"
    cd ~/Projects/cultural-heritage-3dgs/gaussian-splatting
    
    python train.py \
        -s ~/Projects/cultural-heritage-3dgs/data/$scene \
        -m ~/Projects/cultural-heritage-3dgs/exp/baseline_3dgs/$scene \
        --iterations 30000 \
        --eval \
        --save_iterations 30000 \
        --test_iterations 30000 \
        --quiet
    
    echo "Baseline $scene 训练完成"
done

# 训练多尺度改进（需要你的多尺度代码）
echo "=== 训练多尺度改进模型 ==="
for scene in "${SCENES[@]}"; do
    echo "训练多尺度: $scene"
    cd ~/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs
    
    # 这里需要你的多尺度训练代码
    # python train_multiscale.py \
    #     -s ~/Projects/cultural-heritage-3dgs/data/$scene \
    #     -m ~/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs/$scene \
    #     --iterations 30000 \
    #     --use_multiscale \
    #     --density_from_grad \
    #     --lod_levels 4 \
    #     --eval \
    #     --save_iterations 30000 \
    #     --test_iterations 30000 \
    #     --quiet
    
    echo "多尺度 $scene 训练完成"
done

echo "所有训练完成！"
echo "下一步: 运行 render_all.sh 生成测试图像"
