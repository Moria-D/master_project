#!/bin/bash
# 渲染所有测试图像

set -e

echo "=== 渲染测试图像 ==="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ch-3dgs-gpu

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 场景列表
SCENES=("mipnerf360/garden" "mipnerf360/room" "mipnerf360/bicycle")

# 渲染Baseline
echo "=== 渲染Baseline ==="
for scene in "${SCENES[@]}"; do
    echo "渲染Baseline: $scene"
    cd ~/Projects/cultural-heritage-3dgs/gaussian-splatting
    
    python render.py \
        -m ~/Projects/cultural-heritage-3dgs/exp/baseline_3dgs/$scene/point_cloud/iteration_30000 \
        --test_transforms ~/Projects/cultural-heritage-3dgs/data/$scene/transforms_test.json \
        --outdir ~/Projects/cultural-heritage-3dgs/eval/renders/baseline/$scene
    
    echo "Baseline $scene 渲染完成"
done

# 渲染多尺度改进
echo "=== 渲染多尺度改进 ==="
for scene in "${SCENES[@]}"; do
    echo "渲染多尺度: $scene"
    cd ~/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs
    
    # 这里需要你的多尺度渲染代码
    # python render_multiscale.py \
    #     -m ~/Projects/cultural-heritage-3dgs/exp/multiscale_3dgs/$scene/iteration_30000 \
    #     --test_transforms ~/Projects/cultural-heritage-3dgs/data/$scene/transforms_test.json \
    #     --outdir ~/Projects/cultural-heritage-3dgs/eval/renders/multiscale/$scene
    
    echo "多尺度 $scene 渲染完成"
done

echo "所有渲染完成！"
echo "下一步: 下载结果到本地进行评估"
