#!/bin/bash

set -euo pipefail

MATRIX_FILE="$(dirname "$0")/experiment_matrix.csv"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="$PROJECT_ROOT/data"
EXP_BASELINE="$PROJECT_ROOT/exp/baseline_3dgs"
EXP_MULTI="$PROJECT_ROOT/exp/multiscale_3dgs"

if [ ! -f "$MATRIX_FILE" ]; then
    echo "找不到矩阵文件: $MATRIX_FILE"
    exit 1
fi

echo "读取矩阵: $MATRIX_FILE"
# 跳过表头
TAIL_CONTENT=$(tail -n +2 "$MATRIX_FILE")

IFS=$'\n'
for line in $TAIL_CONTENT; do
    # CSV: scene,group,mode,lod_ratios,tau,iterations,seed,enabled
    scene=$(echo "$line" | awk -F, '{print $1}')
    group=$(echo "$line" | awk -F, '{print $2}')
    mode=$(echo "$line" | awk -F, '{print $3}')
    lod_ratios=$(echo "$line" | awk -F, '{print $4}')
    tau=$(echo "$line" | awk -F, '{print $5}')
    iterations=$(echo "$line" | awk -F, '{print $6}')
    seed=$(echo "$line" | awk -F, '{print $7}')
    enabled=$(echo "$line" | awk -F, '{print $8}')

    [ "$enabled" != "1" ] && continue

    data_dir="$DATA_ROOT/$scene"
    if [ "$mode" = "baseline" ]; then
        out_dir="$EXP_BASELINE/$scene"
        echo "[Baseline] $scene -> $out_dir (iters=$iterations)"
        echo "占位: python gaussian-splatting/train.py -s $data_dir -m $out_dir --iterations $iterations --disable_viewer --quiet"
    else
        out_dir="$EXP_MULTI/$scene"
        ratios=${lod_ratios:-1.0/0.6/0.36/0.18}
        echo "[Ours] $group $scene -> $out_dir (iters=$iterations, ratios=$ratios, tau=$tau)"
        echo "占位: python scripts/training/train_multiscale.py -s $data_dir -m $out_dir --iterations $iterations --lod_levels 4 --lod_ratios $ratios --extra --disable_viewer --quiet"
    fi
    echo "---"

done
