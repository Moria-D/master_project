#!/bin/bash
set -e
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
cd "$ROOT/scripts/datasets"

echo "=== 数据集下载与组织 ==="
./download_mipnerf360.sh "$ROOT"
./download_dtu.sh "$ROOT"
./download_tanks_and_temples.sh "$ROOT"
./download_epfl_statue.sh "$ROOT"

echo "=== 目录检查 ==="
find "$ROOT/data" -maxdepth 2 -type d | sort

echo "完成：请根据提示将数据放入对应目录，然后执行 gen_transforms.sh 生成 transforms_*"
