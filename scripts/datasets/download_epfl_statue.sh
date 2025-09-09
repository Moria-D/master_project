#!/bin/bash
set -e
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_DIR="$ROOT/data/epfl_statue"
mkdir -p "$DATA_DIR"
echo "请下载 EPFL Fountain-P11 图像到: $DATA_DIR/Fountain-P11/images"
