#!/bin/bash
set -e
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_DIR="$ROOT/data/tanks_and_temples"
mkdir -p "$DATA_DIR"
echo "请下载 Tanks and Temples (family) 图像到: $DATA_DIR/family/images"
