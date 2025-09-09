#!/bin/bash
set -e
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_DIR="$ROOT/data/dtu"
mkdir -p "$DATA_DIR"
echo "DTU 数据需从官方或社区镜像下载。推荐扫描: 65 83 110 114"
echo "下载后放置为 $DATA_DIR/scanXX，内含 images/ 或 raw/"
