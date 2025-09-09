#!/bin/bash
set -e
# 下载入口需根据官方/镜像更新；此处给出占位与结构
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_DIR="$ROOT/data/mipnerf360"
mkdir -p "$DATA_DIR"
echo "请参考官方链接下载 Mip-NeRF 360 到: $DATA_DIR"
echo "推荐场景: garden, bicycle, room"
echo "下载完成后，保持每个场景为子目录，包含 images/ 与相机文件"
