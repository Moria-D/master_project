#!/bin/bash
set -e
ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_DIR="$ROOT/data/llff"

echo "=== 下载LLFF数据集 ==="
echo "数据目录: $DATA_DIR"
mkdir -p "$DATA_DIR"

# 推荐场景
SCENES=("fern" "horns" "trex" "room")

echo "推荐场景: ${SCENES[*]}"
echo ""
echo "请从以下地址下载LLFF数据:"
echo "https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
echo ""
echo "下载后解压到: $DATA_DIR"
echo "每个场景应该包含: images/, poses_bounds.npy"
echo ""
echo "或者使用git克隆:"
echo "git clone https://github.com/Fyusion/LLFF.git"
echo "cp -r LLFF/data/* $DATA_DIR/"
