#!/bin/bash
# 生成 transforms_train.json / transforms_test.json 的一键脚本
# 优先使用 nerfstudio 的 ns-process-data；若未安装则给出COLMAP到transforms的占位提示
set -e

ROOT=${1:-$HOME/Projects/cultural-heritage-3dgs}
DATA_PATH=$2   # 例如: $ROOT/data/mipnerf360/garden
SPLIT=${3:-9} # 训练:测试 = 9:1

if [ -z "$DATA_PATH" ]; then
  echo "使用: $0 <root_dir(可省)> <scene_data_path> [train_split]"
  exit 1
fi

if command -v ns-process-data >/dev/null 2>&1; then
  echo "检测到 nerfstudio，使用 ns-process-data 进行转换..."
  ns-process-data colmap --data "$DATA_PATH" --output-dir "$DATA_PATH" --skip-colmap || true
  echo "如需拆分，请将输出中的 transforms.json 拆分为 transforms_train/test.json（按${SPLIT}:1划分）。"
else
  echo "未检测到 nerfstudio。请按以下方式准备 transforms_* 文件："
  echo "1) 保持 COLMAP 结果在 $DATA_PATH/colmap/{sparse,dense} 下"
  echo "2) 使用你的转换脚本将相机位姿与图像名写入 transforms_train.json / transforms_test.json"
  echo "3) 拆分比例建议为 ${SPLIT}:1，测试视角覆盖细节与远近尺度"
fi
