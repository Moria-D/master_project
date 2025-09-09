#!/bin/bash
# 使用: ./run_colmap.sh <artifact_name> <image_dir>
set -e
if [ $# -ne 2 ]; then echo "使用: $0 <artifact_name> <image_dir>"; exit 1; fi
ARTIFACT_NAME=$1
IMAGE_DIR=$2
ROOT_DIR=$(cd "$(dirname "$0")"/../.. && pwd)
DATA_DIR="${ROOT_DIR}/data/${ARTIFACT_NAME}"
COLMAP_DIR="${DATA_DIR}/colmap"
mkdir -p "${COLMAP_DIR}/sparse" "${COLMAP_DIR}/dense" "${DATA_DIR}"
if [ ! -d "${IMAGE_DIR}" ]; then echo "错误: 图像目录不存在: ${IMAGE_DIR}"; exit 1; fi
rm -rf "${DATA_DIR}/images"; cp -r "${IMAGE_DIR}" "${DATA_DIR}/images"
DB_PATH="${COLMAP_DIR}/database.db"
echo "[1/6] 特征提取"
colmap feature_extractor --database_path "${DB_PATH}" --image_path "${DATA_DIR}/images" --ImageReader.single_camera 0 | cat
echo "[2/6] 特征匹配"
colmap exhaustive_matcher --database_path "${DB_PATH}" | cat
echo "[3/6] 稀疏重建"
colmap mapper --database_path "${DB_PATH}" --image_path "${DATA_DIR}/images" --output_path "${COLMAP_DIR}/sparse" | cat
echo "[4/6] 去畸变"
colmap image_undistorter --image_path "${DATA_DIR}/images" --input_path "${COLMAP_DIR}/sparse/0" --output_path "${COLMAP_DIR}/dense" --output_type COLMAP | cat
echo "[5/6] 立体匹配"
colmap patch_match_stereo --workspace_path "${COLMAP_DIR}/dense" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true | cat
echo "[6/6] 点云融合"
colmap stereo_fusion --workspace_path "${COLMAP_DIR}/dense" --workspace_format COLMAP --input_type geometric --output_path "${COLMAP_DIR}/dense/fused.ply" | cat
echo "完成。结果目录: ${COLMAP_DIR}"
