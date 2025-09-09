#!/bin/bash
# 上传数据到NTU GPU服务器
# 使用: ./upload_to_ntu.sh <ntu_user> <ntu_host> <remote_path>

set -e

if [ $# -ne 3 ]; then
    echo "使用: $0 <ntu_user> <ntu_host> <remote_path>"
    echo "例如: $0 user gpu.ntu.edu.tw /home/user/Projects/cultural-heritage-3dgs"
    exit 1
fi

NTU_USER=$1
NTU_HOST=$2
REMOTE_PATH=$3

echo "=== 上传数据到NTU GPU服务器 ==="
echo "用户: ${NTU_USER}@${NTU_HOST}"
echo "远程路径: ${REMOTE_PATH}"

# 创建远程目录
ssh ${NTU_USER}@${NTU_HOST} "mkdir -p ${REMOTE_PATH}"

# 上传项目文件（排除大文件）
echo "上传项目脚本和配置..."
rsync -avz --progress \
    --exclude='*.zip' \
    --exclude='*.tar.gz' \
    --exclude='*.ply' \
    --exclude='*.db' \
    --exclude='LLFF/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    ~/Projects/cultural-heritage-3dgs/ \
    ${NTU_USER}@${NTU_HOST}:${REMOTE_PATH}/

# 上传训练数据
echo "上传训练数据..."
rsync -avz --progress \
    ~/Projects/cultural-heritage-3dgs/data/mipnerf360/ \
    ${NTU_USER}@${NTU_HOST}:${REMOTE_PATH}/data/mipnerf360/

echo "上传完成！"
echo "下一步: 在NTU服务器上运行 setup_ntu_env.sh"
