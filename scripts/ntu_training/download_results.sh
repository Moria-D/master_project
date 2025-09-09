#!/bin/bash
# 从NTU下载训练结果

set -e

if [ $# -ne 3 ]; then
    echo "使用: $0 <ntu_user> <ntu_host> <remote_path>"
    exit 1
fi

NTU_USER=$1
NTU_HOST=$2
REMOTE_PATH=$3

echo "=== 下载训练结果 ==="

# 下载实验结果
rsync -avz --progress \
    ${NTU_USER}@${NTU_HOST}:${REMOTE_PATH}/exp/ \
    ~/Projects/cultural-heritage-3dgs/exp/

# 下载渲染结果
rsync -avz --progress \
    ${NTU_USER}@${NTU_HOST}:${REMOTE_PATH}/eval/renders/ \
    ~/Projects/cultural-heritage-3dgs/eval/renders/

echo "下载完成！"
echo "下一步: 运行评估脚本"
