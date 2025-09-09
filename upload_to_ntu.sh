#!/bin/bash
# 自动上传数据到NTU GPU服务器
# 使用方法: ./upload_to_ntu.sh

set -e

# NTU服务器配置
NTU_USERNAME="c240116"
NTU_HOST="10.96.189.12"
NTU_PORT="22"
NTU_PATH="~/Projects/cultural-heritage-3dgs"

echo "=== 开始上传到NTU GPU服务器 ==="
echo "用户名: $NTU_USERNAME"
echo "服务器: $NTU_HOST:$NTU_PORT"
echo "目标路径: $NTU_PATH"
echo ""

# 检查SSH连接
echo "步骤1: 测试SSH连接..."
if ssh -p $NTU_PORT -o ConnectTimeout=10 $NTU_USERNAME@$NTU_HOST "echo 'SSH连接成功'" 2>/dev/null; then
    echo "✓ SSH连接成功"
else
    echo "✗ SSH连接失败，请检查网络和服务器状态"
    exit 1
fi

# 创建远程目录
echo ""
echo "步骤2: 创建远程目录..."
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "mkdir -p $NTU_PATH"

# 上传项目代码（排除大文件和临时文件）
echo ""
echo "步骤3: 上传项目代码..."
rsync -avz -e "ssh -p $NTU_PORT" \
    --exclude='.git' \
    --exclude='data/*/colmap/dense' \
    --exclude='data/*/colmap/database.db*' \
    --exclude='exp/*' \
    --exclude='eval/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    ./ $NTU_USERNAME@$NTU_HOST:$NTU_PATH/

# 上传关键数据
echo ""
echo "步骤4: 上传实验数据..."
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "mkdir -p $NTU_PATH/data"

# 上传MipNeRF360数据集
echo "  上传MipNeRF360数据集..."
rsync -avz -e "ssh -p $NTU_PORT" \
    data/mipnerf360/ $NTU_USERNAME@$NTU_HOST:$NTU_PATH/data/mipnerf360/

echo ""
echo "=== 上传完成！ ==="
echo ""
echo "下一步操作："
echo "1. 登录NTU服务器:"
echo "   ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST"
echo ""
echo "2. 进入项目目录:"
echo "   cd $NTU_PATH"
echo ""
echo "3. 设置环境:"
echo "   chmod +x scripts/ntu_training/setup_ntu_env.sh"
echo "   ./scripts/ntu_training/setup_ntu_env.sh"
echo ""
echo "4. 执行实验:"
echo "   chmod +x scripts/ntu_training/run_complete_experiment.sh"
echo "   ./scripts/ntu_training/run_complete_experiment.sh"
echo ""
echo "5. 下载结果（在本地执行）:"
echo "   ./download_from_ntu.sh"
echo ""
echo "上传完成！现在可以登录NTU服务器开始实验了。"
