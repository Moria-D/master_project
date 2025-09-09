#!/bin/bash
# 结果下载脚本 - 从NTU服务器下载实验结果

set -e

# NTU服务器配置
NTU_USERNAME="c240116"
NTU_HOST="10.96.189.12"
NTU_PORT="22"
NTU_PATH="~/Projects/cultural-heritage-3dgs"

echo "=== 从NTU服务器下载实验结果 ==="
echo "服务器: ${NTU_USERNAME}@${NTU_HOST}:${NTU_PORT}"
echo "源路径: ${NTU_PATH}"

# 检查连接
echo "检查服务器连接..."
ssh -p ${NTU_PORT} ${NTU_USERNAME}@${NTU_HOST} "echo '连接成功!'" || {
    echo "错误: 无法连接到NTU服务器"
    exit 1
}

# 创建本地目录
echo "创建本地目录..."
mkdir -p eval exp

# 下载实验结果
echo "下载实验结果..."
rsync -avz --progress -e "ssh -p ${NTU_PORT}" \
    ${NTU_USERNAME}@${NTU_HOST}:${NTU_PATH}/eval/ ./eval/

# 下载训练模型
echo "下载训练模型..."
rsync -avz --progress -e "ssh -p ${NTU_PORT}" \
    ${NTU_USERNAME}@${NTU_HOST}:${NTU_PATH}/exp/ ./exp/

echo "=== 下载完成 ==="
echo ""
echo "下载的内容:"
echo "- 实验结果: ./eval/"
echo "- 训练模型: ./exp/"
echo ""
echo "下一步操作:"
echo "1. 查看实验结果:"
echo "   ls -la eval/"
echo ""
echo "2. 运行结果分析:"
echo "   python scripts/evaluation/analyze_results.py --results_dir eval"
echo ""
echo "3. 查看生成的图表:"
echo "   open eval/plots/"
