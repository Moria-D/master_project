#!/bin/bash
# 从NTU GPU服务器下载实验结果
# 使用方法: ./download_from_ntu.sh

set -e

# NTU服务器配置
NTU_USERNAME="c240116"
NTU_HOST="10.96.189.12"
NTU_PORT="22"
NTU_PATH="~/Projects/cultural-heritage-3dgs"

echo "=== 从NTU服务器下载实验结果 ==="
echo "用户名: $NTU_USERNAME"
echo "服务器: $NTU_HOST:$NTU_PORT"
echo "源路径: $NTU_PATH"
echo ""

# 检查SSH连接
echo "步骤1: 测试SSH连接..."
if ssh -p $NTU_PORT -o ConnectTimeout=10 $NTU_USERNAME@$NTU_HOST "echo 'SSH连接成功'" 2>/dev/null; then
    echo "✓ SSH连接成功"
else
    echo "✗ SSH连接失败，请检查网络和服务器状态"
    exit 1
fi

# 创建本地目录
echo ""
echo "步骤2: 创建本地目录..."
mkdir -p exp eval

# 下载实验结果
echo ""
echo "步骤3: 下载实验结果..."
echo "  下载训练模型..."
rsync -avz -e "ssh -p $NTU_PORT" \
    $NTU_USERNAME@$NTU_HOST:$NTU_PATH/exp/ ./exp/

echo "  下载评估结果..."
rsync -avz -e "ssh -p $NTU_PORT" \
    $NTU_USERNAME@$NTU_HOST:$NTU_PATH/eval/ ./eval/

# 下载实验报告
echo ""
echo "步骤4: 下载实验报告..."
if ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "test -f $NTU_PATH/eval/experiment_report.txt"; then
    scp -P $NTU_PORT $NTU_USERNAME@$NTU_HOST:$NTU_PATH/eval/experiment_report.txt ./eval/
    echo "✓ 实验报告下载完成"
else
    echo "⚠ 实验报告尚未生成，可能实验还在进行中"
fi

# 检查下载结果
echo ""
echo "步骤5: 检查下载结果..."
echo "本地目录结构："
echo "  exp/ - 训练模型"
echo "  eval/ - 评估结果"
echo ""

if [ -d "exp/baseline_3dgs" ] && [ -d "exp/multiscale_3dgs" ]; then
    echo "✓ 训练模型下载完成"
else
    echo "⚠ 训练模型可能还在进行中"
fi

if [ -d "eval/renders" ] && [ -d "eval/metrics" ]; then
    echo "✓ 评估结果下载完成"
else
    echo "⚠ 评估结果可能还在进行中"
fi

echo ""
echo "=== 下载完成！ ==="
echo ""
echo "下一步操作："
echo "1. 查看实验结果:"
echo "   ls -la eval/"
echo "   ls -la exp/"
echo ""
echo "2. 运行结果分析:"
echo "   python scripts/evaluation/analyze_results.py --results_dir eval"
echo ""
echo "3. 查看生成的图表:"
echo "   open eval/plots/"
echo ""
echo "4. 查看实验报告:"
echo "   cat eval/experiment_report.txt"
echo ""
echo "下载完成！现在可以在本地分析实验结果了。"
