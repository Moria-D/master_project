#!/bin/bash
# 监控NTU服务器上的实验进度
# 使用方法: ./monitor_experiment.sh

set -e

# NTU服务器配置
NTU_USERNAME="c240116"
NTU_HOST="10.96.189.12"
NTU_PORT="22"
NTU_PATH="~/Projects/cultural-heritage-3dgs"

echo "=== 监控NTU服务器实验进度 ==="
echo "用户名: $NTU_USERNAME"
echo "服务器: $NTU_HOST:$NTU_PORT"
echo "项目路径: $NTU_PATH"
echo ""

# 检查SSH连接
echo "步骤1: 测试SSH连接..."
if ssh -p $NTU_PORT -o ConnectTimeout=10 $NTU_USERNAME@$NTU_HOST "echo 'SSH连接成功'" 2>/dev/null; then
    echo "✓ SSH连接成功"
else
    echo "✗ SSH连接失败，请检查网络和服务器状态"
    exit 1
fi

echo ""
echo "步骤2: 检查实验状态..."
echo ""

# 检查GPU使用情况
echo "GPU使用情况:"
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"

echo ""
echo "步骤3: 检查实验目录..."
echo ""

# 检查实验目录
echo "实验目录状态:"
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "ls -la $NTU_PATH/exp/ 2>/dev/null || echo 'exp目录不存在'"
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "ls -la $NTU_PATH/eval/ 2>/dev/null || echo 'eval目录不存在'"

echo ""
echo "步骤4: 检查训练日志..."
echo ""

# 检查训练日志
echo "最近的训练日志:"
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "find $NTU_PATH -name '*.log' -o -name '*.txt' | head -5 | xargs -I {} ls -la {} 2>/dev/null || echo '未找到日志文件'"

echo ""
echo "步骤5: 检查进程状态..."
echo ""

# 检查Python进程
echo "Python训练进程:"
ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST "ps aux | grep python | grep -v grep | head -5 || echo '未找到Python进程'"

echo ""
echo "=== 监控完成 ==="
echo ""
echo "实时监控命令："
echo "1. 实时GPU监控:"
echo "   ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST 'watch -n 1 nvidia-smi'"
echo ""
echo "2. 实时日志监控:"
echo "   ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST 'tail -f $NTU_PATH/eval/experiment_report.txt'"
echo ""
echo "3. 实时目录监控:"
echo "   ssh -p $NTU_PORT $NTU_USERNAME@$NTU_HOST 'watch -n 5 ls -la $NTU_PATH/exp/'"
echo ""
echo "4. 下载最新结果:"
echo "   ./download_from_ntu.sh"
echo ""
echo "监控完成！使用上述命令可以实时查看实验进度。"
