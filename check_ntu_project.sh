#!/bin/bash

# NTU 计算节点项目完整性检查脚本
echo "=== NTU 计算节点项目检查 ==="
echo "时间: $(date)"
echo "用户: $(whoami)"
echo "主机名: $(hostname)"

# 检查项目目录
PROJECT_DIR="$HOME/Projects/cultural-heritage-3dgs"
echo "项目目录: $PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ 项目目录不存在: $PROJECT_DIR"
    echo "请先上传项目文件到此目录"
    exit 1
fi

echo "✅ 项目目录存在"

# 检查关键文件和目录
echo ""
echo "=== 检查关键文件和目录 ==="

# 检查脚本文件
scripts=(
    "run_ntu_experiment.sh"
    "quick_test.sh" 
    "check_ntu_env.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$PROJECT_DIR/$script" ]; then
        echo "✅ $script"
    else
        echo "❌ 缺少 $script"
    fi
done

# 检查数据目录
echo ""
echo "=== 检查数据目录 ==="
if [ -d "$PROJECT_DIR/data/mipnerf360" ]; then
    echo "✅ mipnerf360 数据目录存在"
    scenes=$(ls "$PROJECT_DIR/data/mipnerf360" 2>/dev/null | wc -l)
    echo "   场景数量: $scenes"
    
    # 检查具体场景
    for scene in room bicycle garden; do
        if [ -d "$PROJECT_DIR/data/mipnerf360/$scene" ]; then
            echo "   ✅ $scene 场景存在"
            if [ -f "$PROJECT_DIR/data/mipnerf360/$scene/transforms_train.json" ]; then
                echo "      ✅ transforms_train.json"
            else
                echo "      ❌ 缺少 transforms_train.json"
            fi
        else
            echo "   ❌ $scene 场景不存在"
        fi
    done
else
    echo "❌ mipnerf360 数据目录不存在"
fi

# 检查 gaussian-splatting 目录
echo ""
echo "=== 检查 gaussian-splatting ==="
if [ -d "$PROJECT_DIR/gaussian-splatting" ]; then
    echo "✅ gaussian-splatting 目录存在"
    
    # 检查关键文件
    key_files=(
        "train.py"
        "render.py"
        "scene/gaussian_model.py"
        "scene/cameras.py"
        "utils/general_utils.py"
    )
    
    for file in "${key_files[@]}"; do
        if [ -f "$PROJECT_DIR/gaussian-splatting/$file" ]; then
            echo "   ✅ $file"
        else
            echo "   ❌ 缺少 $file"
        fi
    done
    
    # 检查 submodules
    if [ -d "$PROJECT_DIR/gaussian-splatting/submodules" ]; then
        echo "   ✅ submodules 目录存在"
        for submod in diff-gaussian-rasterization simple-knn; do
            if [ -d "$PROJECT_DIR/gaussian-splatting/submodules/$submod" ]; then
                echo "      ✅ $submod"
            else
                echo "      ❌ 缺少 $submod"
            fi
        done
    else
        echo "   ❌ submodules 目录不存在"
    fi
else
    echo "❌ gaussian-splatting 目录不存在"
fi

# 检查兼容性模块
echo ""
echo "=== 检查兼容性模块 ==="
if [ -f "$PROJECT_DIR/gaussian-splatting/diff_gaussian_rasterization/__init__.py" ]; then
    echo "✅ CPU 兼容模块存在"
else
    echo "❌ CPU 兼容模块不存在"
fi

# 检查 Python 环境
echo ""
echo "=== 检查 Python 环境 ==="
if command -v python3 &> /dev/null; then
    echo "✅ python3 可用: $(python3 --version)"
else
    echo "❌ python3 不可用"
fi

# 检查 CUDA 环境
echo ""
echo "=== 检查 CUDA 环境 ==="
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
elif command -v nvcc &> /dev/null; then
    echo "✅ CUDA 工具链可用"
    nvcc --version | grep "release"
else
    echo "⚠️  CUDA 环境未配置"
fi

echo ""
echo "=== 总结 ==="
echo "如果所有检查都显示 ✅，则项目已准备就绪"
echo "如果有 ❌，请补充相应的文件或依赖"
echo ""
echo "运行实验命令："
echo "  cd $PROJECT_DIR"
echo "  ./check_ntu_env.sh    # 环境检查"
echo "  ./quick_test.sh room 0    # 快速测试"
echo "  ./run_ntu_experiment.sh room 0    # 完整实验"
