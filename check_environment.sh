#!/bin/bash

echo "=== 环境检查 ==="

# 检查基础工具
echo "检查基础工具..."
tools=("colmap" "ffmpeg" "git" "python3" "pip3")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "✓ $tool 已安装"
    else
        echo "✗ $tool 未安装"
    fi
done

# 检查Python包
echo -e "\n检查Python包..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch 未安装"
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" 2>/dev/null || echo "✗ OpenCV 未安装"
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "✗ NumPy 未安装"
python3 -c "import lpips; print('✓ LPIPS')" 2>/dev/null || echo "✗ LPIPS 未安装"

# 检查目录结构
echo -e "\n检查项目结构..."
dirs=("data" "exp" "eval" "scripts")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir 目录存在"
    else
        echo "✗ $dir 目录不存在"
    fi
done

echo -e "\n环境检查完成!"
