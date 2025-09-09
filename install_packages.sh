#!/bin/bash

echo "=== 安装Python包 ==="

# 尝试使用conda安装
echo "尝试使用conda安装..."
conda install -y numpy scipy scikit-image opencv matplotlib seaborn pandas tqdm || {
    echo "conda安装失败，尝试使用pip..."
    
    # 使用pip安装
    pip3 install numpy scipy scikit-image opencv-python matplotlib seaborn pandas tqdm
    
    # 尝试安装PyTorch (CPU版本，适合M4芯片)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 安装LPIPS
    pip3 install lpips
}

echo "Python包安装完成!"
echo "运行 ./check_environment.sh 验证安装"
