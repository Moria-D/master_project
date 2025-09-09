# 文物数字化多尺度高斯泼溅实验

本项目实现了基于多尺度高斯泼溅的文物数字化方法，包含完整的实验流程和评估体系。

## 项目结构

```
cultural-heritage-3dgs/
├── data/                    # 数据目录
│   ├── artifact_1/         # 文物1数据
│   │   ├── images/         # 原始图像
│   │   ├── colmap/         # COLMAP重建结果
│   │   ├── transforms_train.json  # 训练数据
│   │   └── transforms_test.json   # 测试数据
│   └── artifact_2/         # 文物2数据
├── exp/                     # 实验结果
│   ├── baseline_3dgs/      # Baseline模型结果
│   └── multiscale_3dgs/    # 多尺度模型结果
├── eval/                    # 评估结果
│   ├── renders/            # 渲染图像
│   │   ├── baseline/       # Baseline渲染结果
│   │   └── multiscale/     # 多尺度渲染结果
│   ├── metrics/            # 评估指标
│   └── plots/              # 分析图表
└── scripts/                # 脚本文件
    ├── colmap/             # COLMAP重建脚本
    ├── training/           # 训练脚本
    └── evaluation/         # 评估脚本
```

## 实验流程

### 1. 数据准备
- 拍摄文物多角度图像（建议≥50张）
- 确保光照稳定，避免反光
- 覆盖上下左右和细节特写

### 2. COLMAP重建
```bash
cd scripts/colmap
./run_colmap.sh artifact_1 /path/to/images
```

### 3. 生成训练数据
使用nerfstudio或其他工具将COLMAP结果转换为transforms.json：
```bash
ns-process-data colmap --data data/artifact_1 --output-dir data/artifact_1
```

### 4. 训练模型
在GPU服务器上执行：

**Baseline模型：**
```bash
cd scripts/training
./train_baseline.sh artifact_1 0
```

**多尺度模型：**
```bash
cd scripts/training
./train_multiscale.sh artifact_1 0
```

### 5. 渲染测试
在GPU服务器上渲染测试图像：
```bash
# Baseline渲染
python render.py -m exp/baseline_3dgs/artifact_1 \
  --test_transforms data/artifact_1/transforms_test.json \
  --outdir eval/renders/baseline

# 多尺度渲染
python render.py -m exp/multiscale_3dgs/artifact_1 \
  --test_transforms data/artifact_1/transforms_test.json \
  --outdir eval/renders/multiscale
```

### 6. 评估结果
```bash
# 质量评估
python scripts/evaluation/evaluate_quality.py \
  --gt_dir data/artifact_1/images_test \
  --baseline_dir eval/renders/baseline \
  --multiscale_dir eval/renders/multiscale \
  --output_dir eval/metrics

# 效率评估
python scripts/evaluation/evaluate_efficiency.py \
  --baseline_csv eval/metrics/baseline_fps.csv \
  --multiscale_csv eval/metrics/multiscale_fps.csv \
  --output_dir eval/plots
```

## 一键运行
```bash
./run_experiments.sh artifact_1 /path/to/images
```

## 评估指标

### 重建质量
- **PSNR**: 峰值信噪比，衡量像素级差异
- **SSIM**: 结构相似性指数，更符合人眼感知
- **LPIPS**: 学习感知图像块相似性，深度学习指标

### 渲染效率
- **FPS**: 帧率，实时渲染性能
- **LOD切换**: 不同距离下的渲染质量
- **内存占用**: 模型大小和显存使用

### 模型大小
- 总文件大小对比
- 不同LOD层级大小分布

## 环境要求

### 本地环境 (Mac M4)
- COLMAP 3.11+
- Python 3.10+
- PyTorch (CPU/MPS版本)
- OpenCV, NumPy, SciPy
- LPIPS, scikit-image

### GPU环境 (NTU服务器)
- CUDA 12.1+
- PyTorch (CUDA版本)
- 3DGS训练框架

## 注意事项

1. **网络问题**: 如果conda安装失败，可以使用pip安装Python包
2. **GPU训练**: 主要训练工作在GPU服务器完成，本地主要用于数据准备和评估
3. **数据备份**: 重要数据请及时备份
4. **实验记录**: 记录所有实验参数和结果

## 常见问题

### COLMAP重建失败
- 检查图像质量和数量
- 调整COLMAP参数
- 增加图像间的重叠度

### 训练不收敛
- 检查数据质量
- 调整学习率
- 增加训练步数

### 渲染质量差
- 检查相机位姿精度
- 调整高斯点数量
- 优化LOD策略
