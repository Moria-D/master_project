# 实验计划与项目管理

## 目标与验收
- 近景质量：ΔPSNR ≥ −0.1 dB，ΔLPIPS ≤ +0.005
- 效率收益：FPS ≥ +20% 或 显存/模型体积 ≤ −30%
- 切换平滑：LOD切换段帧间ΔPSNR 95% ≤ 0.15 dB

## 分组设计（精简强集）
- Baseline-3DGS：官方默认参数
- Ours-LOD-Post：训练得到的点云→生成 LOD0/1/2/3（比例默认 1.0/0.6/0.36/0.18），距离分段+滞回切换
- Ablation-LOD-Ratio：比例对比 A/B/C
- Ablation-Grad-Only：仅梯度密度（τ∈{0.07,0.1,0.15,0.2}），无LOD

## 场景覆盖
- 主：mipnerf360/room（全流程与消融）
- 复核：mipnerf360/garden、bicycle（关键组）
- 文物：family 或 artifact_1（至少1组）

## 里程碑与任务

| 模块 | 任务 | 产出 | 完成标准 | 截止 |
|---|---|---|---|---|
| 环境 | CUDA+PyTorch+扩展安装 | 三模块可import | torch/cuda为True，DGR/SKNN可import | 第1周 |
| 基线 | Baseline(room) 30k | baseline点云 | 可渲染测试 | 第1周 |
| LOD | 生成 LOD0-3 | lod_*/point_cloud.ply | 比例≈1/0.6/0.36/0.18 | 第2周 |
| 评估 | room 距离分段质量/效率 | 指标与曲线 | 达到阈值 | 第2周 |
| 复核 | garden/bicycle 关键组 | 指标与图表 | 趋势一致 | 第2周 |
| 密度 | 训练期梯度密度 τ 扫描 | Ablation结果 | τ最优确定 | 第3周 |
| 文物 | 1个文物场景评估 | 指标+裁剪图 | 细节优势 | 第3-4周 |
| 论文 | 方法/实验/结果 | 草稿与图表 | 结构完整 | 第4周 |

## 实验矩阵
统一维护在 `scripts/experiments/experiment_matrix.csv`，用 `scripts/experiments/run_matrix.sh` 批量执行。

字段说明：
- scene：room/garden/bicycle/family/artifact_1
- group：Baseline | Ours-LOD-Post | Ablation-LOD-Ratio | Ablation-Grad-Only
- mode：baseline | ours
- lod_ratios：如 `1.0/0.6/0.36/0.18`
- tau：梯度阈值（仅 Grad-Only/Train 用）
- iterations：训练步数（baseline/ours训练用）
- seed：随机种子
- enabled：1 执行，0 跳过

## 执行与产出
- 执行：`bash scripts/experiments/run_matrix.sh`
- 产出：
  - 模型：`exp/{baseline_3dgs|multiscale_3dgs}/...`
  - 指标：`eval/metrics/*.csv`，图表：`eval/plots/*`
  - 渲染：`eval/renders/{baseline|multiscale}/...`


