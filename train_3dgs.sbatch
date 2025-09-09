#!/bin/bash
#SBATCH --job-name=3dgs-train
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

module load cuda/12.8.0

eval "$(conda shell.bash hook)"
conda activate ch-3dgs-gpu

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

# 编译CUDA扩展（首次或环境变动时）
cd $HOME/Projects/cultural-heritage-3dgs/gaussian-splatting
pip install --upgrade pip setuptools wheel ninja cmake
pip install -e .

# 简单检查扩展是否可用
python - << 'PY'
import importlib, sys
assert importlib.util.find_spec("diff_gaussian_rasterization"), "Missing diff_gaussian_rasterization"
assert importlib.util.find_spec("simple_knn"), "Missing simple_knn"
print("Extensions ready.")
PY

# 启动完整实验（脚本已默认 mipnerf360/room）
cd $HOME/Projects/cultural-heritage-3dgs
chmod +x scripts/ntu_training/run_complete_experiment.sh
./scripts/ntu_training/run_complete_experiment.sh
