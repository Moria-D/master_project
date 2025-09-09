#!/usr/bin/env python3
"""
多尺度高斯泼溅训练脚本（可运行版）

策略：
1) 直接调用官方3DGS的 train.py 在目标目录训练得到最终点云；
2) 基于训练得到的 point_cloud.ply 生成 4 个LOD下采样版本；
3) 将LOD结果写入 ${MODEL_DIR}/point_cloud/lod_{level}/point_cloud.ply。

注意：本脚本不依赖CUDA扩展在本机可用，建议在GPU服务器运行。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except Exception:
    PlyData = None
    PlyElement = None


def run_3dgs_training(gs_root: Path, data_dir: Path, model_dir: Path, iterations: int, save_iterations: int, test_iterations: int, extra_args: List[str]) -> None:
    train_py = gs_root / "train.py"
    cmd = [sys.executable, str(train_py), "-s", str(data_dir), "-m", str(model_dir), "--iterations", str(iterations), "--save_iterations", str(save_iterations), "--test_iterations", str(test_iterations), "--quiet"]
    if extra_args:
        cmd.extend(extra_args)
    print("运行3DGS训练:", " ".join(cmd))
    subprocess.check_call(cmd)


def find_latest_ply(model_dir: Path) -> Path:
    pc_dir = model_dir / "point_cloud"
    if not pc_dir.exists():
        raise FileNotFoundError(f"未找到point_cloud目录: {pc_dir}")
    candidates = sorted(pc_dir.glob("iteration_*/point_cloud.ply"), key=lambda p: int(p.parent.name.split("_")[-1]))
    if not candidates:
        raise FileNotFoundError("未找到任何迭代的point_cloud.ply")
    return candidates[-1]


def downsample_ply(src_ply: Path, dst_ply: Path, keep_ratio: float) -> None:
    if PlyData is None:
        raise RuntimeError("缺少 plyfile 依赖，请先 pip install plyfile")
    src = PlyData.read(str(src_ply))
    vert = src["vertex"]

    num_points = vert.count
    keep_num = max(1, int(num_points * keep_ratio))
    indices = np.random.RandomState(42).permutation(num_points)[:keep_num]

    new_verts = vert.data[indices]
    new_el = PlyElement.describe(new_verts, "vertex")
    PlyData([new_el], text=False).write(str(dst_ply))


def generate_lods(model_dir: Path, base_ply: Path, ratios: List[float]) -> None:
    for level, r in enumerate(ratios):
        lod_dir = model_dir / "point_cloud" / f"lod_{level}"
        lod_dir.mkdir(parents=True, exist_ok=True)
        dst = lod_dir / "point_cloud.ply"
        if level == 0:
            # LOD0 直接拷贝为完整分辨率
            if dst.resolve() != base_ply.resolve():
                dst.write_bytes(base_ply.read_bytes())
        else:
            downsample_ply(base_ply, dst, r)
        print(f"生成 LOD{level} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多尺度训练（包装3DGS并生成LOD）")
    parser.add_argument("-s", "--source", required=True, help="数据目录")
    parser.add_argument("-m", "--model", required=True, help="模型输出目录")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--save_iterations", type=int, default=30000)
    parser.add_argument("--test_iterations", type=int, default=30000)
    parser.add_argument("--lod_levels", type=int, default=4)
    parser.add_argument("--lod_ratios", type=str, default="1.0,0.5,0.25,0.125", help="各层级保留比例, 逗号分隔")
    parser.add_argument("--skip_training", action="store_true", help="跳过3DGS训练，仅生成LOD")
    parser.add_argument("--base_ply", type=str, default=None, help="当跳过训练时，指定已有的point_cloud.ply路径")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="透传给3DGS的其它参数")
    args = parser.parse_args()

    gs_root = (Path(__file__).resolve().parents[2] / "gaussian-splatting").resolve()
    data_dir = Path(args.source).resolve()
    model_dir = Path(args.model).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    lod_ratios = [float(x) for x in args.lod_ratios.split(",")]
    if len(lod_ratios) != args.lod_levels:
        print("警告: lod_levels 与 lod_ratios 数量不一致，按lod_ratios长度为准")

    # 1) 运行官方3DGS训练（可跳过）
    if not args.skip_training:
        run_3dgs_training(gs_root, data_dir, model_dir, args.iterations, args.save_iterations, args.test_iterations, args.extra or [])

    # 2) 生成LOD
    if args.skip_training:
        if not args.base_ply:
            raise ValueError("skip_training 模式需要提供 --base_ply")
        base_ply = Path(args.base_ply).resolve()
        if not base_ply.exists():
            raise FileNotFoundError(f"指定的 base_ply 不存在: {base_ply}")
    else:
        base_ply = find_latest_ply(model_dir)
    generate_lods(model_dir, base_ply, lod_ratios)

    print("多尺度训练完成，LOD已生成。")


if __name__ == "__main__":
    main()
