#!/usr/bin/env python3
"""
将 Mip-NeRF 360 的 poses_bounds.npy 转换为 transforms_train.json 和 transforms_test.json
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

def load_poses_bounds(poses_bounds_path):
    """加载 poses_bounds.npy 文件"""
    poses_bounds = np.load(poses_bounds_path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # N, 3, 5
    bounds = poses_bounds[:, 15:17]  # N, 2
    return poses, bounds

def poses_to_transforms(poses, bounds, image_dir, split_ratio=0.9):
    """将 poses 转换为 transforms 格式"""
    transforms = {
        "camera_angle_x": 0.6911112070083618,  # 默认值，可根据实际相机参数调整
        "frames": []
    }
    
    # 获取图像文件列表
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_files) != len(poses):
        print(f"警告: 图像数量 ({len(image_files)}) 与位姿数量 ({len(poses)}) 不匹配")
        return None
    
    # 创建帧数据
    for i, (pose, bound, image_file) in enumerate(zip(poses, bounds, image_files)):
        # 提取旋转矩阵和平移向量
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # 转换为 4x4 变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t
        
        frame = {
            "file_path": f"./images/{image_file}",
            "transform_matrix": transform_matrix.tolist(),
            "near": float(bound[0]),
            "far": float(bound[1])
        }
        transforms["frames"].append(frame)
    
    return transforms

def split_transforms(transforms, split_ratio=0.9):
    """按比例分割训练和测试数据"""
    n_frames = len(transforms["frames"])
    n_train = int(n_frames * split_ratio)
    
    train_transforms = {
        "camera_angle_x": transforms["camera_angle_x"],
        "frames": transforms["frames"][:n_train]
    }
    
    test_transforms = {
        "camera_angle_x": transforms["camera_angle_x"],
        "frames": transforms["frames"][n_train:]
    }
    
    return train_transforms, test_transforms

def main():
    if len(sys.argv) != 2:
        print("使用: python convert_mipnerf360.py <scene_path>")
        print("例如: python convert_mipnerf360.py /path/to/mipnerf360/garden")
        sys.exit(1)
    
    scene_path = Path(sys.argv[1])
    poses_bounds_path = scene_path / "poses_bounds.npy"
    image_dir = scene_path / "images"
    
    if not poses_bounds_path.exists():
        print(f"错误: poses_bounds.npy 不存在: {poses_bounds_path}")
        sys.exit(1)
    
    if not image_dir.exists():
        print(f"错误: images 目录不存在: {image_dir}")
        sys.exit(1)
    
    print(f"处理场景: {scene_path}")
    
    # 加载数据
    poses, bounds = load_poses_bounds(poses_bounds_path)
    print(f"加载了 {len(poses)} 个相机位姿")
    
    # 转换为 transforms 格式
    transforms = poses_to_transforms(poses, bounds, image_dir)
    if transforms is None:
        sys.exit(1)
    
    # 分割训练和测试
    train_transforms, test_transforms = split_transforms(transforms)
    print(f"训练帧: {len(train_transforms['frames'])}, 测试帧: {len(test_transforms['frames'])}")
    
    # 保存文件
    train_path = scene_path / "transforms_train.json"
    test_path = scene_path / "transforms_test.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    print(f"保存完成:")
    print(f"  训练: {train_path}")
    print(f"  测试: {test_path}")

if __name__ == "__main__":
    main()
