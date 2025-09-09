#!/usr/bin/env python3
"""
文物数据集处理脚本
将文物图像转换为3DGS训练格式
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import shutil

def create_transforms_json(image_dir: str, colmap_dir: str, output_dir: str, 
                          train_ratio: float = 0.8) -> None:
    """创建transforms.json文件"""
    
    # 读取COLMAP结果
    cameras_file = os.path.join(colmap_dir, "sparse", "0", "cameras.bin")
    images_file = os.path.join(colmap_dir, "sparse", "0", "images.bin")
    points_file = os.path.join(colmap_dir, "sparse", "0", "points3D.bin")
    
    if not all(os.path.exists(f) for f in [cameras_file, images_file, points_file]):
        print("COLMAP文件不存在，请先运行COLMAP重建")
        return
    
    # 读取相机参数
    from colmap_read_model import read_cameras_binary, read_images_binary
    
    cameras = read_cameras_binary(cameras_file)
    images = read_images_binary(images_file)
    
    # 获取图像列表
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # 分割训练和测试集
    num_train = int(len(image_files) * train_ratio)
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]
    
    # 创建transforms_train.json
    transforms_train = {
        "camera_angle_x": 0.8575560541152954,  # 默认值，需要根据实际相机调整
        "frames": []
    }
    
    transforms_test = {
        "camera_angle_x": 0.8575560541152954,
        "frames": []
    }
    
    # 处理训练图像
    for i, img_file in enumerate(train_files):
        img_path = os.path.join(image_dir, img_file)
        
        # 读取图像尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 查找对应的COLMAP数据
        img_name = img_file
        if img_name in images:
            img_data = images[img_name]
            # 提取相机姿态
            qvec = img_data.qvec
            tvec = img_data.tvec
            
            # 转换为变换矩阵
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = tvec
            
        else:
            # 如果没有COLMAP数据，使用默认变换
            transform_matrix = np.eye(4)
        
        frame = {
            "file_path": f"./images/{img_file}",
            "transform_matrix": transform_matrix.tolist()
        }
        transforms_train["frames"].append(frame)
    
    # 处理测试图像
    for img_file in test_files:
        img_path = os.path.join(image_dir, img_file)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 查找对应的COLMAP数据
        img_name = img_file
        if img_name in images:
            img_data = images[img_name]
            qvec = img_data.qvec
            tvec = img_data.tvec
            
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = tvec
            
        else:
            transform_matrix = np.eye(4)
        
        frame = {
            "file_path": f"./images/{img_file}",
            "transform_matrix": transform_matrix.tolist()
        }
        transforms_test["frames"].append(frame)
    
    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "transforms_train.json"), "w") as f:
        json.dump(transforms_train, f, indent=2)
    
    with open(os.path.join(output_dir, "transforms_test.json"), "w") as f:
        json.dump(transforms_test, f, indent=2)
    
    # 复制图像文件
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for img_file in image_files:
        src = os.path.join(image_dir, img_file)
        dst = os.path.join(images_dir, img_file)
        shutil.copy2(src, dst)
    
    print(f"处理完成！")
    print(f"训练图像: {len(transforms_train['frames'])} 张")
    print(f"测试图像: {len(transforms_test['frames'])} 张")
    print(f"输出目录: {output_dir}")

def process_artifact_dataset(artifact_name: str, image_dir: str, 
                           colmap_dir: str = None) -> None:
    """处理文物数据集"""
    
    output_dir = f"data/{artifact_name}"
    
    if colmap_dir is None:
        colmap_dir = f"data/{artifact_name}/colmap"
    
    # 检查COLMAP结果是否存在
    if not os.path.exists(colmap_dir):
        print(f"COLMAP目录不存在: {colmap_dir}")
        print("请先运行COLMAP重建")
        return
    
    # 创建transforms.json
    create_transforms_json(image_dir, colmap_dir, output_dir)
    
    print(f"文物数据集 {artifact_name} 处理完成！")

def main():
    parser = argparse.ArgumentParser(description="文物数据集处理")
    parser.add_argument("--artifact_name", type=str, required=True,
                       help="文物名称")
    parser.add_argument("--image_dir", type=str, required=True,
                       help="图像目录路径")
    parser.add_argument("--colmap_dir", type=str, default=None,
                       help="COLMAP结果目录")
    
    args = parser.parse_args()
    
    process_artifact_dataset(args.artifact_name, args.image_dir, args.colmap_dir)

if __name__ == "__main__":
    main()
