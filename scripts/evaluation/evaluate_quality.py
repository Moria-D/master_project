#!/usr/bin/env python3
"""
重建质量评估脚本
计算PSNR、SSIM、LPIPS指标
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

def load_image(image_path):
    """加载图像并归一化到[0,1]"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def calculate_metrics(gt_path, pred_path, lpips_fn):
    """计算单个图像的指标"""
    try:
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)
        
        # PSNR
        psnr_val = psnr(gt_img, pred_img, data_range=1.0)
        
        # SSIM
        ssim_val = ssim(gt_img, pred_img, channel_axis=2, data_range=1.0)
        
        # LPIPS
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        lpips_val = lpips_fn(gt_tensor, pred_tensor).item()
        
        return {
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_val
        }
    except Exception as e:
        print(f"计算指标时出错 {gt_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='评估重建质量')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实图像目录')
    parser.add_argument('--baseline_dir', type=str, required=True, help='Baseline渲染结果目录')
    parser.add_argument('--multiscale_dir', type=str, required=True, help='多尺度渲染结果目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化LPIPS
    lpips_fn = lpips.LPIPS(net='vgg')
    
    # 获取所有图像文件
    gt_files = sorted(Path(args.gt_dir).glob('*.png'))
    baseline_files = sorted(Path(args.baseline_dir).glob('*.png'))
    multiscale_files = sorted(Path(args.multiscale_dir).glob('*.png'))
    
    print(f"找到 {len(gt_files)} 个真实图像")
    print(f"找到 {len(baseline_files)} 个Baseline渲染图像")
    print(f"找到 {len(multiscale_files)} 个多尺度渲染图像")
    
    # 评估Baseline
    baseline_results = []
    for gt_file in gt_files:
        baseline_file = Path(args.baseline_dir) / gt_file.name
        if baseline_file.exists():
            metrics = calculate_metrics(gt_file, baseline_file, lpips_fn)
            if metrics:
                metrics['image'] = gt_file.name
                baseline_results.append(metrics)
    
    # 评估多尺度
    multiscale_results = []
    for gt_file in gt_files:
        multiscale_file = Path(args.multiscale_dir) / gt_file.name
        if multiscale_file.exists():
            metrics = calculate_metrics(gt_file, multiscale_file, lpips_fn)
            if metrics:
                metrics['image'] = gt_file.name
                multiscale_results.append(metrics)
    
    # 保存结果
    baseline_df = pd.DataFrame(baseline_results)
    multiscale_df = pd.DataFrame(multiscale_results)
    
    baseline_df.to_csv(os.path.join(args.output_dir, 'baseline_metrics.csv'), index=False)
    multiscale_df.to_csv(os.path.join(args.output_dir, 'multiscale_metrics.csv'), index=False)
    
    # 计算平均值
    print("\n=== 评估结果 ===")
    print("Baseline:")
    print(f"  PSNR: {baseline_df['psnr'].mean():.4f}")
    print(f"  SSIM: {baseline_df['ssim'].mean():.4f}")
    print(f"  LPIPS: {baseline_df['lpips'].mean():.4f}")
    
    print("\n多尺度:")
    print(f"  PSNR: {multiscale_df['psnr'].mean():.4f}")
    print(f"  SSIM: {multiscale_df['ssim'].mean():.4f}")
    print(f"  LPIPS: {multiscale_df['lpips'].mean():.4f}")
    
    # 保存汇总结果
    summary = {
        'baseline': {
            'psnr_mean': float(baseline_df['psnr'].mean()),
            'ssim_mean': float(baseline_df['ssim'].mean()),
            'lpips_mean': float(baseline_df['lpips'].mean()),
            'count': len(baseline_df)
        },
        'multiscale': {
            'psnr_mean': float(multiscale_df['psnr'].mean()),
            'ssim_mean': float(multiscale_df['ssim'].mean()),
            'lpips_mean': float(multiscale_df['lpips'].mean()),
            'count': len(multiscale_df)
        }
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    main()
