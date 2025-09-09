#!/usr/bin/env python3
"""
渲染效率评估脚本
分析FPS和LOD切换效果
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_fps_data(csv_path):
    """加载FPS数据"""
    if not os.path.exists(csv_path):
        print(f"警告: FPS数据文件不存在: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def analyze_fps(baseline_csv, multiscale_csv, output_dir):
    """分析FPS数据"""
    baseline_df = load_fps_data(baseline_csv)
    multiscale_df = load_fps_data(multiscale_csv)
    
    if baseline_df is None or multiscale_df is None:
        print("无法加载FPS数据，跳过效率分析")
        return
    
    # 计算统计信息
    baseline_stats = {
        'mean_fps': baseline_df['fps'].mean(),
        'std_fps': baseline_df['fps'].std(),
        'min_fps': baseline_df['fps'].min(),
        'max_fps': baseline_df['fps'].max()
    }
    
    multiscale_stats = {
        'mean_fps': multiscale_df['fps'].mean(),
        'std_fps': multiscale_df['fps'].std(),
        'min_fps': multiscale_df['fps'].min(),
        'max_fps': multiscale_df['fps'].max()
    }
    
    # 绘制FPS曲线
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(baseline_df['frame_idx'], baseline_df['fps'], 
             label='Baseline', color='blue', alpha=0.7)
    plt.plot(multiscale_df['frame_idx'], multiscale_df['fps'], 
             label='Multiscale LOD', color='red', alpha=0.7)
    plt.xlabel('Frame Index')
    plt.ylabel('FPS')
    plt.title('FPS Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制FPS分布
    plt.subplot(1, 2, 2)
    plt.hist(baseline_df['fps'], bins=30, alpha=0.7, label='Baseline', color='blue')
    plt.hist(multiscale_df['fps'], bins=30, alpha=0.7, label='Multiscale LOD', color='red')
    plt.xlabel('FPS')
    plt.ylabel('Frequency')
    plt.title('FPS Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fps_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计结果
    stats = {
        'baseline': baseline_stats,
        'multiscale': multiscale_stats,
        'improvement': {
            'fps_improvement': (multiscale_stats['mean_fps'] - baseline_stats['mean_fps']) / baseline_stats['mean_fps'] * 100,
            'min_fps_improvement': (multiscale_stats['min_fps'] - baseline_stats['min_fps']) / baseline_stats['min_fps'] * 100
        }
    }
    
    with open(os.path.join(output_dir, 'fps_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印结果
    print("=== FPS分析结果 ===")
    print(f"Baseline平均FPS: {baseline_stats['mean_fps']:.2f}")
    print(f"多尺度平均FPS: {multiscale_stats['mean_fps']:.2f}")
    print(f"FPS提升: {stats['improvement']['fps_improvement']:.2f}%")
    print(f"最低FPS提升: {stats['improvement']['min_fps_improvement']:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='评估渲染效率')
    parser.add_argument('--baseline_csv', type=str, required=True, help='Baseline FPS数据')
    parser.add_argument('--multiscale_csv', type=str, required=True, help='多尺度FPS数据')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_fps(args.baseline_csv, args.multiscale_csv, args.output_dir)

if __name__ == '__main__':
    main()
