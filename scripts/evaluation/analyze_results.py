#!/usr/bin/env python3
"""
实验结果分析脚本
生成论文所需的图表和数据
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import cv2
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentAnalyzer:
    """实验结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.plots_dir = self.results_dir / "plots"
        self.renders_dir = self.results_dir / "renders"
        
        # 创建输出目录
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_metrics(self) -> Dict:
        """加载评估指标"""
        metrics = {}
        
        # 加载质量指标
        quality_file = self.metrics_dir / "quality_metrics.csv"
        if quality_file.exists():
            metrics['quality'] = pd.read_csv(quality_file)
            
        # 加载效率指标
        efficiency_file = self.metrics_dir / "efficiency_metrics.csv"
        if efficiency_file.exists():
            metrics['efficiency'] = pd.read_csv(efficiency_file)
            
        # 加载模型大小
        size_file = self.metrics_dir / "model_size_comparison.txt"
        if size_file.exists():
            with open(size_file, 'r') as f:
                metrics['size'] = f.read()
                
        return metrics
    
    def plot_quality_comparison(self, metrics: pd.DataFrame):
        """绘制质量对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # PSNR对比
        methods = ['Baseline', 'Multiscale']
        psnr_values = [
            metrics['baseline_psnr'].mean(),
            metrics['multiscale_psnr'].mean()
        ]
        psnr_std = [
            metrics['baseline_psnr'].std(),
            metrics['multiscale_psnr'].std()
        ]
        
        axes[0].bar(methods, psnr_values, yerr=psnr_std, capsize=5)
        axes[0].set_title('PSNR对比')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].grid(True, alpha=0.3)
        
        # SSIM对比
        ssim_values = [
            metrics['baseline_ssim'].mean(),
            metrics['multiscale_ssim'].mean()
        ]
        ssim_std = [
            metrics['baseline_ssim'].std(),
            metrics['multiscale_ssim'].std()
        ]
        
        axes[1].bar(methods, ssim_values, yerr=ssim_std, capsize=5)
        axes[1].set_title('SSIM对比')
        axes[1].set_ylabel('SSIM')
        axes[1].grid(True, alpha=0.3)
        
        # LPIPS对比
        lpips_values = [
            metrics['baseline_lpips'].mean(),
            metrics['multiscale_lpips'].mean()
        ]
        lpips_std = [
            metrics['baseline_lpips'].std(),
            metrics['multiscale_lpips'].std()
        ]
        
        axes[2].bar(methods, lpips_values, yerr=lpips_std, capsize=5)
        axes[2].set_title('LPIPS对比')
        axes[2].set_ylabel('LPIPS (越低越好)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "quality_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_efficiency_analysis(self, metrics: pd.DataFrame):
        """绘制效率分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # FPS对比
        if 'baseline_fps' in metrics.columns and 'multiscale_fps' in metrics.columns:
            fps_data = []
            labels = []
            
            for method in ['baseline', 'multiscale']:
                fps_values = metrics[f'{method}_fps'].dropna()
                fps_data.extend(fps_values)
                labels.extend([method.capitalize()] * len(fps_values))
            
            fps_df = pd.DataFrame({'FPS': fps_data, 'Method': labels})
            
            sns.boxplot(data=fps_df, x='Method', y='FPS', ax=axes[0])
            axes[0].set_title('渲染帧率对比')
            axes[0].set_ylabel('FPS')
            axes[0].grid(True, alpha=0.3)
        
        # 内存占用对比
        if 'baseline_memory' in metrics.columns and 'multiscale_memory' in metrics.columns:
            memory_data = []
            labels = []
            
            for method in ['baseline', 'multiscale']:
                memory_values = metrics[f'{method}_memory'].dropna()
                memory_data.extend(memory_values)
                labels.extend([method.capitalize()] * len(memory_values))
            
            memory_df = pd.DataFrame({'Memory (MB)': memory_data, 'Method': labels})
            
            sns.boxplot(data=memory_df, x='Method', y='Memory (MB)', ax=axes[1])
            axes[1].set_title('内存占用对比')
            axes[1].set_ylabel('Memory (MB)')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "efficiency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_lod_analysis(self, metrics: pd.DataFrame):
        """绘制LOD分析图"""
        if 'distance' not in metrics.columns or 'lod_level' not in metrics.columns:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 距离-LOD层级关系
        lod_data = metrics.groupby('distance')['lod_level'].mean()
        axes[0].plot(lod_data.index, lod_data.values, 'o-', linewidth=2, markersize=6)
        axes[0].set_xlabel('距离 (m)')
        axes[0].set_ylabel('LOD层级')
        axes[0].set_title('距离-LOD层级关系')
        axes[0].grid(True, alpha=0.3)
        
        # 距离-质量关系
        if 'quality_at_distance' in metrics.columns:
            quality_data = metrics.groupby('distance')['quality_at_distance'].mean()
            axes[1].plot(quality_data.index, quality_data.values, 's-', linewidth=2, markersize=6)
            axes[1].set_xlabel('距离 (m)')
            axes[1].set_ylabel('质量指标')
            axes[1].set_title('距离-质量关系')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "lod_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_statistical_analysis(self, metrics: Dict) -> str:
        """生成统计分析报告"""
        report = []
        report.append("# 统计分析报告")
        report.append("")
        
        if 'quality' in metrics:
            quality_df = metrics['quality']
            
            # 质量指标统计
            report.append("## 质量指标统计分析")
            report.append("")
            
            for metric in ['psnr', 'ssim', 'lpips']:
                baseline_col = f'baseline_{metric}'
                multiscale_col = f'multiscale_{metric}'
                
                if baseline_col in quality_df.columns and multiscale_col in quality_df.columns:
                    baseline_values = quality_df[baseline_col].dropna()
                    multiscale_values = quality_df[multiscale_col].dropna()
                    
                    # 描述性统计
                    report.append(f"### {metric.upper()} 统计")
                    report.append(f"- Baseline: 均值={baseline_values.mean():.4f}, 标准差={baseline_values.std():.4f}")
                    report.append(f"- Multiscale: 均值={multiscale_values.mean():.4f}, 标准差={multiscale_values.std():.4f}")
                    
                    # t检验
                    t_stat, p_value = stats.ttest_ind(baseline_values, multiscale_values)
                    report.append(f"- t检验: t={t_stat:.4f}, p={p_value:.4f}")
                    
                    # 改进百分比
                    if metric != 'lpips':  # PSNR和SSIM越高越好
                        improvement = ((multiscale_values.mean() - baseline_values.mean()) / baseline_values.mean()) * 100
                        report.append(f"- 改进: {improvement:+.2f}%")
                    else:  # LPIPS越低越好
                        improvement = ((baseline_values.mean() - multiscale_values.mean()) / baseline_values.mean()) * 100
                        report.append(f"- 改进: {improvement:+.2f}%")
                    
                    report.append("")
        
        if 'efficiency' in metrics:
            efficiency_df = metrics['efficiency']
            report.append("## 效率指标统计分析")
            report.append("")
            
            # FPS统计
            if 'baseline_fps' in efficiency_df.columns and 'multiscale_fps' in efficiency_df.columns:
                baseline_fps = efficiency_df['baseline_fps'].dropna()
                multiscale_fps = efficiency_df['multiscale_fps'].dropna()
                
                fps_improvement = ((multiscale_fps.mean() - baseline_fps.mean()) / baseline_fps.mean()) * 100
                report.append(f"### FPS 统计")
                report.append(f"- Baseline: 均值={baseline_fps.mean():.2f} FPS")
                report.append(f"- Multiscale: 均值={multiscale_fps.mean():.2f} FPS")
                report.append(f"- 性能提升: {fps_improvement:+.2f}%")
                report.append("")
        
        return "\n".join(report)
    
    def create_artifact_detail_analysis(self):
        """创建文物细节分析"""
        # 这里可以添加文物特定的分析
        # 比如细节保持度、纹理质量等
        pass
    
    def generate_latex_tables(self, metrics: Dict) -> str:
        """生成LaTeX表格"""
        latex_tables = []
        
        if 'quality' in metrics:
            quality_df = metrics['quality']
            
            # 质量指标表格
            table_data = []
            for metric in ['psnr', 'ssim', 'lpips']:
                baseline_col = f'baseline_{metric}'
                multiscale_col = f'multiscale_{metric}'
                
                if baseline_col in quality_df.columns and multiscale_col in quality_df.columns:
                    baseline_mean = quality_df[baseline_col].mean()
                    baseline_std = quality_df[baseline_col].std()
                    multiscale_mean = quality_df[multiscale_col].mean()
                    multiscale_std = quality_df[multiscale_col].std()
                    
                    table_data.append([
                        metric.upper(),
                        f"{baseline_mean:.4f}±{baseline_std:.4f}",
                        f"{multiscale_mean:.4f}±{multiscale_std:.4f}"
                    ])
            
            if table_data:
                latex_table = "\\begin{table}[htbp]\n"
                latex_table += "\\centering\n"
                latex_table += "\\caption{质量指标对比结果}\n"
                latex_table += "\\label{tab:quality_comparison}\n"
                latex_table += "\\begin{tabular}{lcc}\n"
                latex_table += "\\hline\n"
                latex_table += "指标 & Baseline & Multiscale \\\\\n"
                latex_table += "\\hline\n"
                
                for row in table_data:
                    latex_table += f"{row[0]} & {row[1]} & {row[2]} \\\\\n"
                
                latex_table += "\\hline\n"
                latex_table += "\\end{tabular}\n"
                latex_table += "\\end{table}\n"
                
                latex_tables.append(latex_table)
        
        return "\n\n".join(latex_tables)
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始实验结果分析...")
        
        # 加载指标
        metrics = self.load_metrics()
        
        if not metrics:
            print("未找到评估指标文件")
            return
        
        # 生成图表
        if 'quality' in metrics:
            print("生成质量对比图...")
            self.plot_quality_comparison(metrics['quality'])
            
        if 'efficiency' in metrics:
            print("生成效率分析图...")
            self.plot_efficiency_analysis(metrics['efficiency'])
            self.plot_lod_analysis(metrics['efficiency'])
        
        # 生成统计报告
        print("生成统计分析报告...")
        report = self.generate_statistical_analysis(metrics)
        with open(self.plots_dir / "statistical_analysis.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 生成LaTeX表格
        print("生成LaTeX表格...")
        latex_tables = self.generate_latex_tables(metrics)
        with open(self.plots_dir / "latex_tables.tex", 'w', encoding='utf-8') as f:
            f.write(latex_tables)
        
        print(f"分析完成！结果保存在: {self.plots_dir}")

def main():
    parser = argparse.ArgumentParser(description="实验结果分析")
    parser.add_argument("--results_dir", type=str, default="eval",
                       help="结果目录路径")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(args.results_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
