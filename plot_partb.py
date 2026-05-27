# coding: utf-8
"""
Part B 绘图脚本 - 基于已有的实验结果
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 实验数据（从训练日志中提取）
baseline_data = {
    'tiny': {'params': 3622928, 'best_ppl': 231.17},
    'small': {'params': 7398544, 'best_ppl': 230.83},
    'base_small': {'params': 11434256, 'best_ppl': 233.67},
    'medium': {'params': 26589712, 'best_ppl': 230.66},
    'base': {'params': 43838224, 'best_ppl': 231.40},
    'large': {'params': 78398480, 'best_ppl': 239.46},
}

# 架构变体数据
variants_data = {
    # QK Norm
    'qk_norm_128': {'params': 3622928, 'best_ppl': None},  # 需要运行
    'qk_norm_256': {'params': 11434256, 'best_ppl': None},
    'qk_norm_384': {'params': 26589712, 'best_ppl': None},
    # Attention Gate
    'attn_gate_128': {'params': 3622928, 'best_ppl': None},
    'attn_gate_256': {'params': 11434256, 'best_ppl': None},
    'attn_gate_384': {'params': 26589712, 'best_ppl': None},
    # Value Embedding
    'value_embed_128': {'params': 3622928, 'best_ppl': None},
    'value_embed_256': {'params': 11434256, 'best_ppl': None},
    'value_embed_384': {'params': 26589712, 'best_ppl': None},
}

def plot_scaling_law():
    """图1: Loss vs Non-Embedding Parameters (log-log)"""
    params = [d['params'] for d in baseline_data.values()]
    ppls = [d['best_ppl'] for d in baseline_data.values()]
    
    # 排序（按参数量）
    sorted_idx = np.argsort(params)
    params = np.array(params)[sorted_idx]
    ppls = np.array(ppls)[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制数据点
    ax.loglog(params, ppls, 'bo-', markersize=10, linewidth=2, label='Baseline')
    
    # 线性拟合 (log-log space)
    log_params = np.log10(params)
    log_ppls = np.log10(ppls)
    coeffs = np.polyfit(log_params, log_ppls, 1)
    poly = np.poly1d(coeffs)
    
    # 拟合线
    params_fit = np.logspace(np.log10(params.min()), np.log10(params.max()), 100)
    ppls_fit = 10 ** poly(np.log10(params_fit))
    ax.loglog(params_fit, ppls_fit, 'r--', linewidth=2, 
              label=f'Power-law fit: slope = {coeffs[0]:.3f}')
    
    # 添加标注
    for name, d in baseline_data.items():
        ax.annotate(name, (d['params'], d['best_ppl']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Non-Embedding Parameters', fontsize=12)
    ax.set_ylabel('Validation Perplexity', fontsize=12)
    ax.set_title('Scaling Law: Performance vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('scaling_law.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: scaling_law.png (slope = {coeffs[0]:.3f})")
    
    return coeffs

def plot_variants_comparison():
    """图2: 架构变体对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Baseline
    params_b = [d['params'] for d in baseline_data.values()]
    ppls_b = [d['best_ppl'] for d in baseline_data.values()]
    sorted_idx = np.argsort(params_b)
    params_b = np.array(params_b)[sorted_idx]
    ppls_b = np.array(ppls_b)[sorted_idx]
    ax.loglog(params_b, ppls_b, 'ko-', linewidth=2, markersize=8, label='Baseline')
    
    # 这里需要填入变体的实际数据
    # 示例数据（需要替换为实际运行结果）
    variants_results = {
        'QK Norm': {'params': [3622928, 11434256, 26589712], 
                    'ppls': [232.5, 234.1, 231.8],
                    'marker': 's', 'color': 'blue'},
        'Attention Gate': {'params': [3622928, 11434256, 26589712],
                          'ppls': [233.2, 235.0, 232.5],
                          'marker': '^', 'color': 'red'},
        'Value Embedding': {'params': [3622928, 11434256, 26589712],
                           'ppls': [231.9, 233.5, 231.0],
                           'marker': 'd', 'color': 'green'},
    }
    
    for name, data in variants_results.items():
        ax.loglog(data['params'], data['ppls'], 
                 marker=data['marker'], color=data['color'], 
                 linestyle='--', linewidth=1.5, markersize=8,
                 label=name)
    
    ax.set_xlabel('Non-Embedding Parameters', fontsize=12)
    ax.set_ylabel('Validation Perplexity', fontsize=12)
    ax.set_title('Architectural Variants Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('variants_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: variants_comparison.png")

def create_summary_table():
    """创建结果汇总表"""
    print("\n" + "="*70)
    print("PART B RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Params':<15} {'Best Valid PPL':<15} {'Note':<10}")
    print("-"*70)
    for name, d in baseline_data.items():
        print(f"{name:<15} {d['params']:<15,} {d['best_ppl']:<15.2f} {'baseline':<10}")
    print("-"*70)
    
    # 保存为 CSV
    import csv
    with open('partb_results_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Non-Embed Parameters', 'Best Valid PPL', 'Type'])
        for name, d in baseline_data.items():
            writer.writerow([name, d['params'], d['best_ppl'], 'baseline'])
    print("Saved: partb_results_summary.csv")

def main():
    print("Generating Part B plots...")
    coeffs = plot_scaling_law()
    plot_variants_comparison()
    create_summary_table()
    
    # 保存拟合参数
    with open('scaling_fit.json', 'w') as f:
        json.dump({
            'slope': float(coeffs[0]),
            'intercept': float(coeffs[1]),
            'description': 'log(PPL) = slope * log(params) + intercept'
        }, f, indent=2)
    print("Saved: scaling_fit.json")
    
    print("\nDone! Generated files:")
    print("  - scaling_law.png")
    print("  - variants_comparison.png")
    print("  - partb_results_summary.csv")
    print("  - scaling_fit.json")

if __name__ == '__main__':
    main()