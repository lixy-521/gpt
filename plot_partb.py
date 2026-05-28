# coding: utf-8
"""
Part B 最终绘图脚本 - 基于完整实验结果
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# ========== 数据定义 ==========
# Baseline 数据
baseline_data = {
    'tiny': {'params': 3622928, 'ppl': 231.17},
    'small': {'params': 7398544, 'ppl': 230.83},
    'base_small': {'params': 11434256, 'ppl': 233.67},
    'medium': {'params': 26589712, 'ppl': 230.66},
    'base': {'params': 43838224, 'ppl': 231.40},
    'large': {'params': 78398480, 'ppl': 239.46},
    'xlarge': {'params': 128691472, 'ppl': 243.90},
}

# 架构变体数据 - 修正结构
variants_data = {
    'QK Norm': {
        'sizes': [128, 256, 384],
        'params': [3623440, 11435024, 26590736],
        'ppls': [263.37, 257.51, 255.12],
        'color': 'green',
        'marker': 's'
    },
    'Attention Gate': {
        'sizes': [128, 256, 384],
        'params': [3622944, 11434304, 26589808],
        'ppls': [234.02, 226.31, 230.97],
        'color': 'red',
        'marker': '^'
    },
    'Value Embedding': {
        'sizes': [128, 256, 384],
        'params': [4902928, 13994256, 30429712],
        'ppls': [240.70, 228.35, 230.85],
        'color': 'purple',
        'marker': 'd'
    },
}

# Baseline 最终训练/验证困惑度（用于过拟合分析）
baseline_final = {
    'tiny': {'train': 115.68, 'valid': 236.56},
    'small': {'train': 94.83, 'valid': 247.86},
    'base_small': {'train': 85.66, 'valid': 254.21},
    'medium': {'train': 71.66, 'valid': 259.12},
    'base': {'train': 47.81, 'valid': 298.71},
    'large': {'train': 46.91, 'valid': 303.44},
    'xlarge': {'train': 46.64, 'valid': 312.80},
}

def plot_scaling_law():
    """图1: Scaling Law - 参数量 vs 困惑度"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 按参数量排序
    names = list(baseline_data.keys())
    params = [baseline_data[n]['params'] for n in names]
    ppls = [baseline_data[n]['ppl'] for n in names]
    sorted_idx = np.argsort(params)
    params = np.array(params)[sorted_idx]
    ppls = np.array(ppls)[sorted_idx]
    names_sorted = np.array(names)[sorted_idx]
    
    # 绘制 Baseline 点
    ax.loglog(params, ppls, 'bo-', linewidth=2, markersize=10, label='Baseline')
    
    # 线性拟合 (log-log)
    log_params = np.log10(params)
    log_ppls = np.log10(ppls)
    coeffs = np.polyfit(log_params, log_ppls, 1)
    poly = np.poly1d(coeffs)
    
    params_fit = np.logspace(np.log10(params.min()), np.log10(params.max()), 100)
    ppls_fit = 10 ** poly(np.log10(params_fit))
    ax.loglog(params_fit, ppls_fit, 'r--', linewidth=2, 
              label=f'Power-law fit: slope = {coeffs[0]:.3f}')
    
    # 标注数据点
    for name, p, ppl in zip(names_sorted, params, ppls):
        ax.annotate(name, (p, ppl), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
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
    params_b = [baseline_data[n]['params'] for n in baseline_data]
    ppls_b = [baseline_data[n]['ppl'] for n in baseline_data]
    sorted_idx = np.argsort(params_b)
    params_b = np.array(params_b)[sorted_idx]
    ppls_b = np.array(ppls_b)[sorted_idx]
    ax.loglog(params_b, ppls_b, 'ko-', linewidth=2, markersize=8, 
              label='Baseline', zorder=10)
    
    # 变体数据
    for name, data in variants_data.items():
        params_v = data['params']
        ppls_v = data['ppls']
        sizes = data['sizes']
        
        ax.loglog(params_v, ppls_v, 
                 marker=data['marker'], color=data['color'],
                 linestyle='--', linewidth=1.5, markersize=9,
                 label=name, zorder=5)
        
        # 标注每个点对应的维度
        for p, ppl, s in zip(params_v, ppls_v, sizes):
            ax.annotate(f'{s}d', (p, ppl), xytext=(5, -8), 
                       textcoords='offset points', fontsize=8, color=data['color'])
        
        # 标注最佳点
        best_idx = np.argmin(ppls_v)
        ax.annotate(f'best: {ppls_v[best_idx]:.1f}', 
                   (params_v[best_idx], ppls_v[best_idx]),
                   xytext=(5, 10), textcoords='offset points', 
                   fontsize=8, color=data['color'], weight='bold')
    
    ax.set_xlabel('Non-Embedding Parameters', fontsize=12)
    ax.set_ylabel('Validation Perplexity', fontsize=12)
    ax.set_title('Architectural Variants Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('variants_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: variants_comparison.png")

def plot_overfitting_analysis():
    """图3: 过拟合分析 - 训练 vs 验证困惑度"""
    models = list(baseline_final.keys())
    train_ppls = [baseline_final[m]['train'] for m in models]
    valid_ppls = [baseline_final[m]['valid'] for m in models]
    params = [baseline_data[m]['params'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_ppls, width, label='Final Train PPL', color='steelblue')
    bars2 = ax.bar(x + width/2, valid_ppls, width, label='Final Valid PPL', color='coral')
    
    # 添加过拟合比率标签
    for i, (train, valid) in enumerate(zip(train_ppls, valid_ppls)):
        ratio = valid / train
        ax.text(i, valid + 10, f'{ratio:.1f}x', ha='center', fontsize=9, color='red')
    
    # 添加参数量标签
    for i, p in enumerate(params):
        ax.text(i, -30, f'{p//1000000:.0f}M', ha='center', fontsize=8, color='gray')
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Overfitting Analysis: Train vs Validation Perplexity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加 y 轴范围
    ax.set_ylim(0, 350)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: overfitting_analysis.png")

def plot_combined_comparison():
    """图4: 组合对比图 - 各规模下变体性能"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = [128, 256, 384]
    x = np.arange(len(sizes))
    width = 0.25
    
    # Baseline 在对应规模的值
    baseline_at_size = {
        128: baseline_data['tiny']['ppl'],
        256: baseline_data['base_small']['ppl'],
        384: baseline_data['medium']['ppl'],
    }
    
    colors = {'QK Norm': 'green', 'Attention Gate': 'red', 'Value Embedding': 'purple'}
    markers = {'QK Norm': 's', 'Attention Gate': '^', 'Value Embedding': 'd'}
    
    # 绘制 Baseline 线
    ax.axhline(y=baseline_at_size[128], xmin=0, xmax=0.25, color='blue', linestyle='-', linewidth=2, label='Baseline')
    ax.axhline(y=baseline_at_size[256], xmin=0.35, xmax=0.65, color='blue', linewidth=2)
    ax.axhline(y=baseline_at_size[384], xmin=0.7, xmax=1.0, color='blue', linewidth=2)
    
    # 添加 Baseline 标注
    ax.text(-0.1, baseline_at_size[128] + 2, f'231.2', ha='center', fontsize=8, color='blue')
    ax.text(0.9, baseline_at_size[256] + 2, f'233.7', ha='center', fontsize=8, color='blue')
    ax.text(1.9, baseline_at_size[384] + 2, f'230.7', ha='center', fontsize=8, color='blue')
    
    # 绘制各变体
    for i, (name, data) in enumerate(variants_data.items()):
        offset = (i - 1) * width
        ppls_at_size = data['ppls']
        ax.bar(x + offset, ppls_at_size, width, label=name, 
               color=colors[name], alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Validation Perplexity', fontsize=12)
    ax.set_title('Architectural Variants Performance by Model Size', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['128 (tiny)', '256 (base_small)', '384 (medium)'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('combined_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: combined_comparison.png")

def create_summary_table():
    """生成结果汇总表"""
    print("\n" + "="*80)
    print("PART B - COMPLETE RESULTS SUMMARY")
    print("="*80)
    
    print("\n--- Baseline Models ---")
    print(f"{'Model':<12} {'Params':<15} {'Best PPL':<12} {'Final Train PPL':<15} {'Final Valid PPL':<15} {'Ratio':<10}")
    print("-"*75)
    for name in baseline_data.keys():
        data = baseline_data[name]
        final = baseline_final[name]
        ratio = final['valid'] / final['train']
        print(f"{name:<12} {data['params']:<15,} {data['ppl']:<12.2f} {final['train']:<15.2f} {final['valid']:<15.2f} {ratio:<10.1f}x")
    
    print("\n--- Architectural Variants (Best Performance) ---")
    print(f"{'Variant':<18} {'128-dim (tiny)':<18} {'256-dim (base_small)':<18} {'384-dim (medium)':<18}")
    print("-"*72)
    for name, data in variants_data.items():
        ppl_128 = data['ppls'][0]
        ppl_256 = data['ppls'][1]
        ppl_384 = data['ppls'][2]
        print(f"{name:<18} {ppl_128:<18.2f} {ppl_256:<18.2f} {ppl_384:<18.2f}")
    
    # 最佳改进
    print("\n--- Best Improvements vs Baseline ---")
    baseline_256 = baseline_data['base_small']['ppl']
    best_gate = variants_data['Attention Gate']['ppls'][1]
    best_ve = variants_data['Value Embedding']['ppls'][1]
    print(f"Attention Gate (256-dim): {best_gate:.2f} (improved by {baseline_256 - best_gate:.2f})")
    print(f"Value Embedding (256-dim): {best_ve:.2f} (improved by {baseline_256 - best_ve:.2f})")
    
    print("\n--- Key Findings ---")
    print(f"1. Best baseline model: medium (26.6M params) with PPL = 230.66")
    print(f"2. Best architectural variant: Attention Gate (256-dim) with PPL = 226.31")
    print(f"3. QK Norm consistently underperforms baseline across all scales")
    print(f"4. Overfitting worsens with model size: xlarge has {baseline_final['xlarge']['valid'] / baseline_final['xlarge']['train']:.1f}x gap")

def save_results():
    """保存结果到 JSON"""
    results = {
        'baseline': {k: {'params': v['params'], 'best_ppl': v['ppl'], 
                        'final_train': baseline_final[k]['train'],
                        'final_valid': baseline_final[k]['valid']} 
                    for k, v in baseline_data.items()},
        'variants': {
            name: {
                'sizes': data['sizes'],
                'params': data['params'],
                'ppls': data['ppls']
            } for name, data in variants_data.items()
        }
    }
    with open('partb_results_complete.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: partb_results_complete.json")

def main():
    print("Generating Part B plots with complete data...")
    
    plot_scaling_law()
    plot_variants_comparison()
    plot_overfitting_analysis()
    plot_combined_comparison()
    create_summary_table()
    save_results()
    
    print("\n" + "="*50)
    print("Generated files:")
    print("  - scaling_law.png")
    print("  - variants_comparison.png")
    print("  - overfitting_analysis.png")
    print("  - combined_comparison.png")
    print("  - partb_results_complete.json")
    print("="*50)

if __name__ == '__main__':
    main()