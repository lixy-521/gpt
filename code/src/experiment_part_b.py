# experiment_part_b.py
import argparse
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data
import model

def count_non_embed_params(model):
    """计算非嵌入参数数量（排除 token_embedding 和 head）"""
    total = 0
    for name, param in model.named_parameters():
        if 'token_embedding' not in name and 'head' not in name and 'value_embedding' not in name:
            total += param.numel()
    return total

def train_fixed_steps(model, data_loader, device, num_steps=5000, lr=1e-3):
    """固定步数训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    data_loader.set_train()
    
    total_loss = 0.0
    step = 0
    
    while step < num_steps:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        if step % 500 == 0:
            print(f"  Step {step}/{num_steps}, loss: {loss.item():.4f}")
        
        if end_flag:
            data_loader.set_train()  # 重置 epoch
    
    return total_loss / num_steps

def evaluate_loss(model, data_loader, device, max_batches=None):
    """评估模型损失"""
    model.eval()
    data_loader.set_valid()
    
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        while True:
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target)
            total_loss += loss.item()
            total_steps += 1
            
            if max_batches and total_steps >= max_batches:
                break
            if end_flag:
                break
    
    return total_loss / total_steps

def compute_position_loss(model, data_loader, device, max_sql=256):
    """计算每个位置的损失"""
    model.eval()
    data_loader.set_valid()
    
    # 初始化位置损失统计
    position_losses = defaultdict(list)
    
    with torch.no_grad():
        while True:
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            
            logits = model(data)  # [seq_len, batch, vocab]
            seq_len = logits.size(0)
            
            # 计算每个位置的损失
            for pos in range(seq_len):
                # 获取该位置所有 batch 的 logits
                pos_logits = logits[pos]  # [batch, vocab]
                # 对应的 target 位置（target 是一维的，需要计算对应的索引）
                batch_size = pos_logits.size(0)
                pos_target = target[pos:pos + batch_size * (seq_len):seq_len]
                
                if len(pos_target) > 0:
                    loss = nn.functional.cross_entropy(pos_logits, pos_target)
                    position_losses[pos].append(loss.item())
            
            if end_flag:
                break
    
    # 计算每个位置的平均损失
    avg_losses = {pos: np.mean(losses) for pos, losses in position_losses.items()}
    return avg_losses

def run_scaling_law_experiment(data_path, device, num_steps=5000):
    """运行规模法则实验"""
    print("\n" + "="*60)
    print("Part B.1: Scaling Laws Experiment")
    print("="*60)
    
    # 定义不同的模型配置
    configs = [
        # (dim, num_layers, num_heads, name)
        (32, 2, 2, "Tiny"),
        (32, 4, 4, "Small-1"),
        (64, 2, 2, "Small-2"),
        (64, 4, 4, "Medium-1"),
        (64, 6, 4, "Medium-2"),
        (96, 4, 6, "Large-1"),
        (96, 6, 6, "Large-2"),
        (128, 4, 4, "Base"),
        (128, 6, 8, "Big"),
    ]
    
    results = []
    
    for dim, num_layers, num_heads, name in configs:
        print(f"\n--- Training {name} (dim={dim}, layers={num_layers}, heads={num_heads}) ---")
        
        # 重新加载数据（确保每次使用相同的数据）
        batch_size = {'train': 16, 'valid': 16}
        data_loader = data.Corpus(data_path, batch_size, 256)
        
        # 创建模型
        lm_model = model.CausalLMM(
            vocab_size=len(data_loader.vocabulary),
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            use_qk_norm=False,
            use_attn_gate=False,
            use_value_embed=False
        )
        lm_model = lm_model.to(device)
        
        # 计算非嵌入参数
        non_embed_params = count_non_embed_params(lm_model)
        
        # 训练固定步数
        train_loss = train_fixed_steps(lm_model, data_loader, device, num_steps)
        
        # 评估验证损失
        valid_loss = evaluate_loss(lm_model, data_loader, device)
        
        results.append({
            'name': name,
            'dim': dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'non_embed_params': non_embed_params,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'perplexity': math.exp(valid_loss)
        })
        
        print(f"  Non-embed params: {non_embed_params:,}")
        print(f"  Valid loss: {valid_loss:.4f}, Perplexity: {math.exp(valid_loss):.2f}")
    
    return results

def run_architecture_variants(data_path, device, num_steps=5000):
    """运行架构变体实验"""
    print("\n" + "="*60)
    print("Part B.2: Architecture Variants Experiment")
    print("="*60)
    
    # 定义不同规模的配置
    sizes = [
        (64, 2, 4, "Small"),
        (64, 4, 4, "Medium"),
        (96, 4, 6, "Large"),
    ]
    
    architectures = [
        ('baseline', False, False, False),
        ('qk_norm', True, False, False),
        ('attn_gate', False, True, False),
        ('value_embed', False, False, True),
    ]
    
    all_results = []
    
    for dim, num_layers, num_heads, size_name in sizes:
        for arch_name, use_qk_norm, use_attn_gate, use_value_embed in architectures:
            exp_name = f"{size_name}_{arch_name}"
            print(f"\n--- {exp_name} (dim={dim}, layers={num_layers}, heads={num_heads}) ---")
            
            # 重新加载数据
            batch_size = {'train': 16, 'valid': 16}
            data_loader = data.Corpus(data_path, batch_size, 256)
            
            # 创建模型
            lm_model = model.CausalLMM(
                vocab_size=len(data_loader.vocabulary),
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=0.1,
                use_qk_norm=use_qk_norm,
                use_attn_gate=use_attn_gate,
                use_value_embed=use_value_embed
            )
            lm_model = lm_model.to(device)
            
            # 计算非嵌入参数
            non_embed_params = count_non_embed_params(lm_model)
            
            # 训练固定步数
            train_loss = train_fixed_steps(lm_model, data_loader, device, num_steps)
            
            # 评估验证损失
            valid_loss = evaluate_loss(lm_model, data_loader, device)
            
            all_results.append({
                'exp_name': exp_name,
                'size_name': size_name,
                'arch_name': arch_name,
                'dim': dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'non_embed_params': non_embed_params,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'perplexity': math.exp(valid_loss),
                'use_qk_norm': use_qk_norm,
                'use_attn_gate': use_attn_gate,
                'use_value_embed': use_value_embed
            })
            
            print(f"  Non-embed params: {non_embed_params:,}")
            print(f"  Valid loss: {valid_loss:.4f}, Perplexity: {math.exp(valid_loss):.2f}")
    
    return all_results

def run_position_analysis(best_model_path, data_path, device):
    """运行位置损失分析"""
    print("\n" + "="*60)
    print("Part B.1.2: Position-wise Loss Analysis")
    print("="*60)
    
    # 加载最佳模型
    batch_size = {'train': 16, 'valid': 16}
    data_loader = data.Corpus(data_path, batch_size, 256)
    
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # 获取模型配置
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        lm_model = model.CausalLMM(
            vocab_size=len(data_loader.vocabulary),
            dim=saved_args.get('emb_dim', 128),
            num_layers=saved_args.get('num_layers', 4),
            num_heads=saved_args.get('num_heads', 4),
            dropout=0.1
        )
    else:
        lm_model = model.CausalLMM(
            vocab_size=len(data_loader.vocabulary),
            dim=128,
            num_layers=4,
            num_heads=4,
            dropout=0.1
        )
    
    # 加载权重
    model_state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                          if 'rope.cos_cached' not in k and 'rope.sin_cached' not in k}
    lm_model.load_state_dict(filtered_state_dict, strict=False)
    lm_model = lm_model.to(device)
    
    # 计算位置损失
    position_losses = compute_position_loss(lm_model, data_loader, device)
    
    return position_losses

def plot_scaling_law(results, save_path='scaling_law.png'):
    """绘制规模法则图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 验证损失 vs 非嵌入参数 (log-log)
    ax1 = axes[0]
    params = [r['non_embed_params'] for r in results]
    losses = [r['valid_loss'] for r in results]
    
    ax1.loglog(params, losses, 'o-', color='blue', linewidth=2, markersize=8, label='Experimental')
    
    # 线性拟合 (在 log-log 空间)
    log_params = np.log10(params)
    log_losses = np.log10(losses)
    coeffs = np.polyfit(log_params, log_losses, 1)
    poly = np.poly1d(coeffs)
    fitted_log_losses = poly(log_params)
    fitted_losses = 10 ** fitted_log_losses
    
    ax1.loglog(params, fitted_losses, '--', color='red', linewidth=2, 
               label=f'Power-law fit: $L \\propto P^{{{coeffs[0]:.2f}}}$')
    
    ax1.set_xlabel('Non-embedding Parameters', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Scaling Law: Loss vs Parameters (log-log)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 在图上标注每个点
    for r in results:
        ax1.annotate(r['name'], (r['non_embed_params'], r['valid_loss']), 
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    
    # 图2: 困惑度 vs 非嵌入参数
    ax2 = axes[1]
    perplexities = [r['perplexity'] for r in results]
    ax2.semilogx(params, perplexities, 's-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Non-embedding Parameters', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Perplexity vs Model Size', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Scaling law plot saved to {save_path}")

def plot_architecture_comparison(all_results, save_path='architecture_comparison.png'):
    """绘制架构变体比较图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按架构分组
    arch_groups = {}
    for r in all_results:
        arch = r['arch_name']
        if arch not in arch_groups:
            arch_groups[arch] = {'params': [], 'losses': []}
        arch_groups[arch]['params'].append(r['non_embed_params'])
        arch_groups[arch]['losses'].append(r['valid_loss'])
    
    # 颜色和标记
    colors = {
        'baseline': 'blue',
        'qk_norm': 'red',
        'attn_gate': 'green',
        'value_embed': 'orange'
    }
    markers = {
        'baseline': 'o',
        'qk_norm': 's',
        'attn_gate': '^',
        'value_embed': 'D'
    }
    labels = {
        'baseline': 'Baseline (Original)',
        'qk_norm': 'QK Norm',
        'attn_gate': 'Attention Gate',
        'value_embed': 'Value Embedding'
    }
    
    for arch, data in arch_groups.items():
        # 按参数量排序
        sorted_indices = np.argsort(data['params'])
        params_sorted = np.array(data['params'])[sorted_indices]
        losses_sorted = np.array(data['losses'])[sorted_indices]
        
        ax.loglog(params_sorted, losses_sorted, 
                 marker=markers[arch], color=colors[arch], 
                 linewidth=2, markersize=10, label=labels[arch])
    
    ax.set_xlabel('Non-embedding Parameters', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Architecture Comparison: Loss vs Model Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Architecture comparison plot saved to {save_path}")

def plot_position_loss(position_losses, save_path='position_loss.png'):
    """绘制位置损失曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 按位置分组 (每32个位置一组)
    positions = sorted(position_losses.keys())
    losses = [position_losses[p] for p in positions]
    
    # 原始曲线
    ax.plot(positions, losses, 'b-', linewidth=2, alpha=0.7, label='Per-position loss')
    
    # 计算滑动平均
    window_size = 8
    if len(losses) > window_size:
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_positions = positions[window_size//2:-(window_size//2)]
        ax.plot(smoothed_positions, smoothed, 'r-', linewidth=3, label='Smoothed (window=8)')
    
    # 标记分组边界
    for group in range(1, 9):
        pos = group * 32
        if pos < max(positions):
            ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
            ax.text(pos, min(losses), f'Pos {pos}', rotation=90, fontsize=8)
    
    ax.set_xlabel('Token Position in Sequence', fontsize=12)
    ax.set_ylabel('Average Cross-Entropy Loss', fontsize=12)
    ax.set_title('Loss vs Token Position: Effect of Context Length', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Position loss plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/ptb')
    parser.add_argument('--num_steps', type=int, default=5000, 
                        help='Number of training steps for each model')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--skip_scaling', action='store_true', 
                        help='Skip scaling law experiment')
    parser.add_argument('--skip_arch', action='store_true',
                        help='Skip architecture experiment')
    parser.add_argument('--skip_position', action='store_true',
                        help='Skip position analysis')
    parser.add_argument('--best_model_path', type=str, 
                        default='../checkpoints/best_model.pt',
                        help='Path to best model for position analysis')
    args = parser.parse_args()
    
    # 设置设备
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    results_dir = 'experiment_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 规模法则实验
    scaling_results = None
    if not args.skip_scaling:
        scaling_results = run_scaling_law_experiment(args.data_path, device, args.num_steps)
        
        # 保存结果
        with open(os.path.join(results_dir, 'scaling_results.json'), 'w') as f:
            json.dump(scaling_results, f, indent=2)
        
        # 绘制规模法则图
        plot_scaling_law(scaling_results, os.path.join(results_dir, 'scaling_law.png'))
    
    # 2. 架构变体实验
    arch_results = None
    if not args.skip_arch:
        arch_results = run_architecture_variants(args.data_path, device, args.num_steps)
        
        # 保存结果
        with open(os.path.join(results_dir, 'arch_results.json'), 'w') as f:
            json.dump(arch_results, f, indent=2)
        
        # 绘制架构比较图
        plot_architecture_comparison(arch_results, os.path.join(results_dir, 'architecture_comparison.png'))
    
    # 3. 位置损失分析
    if not args.skip_position:
        # 首先，从规模实验中找出最佳模型
        if scaling_results:
            best_model = min(scaling_results, key=lambda x: x['valid_loss'])
            print(f"\nBest model from scaling experiment: {best_model['name']}")
            print(f"  Valid loss: {best_model['valid_loss']:.4f}")
        
        # 运行位置分析
        position_losses = run_position_analysis(args.best_model_path, args.data_path, device)
        
        # 保存结果
        with open(os.path.join(results_dir, 'position_losses.json'), 'w') as f:
            # 转换键为字符串以便 JSON 序列化
            json.dump({str(k): v for k, v in position_losses.items()}, f, indent=2)
        
        # 绘制位置损失图
        plot_position_loss(position_losses, os.path.join(results_dir, 'position_loss.png'))
    
    # 打印摘要
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if scaling_results:
        print(f"\nScaling Law Experiment: {len(scaling_results)} models trained")
        print(f"  Parameter range: {min(r['non_embed_params']):,} - {max(r['non_embed_params']):,}")
        print(f"  Loss range: {min(r['valid_loss']):.4f} - {max(r['valid_loss']):.4f}")
    
    if arch_results:
        print(f"\nArchitecture Experiment: {len(arch_results)} models trained")
        for arch in ['baseline', 'qk_norm', 'attn_gate', 'value_embed']:
            arch_data = [r for r in arch_results if r['arch_name'] == arch]
            if arch_data:
                avg_loss = np.mean([r['valid_loss'] for r in arch_data])
                print(f"  {arch}: avg loss = {avg_loss:.4f}")
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main()