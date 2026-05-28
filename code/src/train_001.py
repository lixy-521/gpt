
# coding: utf-8
"""
Part B: Scaling Laws and Architectural Study
用于研究模型规模与性能的关系，以及架构变体的影响
"""
import argparse
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data
import model


def get_model_params(vocab_size, num_layers, num_heads, emb_dim, dropout=0.1,
                     use_qk_norm=False, use_attn_gate=False, use_value_embed=False):
    """计算模型的参数量"""
    lm_model = model.CausalLMM(
        vocab_size=vocab_size,
        dim=emb_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_qk_norm=use_qk_norm,
        use_attn_gate=use_attn_gate,
        use_value_embed=use_value_embed
    )
    total_params = sum(p.numel() for p in lm_model.parameters())
    non_embed_params = sum(p.numel() for p in lm_model.parameters() 
                           if 'embedding' not in p.__repr__().lower())
    return total_params, non_embed_params


def train_single_config(config, data_loader, device, save_dir):
    """
    训练单个配置的模型
    返回: best_valid_ppl, non_embed_params
    """
    vocab_size = len(data_loader.vocabulary)
    
    print("\n" + "="*70)
    print(f"Training config: {config['name']}")
    print(f"  layers={config['num_layers']}, heads={config['num_heads']}, "
          f"dim={config['emb_dim']}, dropout={config.get('dropout', 0.1)}")
    if config.get('use_qk_norm'):
        print("  + QK Norm")
    if config.get('use_attn_gate'):
        print("  + Attention Gate")
    if config.get('use_value_embed'):
        print("  + Value Embedding")
    print("="*70)
    
    # 创建模型
    lm_model = model.CausalLMM(
        vocab_size=vocab_size,
        dim=config['emb_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config.get('dropout', 0.1),
        use_qk_norm=config.get('use_qk_norm', False),
        use_attn_gate=config.get('use_attn_gate', False),
        use_value_embed=config.get('use_value_embed', False)
    )
    lm_model = lm_model.to(device)
    
    total_params, non_embed_params = get_model_params(
        vocab_size, config['num_layers'], config['num_heads'],
        config['emb_dim'], config.get('dropout', 0.1),
        config.get('use_qk_norm', False),
        config.get('use_attn_gate', False),
        config.get('use_value_embed', False)
    )
    
    print(f"Non-embedding parameters: {non_embed_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(lm_model.parameters(), 
                            lr=config.get('lr', 3e-4), 
                            weight_decay=config.get('weight_decay', 0.01))
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    
    # 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 15))
    
    best_valid_ppl = float('inf')
    patience_counter = 0
    patience = config.get('patience', 5)
    
    train_losses = []
    valid_losses = []
    train_ppls = []
    valid_ppls = []
    
    def evaluate():
        data_loader.set_valid()
        lm_model.eval()
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            while True:
                data_batch, target_batch, end_flag = data_loader.get_batch()
                data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                logits = lm_model(data_batch)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_batch)
                total_loss += loss.item()
                steps += 1
                if end_flag:
                    break
        return total_loss / steps, math.exp(total_loss / steps)
    
    def train_one_epoch():
        data_loader.set_train()
        lm_model.train()
        total_loss = 0.0
        steps = 0
        while True:
            data_batch, target_batch, end_flag = data_loader.get_batch()
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            logits = lm_model(data_batch)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_model.parameters(), config.get('grad_clip', 0.5))
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if end_flag:
                break
        return total_loss / steps, math.exp(total_loss / steps)
    
    # 训练循环
    for epoch in range(config.get('epochs', 15)):
        train_loss, train_ppl = train_one_epoch()
        valid_loss, valid_ppl = evaluate()
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_ppls.append(train_ppl)
        valid_ppls.append(valid_ppl)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train PPL={train_ppl:.2f}, Valid PPL={valid_ppl:.2f}")
        
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            patience_counter = 0
            print(f"  New best! Valid PPL={best_valid_ppl:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 保存结果
    result = {
        'name': config['name'],
        'num_layers': config['num_layers'],
        'num_heads': config['num_heads'],
        'emb_dim': config['emb_dim'],
        'non_embed_params': non_embed_params,
        'best_valid_ppl': best_valid_ppl,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_ppls': train_ppls,
        'valid_ppls': valid_ppls,
        'config': config
    }
    
    return result


def plot_scaling_laws(results, save_dir):
    """
    绘制 Scaling Laws 图
    1. Loss vs Non-Embedding Parameters (log-log)
    2. 架构变体对比图
    """
    # 分离 baseline 和变体
    baselines = [r for r in results if not (r['config'].get('use_qk_norm') or 
                                             r['config'].get('use_attn_gate') or 
                                             r['config'].get('use_value_embed'))]
    variants = [r for r in results if (r['config'].get('use_qk_norm') or 
                                        r['config'].get('use_attn_gate') or 
                                        r['config'].get('use_value_embed'))]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: Loss vs Non-Embedding Parameters (log-log)
    ax1 = axes[0]
    params = [r['non_embed_params'] for r in baselines]
    losses = [math.log(r['best_valid_ppl']) for r in baselines]  # log PPL as loss proxy
    
    ax1.loglog(params, losses, 'bo-', markersize=8, linewidth=2, label='Baseline')
    
    # 添加线性拟合
    log_params = np.log10(params)
    log_losses = np.log10([r['best_valid_ppl'] for r in baselines])
    coeffs = np.polyfit(log_params, log_losses, 1)
    poly = np.poly1d(coeffs)
    fitted_losses = 10**poly(log_params)
    ax1.loglog(params, fitted_losses, 'r--', linewidth=2, 
               label=f'Linear fit: slope={coeffs[0]:.2f}')
    
    ax1.set_xlabel('Non-Embedding Parameters')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title('Scaling Law: Performance vs Model Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2: 架构变体对比
    ax2 = axes[1]
    
    # 按参数数量排序
    all_results = sorted(results, key=lambda x: x['non_embed_params'])
    
    # 定义颜色和标记
    styles = {
        'baseline': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'qk_norm': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'attn_gate': {'color': 'red', 'marker': '^', 'linestyle': '--'},
        'value_embed': {'color': 'purple', 'marker': 'd', 'linestyle': '--'}
    }
    
    # 分组绘制
    baseline_params = []
    baseline_ppls = []
    qk_params = []
    qk_ppls = []
    gate_params = []
    gate_ppls = []
    ve_params = []
    ve_ppls = []
    
    for r in all_results:
        params_val = r['non_embed_params']
        ppl_val = r['best_valid_ppl']
        
        if r['config'].get('use_qk_norm'):
            qk_params.append(params_val)
            qk_ppls.append(ppl_val)
        elif r['config'].get('use_attn_gate'):
            gate_params.append(params_val)
            gate_ppls.append(ppl_val)
        elif r['config'].get('use_value_embed'):
            ve_params.append(params_val)
            ve_ppls.append(ppl_val)
        else:
            baseline_params.append(params_val)
            baseline_ppls.append(ppl_val)
    
    if baseline_params:
        ax2.plot(baseline_params, baseline_ppls, 'bo-', linewidth=2, 
                markersize=8, label='Baseline')
    if qk_params:
        ax2.plot(qk_params, qk_ppls, 'gs--', linewidth=2, 
                markersize=8, label='QK Norm')
    if gate_params:
        ax2.plot(gate_params, gate_ppls, 'r^--', linewidth=2, 
                markersize=8, label='Attention Gate')
    if ve_params:
        ax2.plot(ve_params, ve_ppls, 'pd--', linewidth=2, 
                markersize=8, label='Value Embedding')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Non-Embedding Parameters')
    ax2.set_ylabel('Validation Perplexity')
    ax2.set_title('Architectural Variants Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scaling_laws.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scaling laws plot saved to {save_dir}/scaling_laws.png")
    
    # 保存拟合系数
    with open(os.path.join(save_dir, 'scaling_fit.json'), 'w') as f:
        json.dump({
            'slope': coeffs[0],
            'intercept': coeffs[1],
            'r_squared': None  # 可计算
        }, f, indent=2)


def run_positional_analysis(best_model_path, data_loader, device, save_dir):
    """
    研究位置对损失的影响（Context Length Effect）
    """
    print("\n" + "="*70)
    print("Positional Analysis: Loss vs Token Position")
    print("="*70)
    
    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # 需要知道模型配置来重建
    # 这里假设从 checkpoint 中获取
    config = checkpoint.get('args', {})
    
    lm_model = model.CausalLMM(
        vocab_size=len(data_loader.vocabulary),
        dim=config.get('emb_dim', 384),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 24),
        dropout=0.1,
        use_qk_norm=config.get('use_qk_norm', False),
        use_attn_gate=config.get('use_attn_gate', False),
        use_value_embed=config.get('use_value_embed', False)
    )
    lm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不聚合，保留每个位置的损失
    
    # 收集每个位置的损失
    position_losses = {}  # pos -> list of losses
    position_counts = {}  # pos -> count
    
    data_loader.set_valid()
    
    with torch.no_grad():
        while True:
            data_batch, target_batch, end_flag = data_loader.get_batch()
            data_batch = data_batch.to(device)
            target_batch = target_batch.to(device)
            
            logits = lm_model(data_batch)  # [seq_len, batch_size, vocab_size]
            
            # 计算每个位置的损失
            loss_per_token = criterion(
                logits.reshape(-1, logits.size(-1)), 
                target_batch.reshape(-1)
            ).reshape(logits.shape[0], logits.shape[1])  # [seq_len, batch_size]
            
            # 按位置记录
            for pos in range(loss_per_token.shape[0]):
                for batch_idx in range(loss_per_token.shape[1]):
                    token_loss = loss_per_token[pos, batch_idx].item()
                    if token_loss > 0:  # 忽略 padding 或其他无效值
                        if pos not in position_losses:
                            position_losses[pos] = []
                        position_losses[pos].append(token_loss)
            
            if end_flag:
                break
    
    # 计算每个位置的平均损失
    positions = sorted(position_losses.keys())
    avg_losses = [np.mean(position_losses[p]) for p in positions]
    
    # 分组（每 32 个位置一组）
    group_size = 32
    grouped_positions = []
    grouped_losses = []
    
    for i in range(0, max(positions), group_size):
        group_positions = [p for p in positions if i <= p < i + group_size]
        if group_positions:
            group_avg = np.mean([avg_losses[positions.index(p)] for p in group_positions])
            grouped_positions.append(f"{i}-{i+group_size-1}")
            grouped_losses.append(group_avg)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 详细位置损失
    ax1 = axes[0]
    ax1.plot(positions, avg_losses, 'b-', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Loss vs Token Position (Detailed)')
    ax1.grid(True, alpha=0.3)
    
    # 分组损失
    ax2 = axes[1]
    ax2.bar(range(len(grouped_positions)), grouped_losses, color='steelblue', alpha=0.7)
    ax2.set_xticks(range(len(grouped_positions)))
    ax2.set_xticklabels(grouped_positions, rotation=45, ha='right')
    ax2.set_xlabel('Position Group')
    ax2.set_ylabel('Average Loss')
    ax2.set_title('Loss vs Token Position (Grouped by 32)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'positional_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存结果
    pos_results = {
        'positions': positions,
        'avg_losses': avg_losses,
        'grouped_positions': grouped_positions,
        'grouped_losses': grouped_losses
    }
    with open(os.path.join(save_dir, 'positional_results.json'), 'w') as f:
        json.dump(pos_results, f, indent=2)
    
    print(f"Positional analysis saved to {save_dir}/positional_analysis.png")
    
    return pos_results


def save_all_results(results, save_dir):
    """保存所有实验结果到 CSV"""
    csv_path = os.path.join(save_dir, 'partb_all_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Config Name', 'Layers', 'Heads', 'Dim', 'Non-Embed Params', 
                        'Best Valid PPL', 'QK Norm', 'Attn Gate', 'Value Embed'])
        for r in results:
            writer.writerow([
                r['name'],
                r['num_layers'],
                r['num_heads'],
                r['emb_dim'],
                r['non_embed_params'],
                f"{r['best_valid_ppl']:.2f}",
                r['config'].get('use_qk_norm', False),
                r['config'].get('use_attn_gate', False),
                r['config'].get('use_value_embed', False)
            ])
    print(f"All results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Part B: Scaling Laws and Architectural Study')
    parser.add_argument('--data_path', type=str, default='../data/ptb')
    parser.add_argument('--max_sql', type=int, default=256)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./partb_results')
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training, only plot from existing results')
    parser.add_argument('--best_model_path', type=str, default='./checkpoints/best_model.pt',
                        help='Path to best model for positional analysis')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.data_path):
        data_path = args.data_path
    else:
        data_path = os.path.join(script_dir, args.data_path)
    data_path = os.path.normpath(data_path)
    
    print(f"Looking for data at: {data_path}")
    batch_size = {'train': args.train_batch_size, 'valid': args.eval_batch_size}
    data_loader = data.Corpus(data_path, batch_size, args.max_sql)
    vocab_size = len(data_loader.vocabulary)
    print(f"Vocabulary size: {vocab_size}")
    
    # ========== 定义实验配置 ==========
    
    # 1. Scaling Laws 实验：不同规模的 Baseline 模型
    scaling_configs = [
        # 小模型
        {'name': 'tiny', 'num_layers': 4, 'num_heads': 4, 'emb_dim': 128},
        {'name': 'small', 'num_layers': 6, 'num_heads': 6, 'emb_dim': 192},
        {'name': 'base_small', 'num_layers': 6, 'num_heads': 8, 'emb_dim': 256},
        # 中模型
        {'name': 'medium', 'num_layers': 8, 'num_heads': 12, 'emb_dim': 384},
        {'name': 'base', 'num_layers': 8, 'num_heads': 16, 'emb_dim': 512},
        # 大模型
        {'name': 'large', 'num_layers': 10, 'num_heads': 20, 'emb_dim': 640},
        {'name': 'xlarge', 'num_layers': 12, 'num_heads': 24, 'emb_dim': 768},
    ]
    
    # 添加通用训练参数
    common_params = {
        'epochs': 15,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'dropout': 0.1,
        'label_smoothing': 0.1,
        'grad_clip': 0.5,
        'patience': 5
    }
    
    for cfg in scaling_configs:
        cfg.update(common_params)
    
    # 2. 架构变体实验
    # 选择 3 个规模进行对比
    variant_sizes = [
        {'num_layers': 4, 'num_heads': 4, 'emb_dim': 128},   # tiny
        {'num_layers': 6, 'num_heads': 8, 'emb_dim': 256},   # small
        {'num_layers': 8, 'num_heads': 12, 'emb_dim': 384},  # medium
    ]
    
    variant_configs = []
    for size in variant_sizes:
        # Baseline (already in scaling_configs, skip)
        # QK Norm
        cfg_qk = size.copy()
        cfg_qk.update({
            'name': f"qk_norm_{size['emb_dim']}",
            'use_qk_norm': True,
            **common_params
        })
        variant_configs.append(cfg_qk)
        
        # Attention Gate
        cfg_gate = size.copy()
        cfg_gate.update({
            'name': f"attn_gate_{size['emb_dim']}",
            'use_attn_gate': True,
            **common_params
        })
        variant_configs.append(cfg_gate)
        
        # Value Embedding
        cfg_ve = size.copy()
        cfg_ve.update({
            'name': f"value_embed_{size['emb_dim']}",
            'use_value_embed': True,
            **common_params
        })
        variant_configs.append(cfg_ve)
    
    # 合并所有配置
    all_configs = scaling_configs + variant_configs
    
    print("\n" + "="*70)
    print("PART B EXPERIMENT CONFIGURATIONS")
    print("="*70)
    print(f"Total experiments: {len(all_configs)}")
    for cfg in all_configs:
        print(f"  - {cfg['name']}: layers={cfg['num_layers']}, heads={cfg['num_heads']}, "
              f"dim={cfg['emb_dim']}")
    
    # ========== 运行实验 ==========
    results = []
    
    if not args.skip_training:
        for cfg in all_configs:
            result = train_single_config(cfg, data_loader, device, args.save_dir)
            results.append(result)
            
            # 保存中间结果
            save_all_results(results, args.save_dir)
    else:
        # 从已有结果加载
        csv_path = os.path.join(args.save_dir, 'partb_all_results.csv')
        if os.path.exists(csv_path):
            print(f"Loading existing results from {csv_path}")
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                results.append({
                    'name': row['Config Name'],
                    'num_layers': int(row['Layers']),
                    'num_heads': int(row['Heads']),
                    'emb_dim': int(row['Dim']),
                    'non_embed_params': int(row['Non-Embed Params']),
                    'best_valid_ppl': float(row['Best Valid PPL']),
                    'config': {
                        'use_qk_norm': row['QK Norm'] == 'True',
                        'use_attn_gate': row['Attn Gate'] == 'True',
                        'use_value_embed': row['Value Embed'] == 'True'
                    }
                })
    
    # ========== 绘图 ==========
    if results:
        plot_scaling_laws(results, args.save_dir)
        save_all_results(results, args.save_dir)
    
    # ========== 位置分析（使用最佳模型）==========
    if os.path.exists(args.best_model_path):
        pos_results = run_positional_analysis(args.best_model_path, data_loader, 
                                               device, args.save_dir)
    else:
        print(f"Best model not found at {args.best_model_path}, skipping positional analysis")
    
    print("\n" + "="*70)
    print("PART B COMPLETED")
    print(f"Results saved to {args.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()