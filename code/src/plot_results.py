# coding: utf-8
"""
Part A: 绘制 Medium 模型的训练曲线
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Medium 模型的实验数据
train_losses = [6.632, 5.925, 5.621, 5.396, 5.210, 5.047, 4.898, 4.754, 4.616, 4.482, 4.353, 4.229]
valid_losses = [6.109, 5.807, 5.647, 5.562, 5.498, 5.488, 5.468, 5.468, 5.488, 5.505, 5.547, 5.582]
train_ppls = [759.04, 374.13, 276.20, 220.55, 183.06, 155.63, 134.02, 116.10, 101.12, 88.39, 77.71, 68.63]
valid_ppls = [450.02, 332.67, 283.53, 260.30, 244.13, 241.77, 236.89, 236.90, 241.84, 245.95, 256.38, 265.57]

epochs = range(1, len(train_losses) + 1)
best_epoch = 7
best_ppl = 236.89

def plot_loss_curves():
    """图1: 损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')
    ax.plot(epochs, valid_losses, 'r-s', linewidth=2, markersize=8, label='Validation Loss')
    
    # 标记最佳点
    ax.plot(best_epoch, valid_losses[best_epoch-1], 'g*', markersize=15, 
            label=f'Best: {valid_losses[best_epoch-1]:.3f}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves (Medium Model)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('parta_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: parta_loss_curves.png")

def plot_ppl_curves():
    """图2: 困惑度曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(epochs, train_ppls, 'b-o', linewidth=2, markersize=8, label='Training PPL')
    ax.semilogy(epochs, valid_ppls, 'r-s', linewidth=2, markersize=8, label='Validation PPL')
    
    # 标记最佳点
    ax.plot(best_epoch, best_ppl, 'g*', markersize=15, 
            label=f'Best Valid PPL = {best_ppl:.2f}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Perplexity (log scale)', fontsize=12)
    ax.set_title('Training and Validation Perplexity Curves (Medium Model)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('parta_ppl_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: parta_ppl_curves.png")

def plot_combined():
    """图3: 组合图（2x2子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Training')
    axes[0, 0].plot(epochs, valid_losses, 'r-s', linewidth=2, markersize=6, label='Validation')
    axes[0, 0].plot(best_epoch, valid_losses[best_epoch-1], 'g*', markersize=15)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 困惑度曲线
    axes[0, 1].semilogy(epochs, train_ppls, 'b-o', linewidth=2, markersize=6, label='Training')
    axes[0, 1].semilogy(epochs, valid_ppls, 'r-s', linewidth=2, markersize=6, label='Validation')
    axes[0, 1].plot(best_epoch, best_ppl, 'g*', markersize=15)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity (log scale)')
    axes[0, 1].set_title('Perplexity Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 损失下降率
    train_reduction = [(train_losses[0] - l) / train_losses[0] * 100 for l in train_losses]
    valid_reduction = [(valid_losses[0] - l) / valid_losses[0] * 100 for l in valid_losses]
    axes[1, 0].plot(epochs, train_reduction, 'b-o', label='Training')
    axes[1, 0].plot(epochs, valid_reduction, 'r-s', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Reduction (%)')
    axes[1, 0].set_title('Loss Reduction Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 过拟合分析
    ratio = [v / t for v, t in zip(valid_ppls, train_ppls)]
    axes[1, 1].bar(epochs, ratio, color='steelblue', alpha=0.7)
    axes[1, 1].axhline(y=1, color='r', linestyle='--', label='No Overfitting (ratio=1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Valid PPL / Train PPL')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('parta_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: parta_combined.png")

def print_summary():
    """打印训练总结"""
    print("\n" + "="*60)
    print("PART A - MEDIUM MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Model: 8 layers, 12 heads, 384 dim")
    print(f"Total parameters: 26,589,712 (26.6M)")
    print(f"Best validation perplexity: {best_ppl:.2f} (Epoch {best_epoch})")
    print(f"Final training perplexity: {train_ppls[-1]:.2f}")
    print(f"Final validation perplexity: {valid_ppls[-1]:.2f}")
    print(f"Overfitting ratio: {valid_ppls[-1] / train_ppls[-1]:.2f}x")
    print("="*60)

def main():
    print("Generating Part A plots for Medium model...")
    plot_loss_curves()
    plot_ppl_curves()
    plot_combined()
    print_summary()
    
    print("\nGenerated files:")
    print("  - parta_loss_curves.png")
    print("  - parta_ppl_curves.png")
    print("  - parta_combined.png")

if __name__ == '__main__':
    main()