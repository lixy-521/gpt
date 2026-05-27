import json
import matplotlib.pyplot as plt
import os

# 读取训练结果
results_path = './checkpoints/results.json'
if not os.path.exists(results_path):
    results_path = '/home/lxy/gpt/code/src/checkpoints/results.json'  # 根据你的实际路径调整

with open(results_path, 'r') as f:
    results = json.load(f)

train_losses = results['train_losses']
valid_losses = results['valid_losses']
train_ppls = results['train_ppls']
valid_ppls = results['valid_ppls']

epochs = range(1, len(train_losses) + 1)

# 创建图形
plt.figure(figsize=(14, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
plt.plot(epochs, valid_losses, 'r-s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 困惑度曲线（对数坐标）
plt.subplot(1, 2, 2)
plt.semilogy(epochs, train_ppls, 'b-o', label='Training Perplexity')
plt.semilogy(epochs, valid_ppls, 'r-s', label='Validation Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity (log scale)')
plt.title('Training and Validation Perplexity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片（尝试不同格式和方式）
try:
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("图片已保存为 training_curves.png")
except Exception as e:
    print(f"保存 PNG 失败: {e}")
    try:
        plt.savefig('training_curves.pdf')
        print("图片已保存为 training_curves.pdf")
    except Exception as e2:
        print(f"保存 PDF 也失败: {e2}")
        # 如果还不成功，尝试设置非交互后端
        import matplotlib
        matplotlib.use('Agg')
        plt.savefig('training_curves.png')
        print("使用 Agg 后端保存成功")

plt.show()  # 如果无图形界面可能会报错，但没关系