# coding: utf-8
import argparse
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
import json
from datetime import datetime

# 设置 matplotlib 后端，避免显示问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import data
import model

# ------------------------------
# 命令行参数（修改默认值）
# ------------------------------
parser = argparse.ArgumentParser(description='PyTorch ptb Language Model with Plotting')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N', help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, metavar='N', help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256, help='sequence length')
parser.add_argument('--seed', type=int, default=1234, help='set random seed')
parser.add_argument('--num_layers', type=int, default=6, help='number of transformer layers')
parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--emb_dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--grad_clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id')
# 架构变体开关
parser.add_argument('--use_qk_norm', action='store_true', help='use QK Norm')
parser.add_argument('--use_attn_gate', action='store_true', help='use Attention Gate')
parser.add_argument('--use_value_embed', action='store_true', help='use Value Embedding')
# 数据路径
parser.add_argument('--data_path', type=str, default='../data/ptb', help='data directory')
# 检查点
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='save directory')
parser.add_argument('--save_best', action='store_true', default=True)
parser.add_argument('--save_every_epoch', action='store_true', default=True)
parser.add_argument('--load_checkpoint', type=str, default=None)
# 绘图
parser.add_argument('--plot_dir', type=str, default='./plots', help='directory to save plots')
# 调度器类型
parser.add_argument('--scheduler', type=str, default='cosine', 
                    choices=['cosine', 'plateau', 'step', 'none'], help='scheduler type')

args = parser.parse_args()

# 确保 emb_dim 可以被 num_heads 整除
assert args.emb_dim % args.num_heads == 0, "emb_dim must be divisible by num_heads"

# 创建目录
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.plot_dir, exist_ok=True)

# 设置设备
if args.cuda and torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id))
    print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device('cpu')
    print("Using CPU")

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ---------- 加载数据 ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isabs(args.data_path):
    data_path = args.data_path
else:
    data_path = os.path.join(script_dir, args.data_path)
data_path = os.path.normpath(data_path)

print("Looking for data at: {}".format(data_path))
batch_size = {'train': args.train_batch_size, 'valid': args.eval_batch_size}
data_loader = data.Corpus(data_path, batch_size, args.max_sql)

vocab_size = len(data_loader.vocabulary)
print("Vocabulary size: {}".format(vocab_size))

# ---------- 构建模型 ----------
print("\nBuilding model with config:")
print("  num_layers: {}".format(args.num_layers))
print("  num_heads: {}".format(args.num_heads))
print("  emb_dim: {}".format(args.emb_dim))
print("  dropout: {}".format(args.dropout))

lm_model = model.CausalLMM(
    vocab_size=vocab_size,
    dim=args.emb_dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    dropout=args.dropout,
    use_qk_norm=args.use_qk_norm,
    use_attn_gate=args.use_attn_gate,
    use_value_embed=args.use_value_embed
)
lm_model = lm_model.to(device)

total_params = sum(p.numel() for p in lm_model.parameters())
non_embed_params = sum(p.numel() for p in lm_model.parameters() 
                       if 'embedding' not in p.__repr__().lower())
print("Total parameters: {:,}".format(total_params))
print("Non-embedding parameters: {:,}".format(non_embed_params))

# ---------- 优化器和调度器 ----------
optimizer = optim.AdamW(lm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

# 学习率调度器
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("Using CosineAnnealingLR scheduler")
elif args.scheduler == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    print("Using ReduceLROnPlateau scheduler")
elif args.scheduler == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print("Using StepLR scheduler")
else:
    scheduler = None
    print("No learning rate scheduler")

# ---------- 恢复训练（如果需要） ----------
best_valid_ppl = float('inf')
best_valid_loss = float('inf')
start_epoch = 1
train_losses = []
valid_losses = []
train_ppls = []
valid_ppls = []
patience_counter = 0

if args.load_checkpoint:
    print("Loading checkpoint from {}".format(args.load_checkpoint))
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                  if 'rope.cos_cached' not in k and 'rope.sin_cached' not in k}
    lm_model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_valid_ppl = checkpoint.get('best_valid_ppl', float('inf'))
    best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
    train_losses = checkpoint.get('train_losses', [])
    valid_losses = checkpoint.get('valid_losses', [])
    train_ppls = checkpoint.get('train_ppls', [])
    valid_ppls = checkpoint.get('valid_ppls', [])
    patience_counter = checkpoint.get('patience_counter', 0)
    print("Resumed from epoch {}, best valid ppl: {:.2f}".format(start_epoch, best_valid_ppl))

# ---------- 辅助函数 ----------
def save_checkpoint(epoch, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': lm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_ppl': best_valid_ppl,
        'best_valid_loss': best_valid_loss,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_ppls': train_ppls,
        'valid_ppls': valid_ppls,
        'patience_counter': patience_counter,
        'args': vars(args),
    }
    if args.save_every_epoch:
        epoch_path = os.path.join(args.save_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
        torch.save(checkpoint, epoch_path)
        print("Saved checkpoint to {}".format(epoch_path))
    if args.save_best and is_best:
        best_path = os.path.join(args.save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print("Saved best model to {} (valid ppl: {:.2f})".format(best_path, best_valid_ppl))

def save_config():
    config = vars(args)
    config['vocab_size'] = vocab_size
    config['total_params'] = total_params
    config['non_embed_params'] = non_embed_params
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("Saved config to {}".format(os.path.join(args.save_dir, 'config.json')))

def evaluate():
    data_loader.set_valid()
    lm_model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        while True:
            data, target, end_flag = data_loader.get_batch()
            data, target = data.to(device), target.to(device)
            logits = lm_model(data)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target)
            total_loss += loss.item()
            steps += 1
            if end_flag:
                break
    avg_loss = total_loss / steps
    return avg_loss, math.exp(avg_loss)

def train_one_epoch():
    data_loader.set_train()
    lm_model.train()
    total_loss = 0.0
    steps = 0
    
    epoch_start_time = datetime.now()
    
    while True:
        data, target, end_flag = data_loader.get_batch()
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = lm_model(data)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(lm_model.parameters(), args.grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        if steps % 10 == 0:
            print("  Step {}, loss: {:.4f}".format(steps, loss.item()))
        
        if end_flag:
            break
    
    avg_loss = total_loss / steps
    epoch_time = (datetime.now() - epoch_start_time).seconds
    print("  Epoch time: {} seconds".format(epoch_time))
    
    return avg_loss, math.exp(avg_loss)

def plot_curves():
    """绘制训练/验证损失和困惑度曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, valid_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 困惑度曲线（对数坐标）
    axes[0, 1].semilogy(epochs, train_ppls, 'b-o', label='Training Perplexity', linewidth=2, markersize=6)
    axes[0, 1].semilogy(epochs, valid_ppls, 'r-s', label='Validation Perplexity', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity (log scale)')
    axes[0, 1].set_title('Training and Validation Perplexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 标记最佳点
    best_idx = np.argmin(valid_ppls)
    axes[0, 1].plot(best_idx + 1, valid_ppls[best_idx], 'g*', markersize=15, 
                    label='Best PPL={:.2f}'.format(valid_ppls[best_idx]))
    axes[0, 1].legend()
    
    # 损失下降率
    train_loss_reduction = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    valid_loss_reduction = [(valid_losses[0] - loss) / valid_losses[0] * 100 for loss in valid_losses]
    axes[1, 0].plot(epochs, train_loss_reduction, 'b-o', label='Training Loss Reduction', linewidth=2)
    axes[1, 0].plot(epochs, valid_loss_reduction, 'r-s', label='Validation Loss Reduction', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Reduction (%)')
    axes[1, 0].set_title('Loss Reduction Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 1].plot(epochs, [args.lr] * len(epochs), 'g--', label='Initial LR={}'.format(args.lr), linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(args.plot_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved to {}".format(plot_path))

def save_results():
    results = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_ppls': train_ppls,
        'valid_ppls': valid_ppls,
        'best_valid_ppl': best_valid_ppl,
        'best_valid_loss': best_valid_loss,
        'final_train_ppl': train_ppls[-1] if train_ppls else None,
        'final_valid_ppl': valid_ppls[-1] if valid_ppls else None,
        'args': vars(args),
    }
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to {}".format(os.path.join(args.save_dir, 'results.json')))

def print_training_summary():
    """打印训练总结"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("Model config: {} layers, {} heads, {} dim".format(
        args.num_layers, args.num_heads, args.emb_dim))
    print("Total parameters: {:,}".format(total_params))
    print("Non-embedding parameters: {:,}".format(non_embed_params))
    print("Best validation perplexity: {:.2f}".format(best_valid_ppl))
    if best_valid_ppl in valid_ppls:
        best_epoch = valid_ppls.index(best_valid_ppl) + 1
        print("  (Epoch {})".format(best_epoch))
    print("Final validation perplexity: {:.2f}".format(valid_ppls[-1]))
    print("Best validation loss: {:.4f}".format(best_valid_loss))
    print("Final validation loss: {:.4f}".format(valid_losses[-1]))
    
    # 过拟合程度
    overfitting_ratio = valid_ppls[-1] / train_ppls[-1]
    print("Overfitting ratio (valid/train PPL): {:.2f}".format(overfitting_ratio))
    if overfitting_ratio > 2.0:
        print("  Warning: Model may be overfitting. Consider increasing dropout or weight decay.")
    
    # 收敛情况
    if len(valid_losses) > 1:
        loss_improvement = (valid_losses[0] - valid_losses[-1]) / valid_losses[0] * 100
        print("Validation loss improvement: {:.1f}%".format(loss_improvement))
    
    print("Checkpoints saved to: {}".format(args.save_dir))
    print("Plots saved to: {}".format(args.plot_dir))
    print("="*60)

# ---------- 训练循环 ----------
save_config()
print("\n=== Starting Training ===")
print("Training for {} epochs, patience={}".format(args.epochs, args.patience))
print("Learning rate: {}, Weight decay: {}".format(args.lr, args.weight_decay))

for epoch in range(start_epoch, args.epochs + 1):
    print("\n=== Epoch {}/{} ===".format(epoch, args.epochs))
    
    # 训练一个epoch
    train_loss, train_ppl = train_one_epoch()
    
    # 验证
    valid_loss, valid_ppl = evaluate()
    
    # 记录
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_ppls.append(train_ppl)
    valid_ppls.append(valid_ppl)
    
    # 当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    print("Train Loss: {:.4f}, Train PPL: {:.2f}".format(train_loss, train_ppl))
    print("Valid Loss: {:.4f}, Valid PPL: {:.2f}".format(valid_loss, valid_ppl))
    print("Learning Rate: {:.6f}".format(current_lr))
    
    # 更新学习率调度器
    if args.scheduler == 'plateau':
        scheduler.step(valid_loss)
    elif scheduler is not None:
        scheduler.step()
    
    # 检查是否为最佳模型
    is_best = valid_ppl < best_valid_ppl
    if is_best:
        best_valid_ppl = valid_ppl
        best_valid_loss = valid_loss
        patience_counter = 0
        print("  New best model! Valid PPL: {:.2f}".format(best_valid_ppl))
    else:
        patience_counter += 1
        print("  No improvement for {} epoch(s)".format(patience_counter))
    
    # 保存检查点
    save_checkpoint(epoch, is_best)
    
    # 早停检查
    if patience_counter >= args.patience:
        print("\nEarly stopping triggered after epoch {}".format(epoch))
        print("Best validation perplexity: {:.2f}".format(best_valid_ppl))
        break

# 保存最终结果和绘图
save_results()
plot_curves()
print_training_summary()

# 保存 CSV 格式以便查看
import csv
csv_path = os.path.join(args.save_dir, 'training_results.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train Loss', 'Valid Loss', 'Train PPL', 'Valid PPL'])
    for i in range(len(train_losses)):
        writer.writerow([i+1, train_losses[i], valid_losses[i], train_ppls[i], valid_ppls[i]])
print("CSV results saved to {}".format(csv_path))

print("\n=== Training Complete ===")
print("Best validation perplexity: {:.2f}".format(best_valid_ppl))
print("Training curves saved to {}/training_curves.png".format(args.plot_dir))