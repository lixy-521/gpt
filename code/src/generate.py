# generate.py (增强调试版)
import argparse
import torch
import sys
import os
import json

sys.path.append(os.path.dirname(__file__))
import model
import data

def load_model(checkpoint_path, data_loader, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'args' in checkpoint and checkpoint['args']:
        cfg = checkpoint['args']
        lm_model = model.CausalLMM(
            vocab_size=len(data_loader.vocabulary),
            dim=cfg.get('emb_dim', 256),
            num_layers=cfg.get('num_layers', 4),
            num_heads=cfg.get('num_heads', 4),
            dropout=cfg.get('dropout', 0.1),
        )
        print(f"模型配置: dim={cfg.get('emb_dim')}, layers={cfg.get('num_layers')}, heads={cfg.get('num_heads')}")
    else:
        # 尝试从 config.json 读取
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            lm_model = model.CausalLMM(
                vocab_size=len(data_loader.vocabulary),
                dim=cfg.get('emb_dim', 256),
                num_layers=cfg.get('num_layers', 4),
                num_heads=cfg.get('num_heads', 4),
                dropout=cfg.get('dropout', 0.1),
            )
            print(f"从 config.json 加载配置: dim={cfg.get('emb_dim')}")
        else:
            raise ValueError("无法确定模型配置")
    
    # 过滤 rope 缓存
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if 'rope.cos_cached' not in k and 'rope.sin_cached' not in k}
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model.to(device)
    lm_model.eval()
    return lm_model

def generate(model, data_loader, prompt, max_new_tokens=50, temperature=0.8, top_k=0, device='cpu'):
    word_to_id = data_loader.word_id
    id_to_word = {v: k for k, v in word_to_id.items()}
    
    # 编码 prompt
    words = prompt.lower().split()
    input_ids = [word_to_id.get(w, word_to_id.get('<eos>', 0)) for w in words]
    input_ids = torch.tensor(input_ids).unsqueeze(1).to(device)  # [seq_len, 1]
    generated = input_ids.clone()
    
    print(f"初始 tokens: {[id_to_word.get(idx, '<unk>') for idx in input_ids.squeeze().tolist()]}")
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # 限制上下文长度
            context = generated if generated.size(0) <= 256 else generated[-256:]
            logits = model(context)                     # [seq_len, 1, vocab]
            last_logits = logits[-1, 0, :]              # [vocab]
            
            # 温度缩放
            if temperature > 0:
                last_logits = last_logits / temperature
                probs = torch.softmax(last_logits, dim=-1)
                # top-k 采样（可选）
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    next_token = top_k_indices[torch.multinomial(top_k_probs, 1)]
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            
            next_word = id_to_word.get(next_token.item(), '<unk>')
            print(f"Step {step+1}: predicted token id={next_token.item():5d} -> '{next_word}'")
            
            # 停止条件
            if next_token.item() == word_to_id.get('<eos>', -1):
                print("遇到 <eos>，停止生成")
                break
            
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=0)
    
    # 解码全部输出
    full_tokens = generated.squeeze(1).tolist()
    full_words = [id_to_word.get(t, '<unk>') for t in full_tokens]
    return ' '.join(full_words)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data_path', type=str, default='../data/ptb')
    parser.add_argument('--prompt', type=str, default='the world')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU")
    
    # 加载数据（仅用于词汇表）
    batch_size = {'train': 1, 'valid': 1}
    data_loader = data.Corpus(args.data_path, batch_size, 256)
    print(f"词汇表大小: {len(data_loader.vocabulary)}")
    
    # 加载模型
    print(f"从 {args.checkpoint} 加载模型...")
    lm_model = load_model(args.checkpoint, data_loader, device)
    total_params = sum(p.numel() for p in lm_model.parameters())
    print(f"模型参数: {total_params:,}，已加载\n")
    
    print("="*60)
    print(f"提示词: {args.prompt}")
    print(f"温度: {args.temperature}, Top-K: {args.top_k}")
    print(f"最大长度: {args.max_tokens}")
    print("="*60)
    
    output = generate(lm_model, data_loader, args.prompt, args.max_tokens, args.temperature, args.top_k, device)
    print(f"\n生成的文本:\n{output}")
    print("="*60)

if __name__ == "__main__":
    main()