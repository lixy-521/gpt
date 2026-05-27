import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    """将张量的后半部分取负，用于 RoPE 的旋转操作"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将 RoPE 应用于 query 和 key
    q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim] 或 [seq_len, head_dim]
    """
    # 确保 cos, sin 是 [1, 1, seq_len, head_dim]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPE(nn.Module):
    """预先计算旋转位置编码的 cos 和 sin 值，维度等于 head_dim"""
    def __init__(self, head_dim, max_seq_len=4096):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # theta_i = 10000^{-2i/d}
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    def _build_cache(self, seq_len):
        if self.cos_cached is not None and self.cos_cached.size(0) >= seq_len:
            return
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # [seq_len, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)             # [seq_len, head_dim]
        self.cos_cached = emb.cos()                         # [seq_len, head_dim]
        self.sin_cached = emb.sin()

    def forward(self, seq_len):
        self._build_cache(seq_len)
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力，可选 QK Norm 和 Attention Gate"""
    def __init__(self, dim, num_heads, dropout=0.1, use_qk_norm=False, use_attn_gate=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.use_attn_gate = use_attn_gate

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        if use_attn_gate:
            self.gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        batch, seq_len, _ = x.shape

        # 线性变换并拆分为多头
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE（rope_cos/rope_sin 形状 [seq_len, head_dim]）
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # 可选 QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 计算注意力分数并应用因果掩码
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is None:
            # 创建因果掩码（下三角为0，上三角为 -inf）
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 可选 Attention Gate
        if self.use_attn_gate:
            gate_value = self.sigmoid(self.gate)
            attn_probs = attn_probs * gate_value

        # 加权求和并合并多头
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        out = self.wo(out)
        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer Decoder 块：Pre‑Norm + 自注意力 + Pre‑Norm + FFN"""
    def __init__(self, dim, num_heads, dropout=0.1, use_qk_norm=False, use_attn_gate=False):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, use_qk_norm, use_attn_gate)
        self.ffn = SwiGLUFFN(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        # Pre‑Norm + Attention + Residual
        attn_out = self.attn(self.norm1(x), rope_cos, rope_sin, attn_mask)
        x = x + self.dropout(attn_out)
        # Pre‑Norm + FFN + Residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class CausalLMM(nn.Module):
    """
    完整的因果语言模型，支持 RoPE、可选架构变体
    """
    def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8, max_seq_len=4096,
                 dropout=0.1, use_qk_norm=False, use_attn_gate=False, use_value_embed=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_value_embed = use_value_embed

        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, dim)
        if use_value_embed:
            self.value_embedding = nn.Embedding(vocab_size, dim)

        # RoPE（维度为 head_dim）
        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_seq_len)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout, use_qk_norm, use_attn_gate)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        input_ids: [seq_len, batch_size]  （来自 data.py）
        """
        # 转换为 [batch, seq_len] 方便处理
        x = input_ids.transpose(0, 1)      # [batch, seq_len]
        seq_len = x.size(1)

        # 词嵌入
        x = self.token_embedding(x)        # [batch, seq_len, dim]

        # 可选：Value Embedding（加到输入上，影响所有层）
        if self.use_value_embed:
            value_bias = self.value_embedding(input_ids.transpose(0, 1))
            x = x + value_bias

        # 获取 RoPE 的 cos, sin，形状 [seq_len, head_dim]
        rope_cos, rope_sin = self.rope(seq_len)

        # 因果掩码（所有层共享）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)

        # 逐层前向
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, attn_mask=causal_mask)

        # 最终层归一化并投影到词表
        x = self.norm(x)                   # [batch, seq_len, dim]
        logits = self.head(x)              # [batch, seq_len, vocab_size]

        # 返回形状 [seq_len, batch, vocab_size] 以便与 target 对齐
        return logits.transpose(0, 1)