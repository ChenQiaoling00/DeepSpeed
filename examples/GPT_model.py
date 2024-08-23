from torch.nn import functional as F
import torch.nn as nn
import torch
import math
from dataclasses import dataclass

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.n_head == 0
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb, bias=config.bias)
        self.c_proj = nn.Linear(config.n_emb, config.n_emb, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, input):
        B, S, H = input.size()
        q, k, v = self.c_attn(input).split(self.n_emb, dim=2)
        q = q.view(B, S, self.n_head, H // self.n_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, H // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, H // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            T = k.size(-2)
            bias = torch.tril(torch.ones(T, T, device=k.device)).view(1, 1, T, T)
            att = att.masked_fill(bias == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, S, H)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_emb, config.n_emb, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        input = self.c_fc(input)
        input = self.gelu(input)
        input = self.c_proj(input)
        out = self.dropout(input)
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_emb, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_emb, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, input):
        input = input + self.attn(self.ln_1(input))
        out = input + self.mlp(self.ln_2(input))
        return out

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    dropout: float = 0.0
    bias: bool = True

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = ModelConfig(**config)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_emb),
            wpe = nn.Embedding(self.config.block_size, self.config.n_emb),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_emb, bias=self.config.bias),
        ))
        self.lm_head = nn.Linear(self.config.n_emb, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        print('number of parameters: %.2fM' % (self.get_num_params() / 1e6))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, s = idx.size()
        assert s <= self.config.block_size, f"can not forward sequence of length {s}, block size only {self.config.block_size}"
        pos = torch.arange(0, s, dtype=torch.long, device=device)
        
        if idx.dtype != torch.long:
            idx = idx.long()
        if idx.min().item() < 0:
            print(f"Warning: idx contains negative values, setting them to 0.")
            idx = torch.clamp(idx, min=0)
            # 检查 idx 是否越界

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb+pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # import pdb;pdb.set_trace()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return loss
