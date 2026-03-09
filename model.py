import torch
from torch import nn
import einops
import tiktoken

"""TODO: 
    -sensible initializations
    -layer norms
    -residual connections
    -positional encodings
"""


class Transformer(nn.Modul):
    def __init__(self, d_attention, d_qkv, d_ff):
        self.d_attention = d_attention
        self.d_qkv = d_qkv
        self.d_ff = d_ff
        assert self.d_attention % self.d_qkv == 0
        self.n_heads = self.d_attention // self.d_qkv

        self.embed = nn.Params(self.d_attention, self.vocab_size)
        self.


class AttentionHead(nn.Module):
    def __init__(self, d_attention, d_qkv):
        self.Wq = nn.Linear(d_attention, d_qkv)
        self.Wk = nn.Linear(d_attention, d_qkv)
        self.Wv = nn.Linear(d_attention, d_qkv)

    def forward(self, x):
        """
        x: bs, seq_len, d_attention
        """
        q = self.Wq(x) # bs, seq_len, d_qkv
        k = self.Wk(x) # bs, seq_len, d_qkv
        v = self.Wv(x) # bs, seq_len, d_qkv

        bs, seq_len, d_qkv = q.shape
        atten_dot_prods = einops.einsum(q, k, 'bs seq_len d_qkv, bs seq_len d_qkv -> bs seq_len') / torch.sqrt(d_qkv)
        atten_coefs = nn.functional.softmax(atten_dot_prods, dim=1)

        return einops.einsum(v, atten_coefs, 'bs seq_len d_qkv, bs seq_len -> bs d_qkv')
    
class AttentionBlock(nn.Module):
    def __init__(self, d_attention, d_qkv, n_heads):
        self.heads = [AttentionHead(d_attention, d_qkv) for _ in range(n_heads)]
        self.Wo = nn.Linear(d_attention, d_attention)

    def forward(self, x):
        """
        x: batch, seq_len, d_atten
        """
        concat_heads = torch.cat([head(x) for head in self.heads], dim=2)
        return self.Wo(concat_heads)
        

class FeedForwardBlock(nn.Module):
    def __init__(self, d_attention, d_ff):
        self.Wup = nn.Linear(d_attention, d_ff)
        self.Wdp = nn.Linear(d_ff, d_attention)

    def forward(self, x):
        """
        in: (bs, seq_len, d_attention)
        out: (bs, seq_len, d_attention)
        """
        in_shape = x.shape
        
        x = self.Wup(x)
        x = nn.functional.relu(x)
        x = self.Wdp(x)
        x = nn.functional.relu(x)

        assert x.shape == in_shape

        return x