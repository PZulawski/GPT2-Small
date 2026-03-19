import torch
from torch import nn
import numpy as np
import einops

class Transformer(nn.Module):
    def __init__(self, d_attention, d_qkv, d_ff, vocab_size, n_layers, max_ctx):
        super().__init__()
        self.max_ctx = max_ctx
        self.d_attention = d_attention
        self.d_qkv = d_qkv
        self.d_ff = d_ff
        assert self.d_attention % self.d_qkv == 0
        self.n_heads = self.d_attention // self.d_qkv
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.embeds = nn.Embedding(self.vocab_size, self.d_attention)
        self.pos_embeds = nn.Embedding(self.max_ctx, self.d_attention)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.d_attention, self.d_qkv, self.d_ff, self.n_heads)
            for _ in range(self.n_layers)
        )
        self.vocab_project = nn.Linear(self.d_attention, self.vocab_size)
        self.causal_mask = None


    def forward(self, x):
        """
        in: list of token id's 
        """
        bs, seq_len = x.shape
        # embed the input
        x = self.embeds(x) # bs, seq_len, d_attention
        # get position embeds and add to input
        assert seq_len <= self.max_ctx
        pos_embed = self.pos_embeds(torch.arange(seq_len).to(x.device))
        assert x.shape[1:] == pos_embed.shape
        x = x + pos_embed # bs, seq_len, d_attention

        if not self.causal_mask:
            self.causel_mask = torch.triu(torch.zeros((self.max_ctx, self.max_ctx)) -torch.inf, diagonal=1).to(x.device)

        # transformer layers
        for transformer_block in self.blocks:
            x = transformer_block(x, self.causel_mask)
        assert x.shape == (bs, seq_len, self.d_attention)
        
        logits = self.vocab_project(x)
        assert logits.shape == (bs, seq_len, self.vocab_size)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_attention, d_qkv, d_ff, n_heads):
        super().__init__()
        self.attention_block = AttentionBlock(d_attention, d_qkv, n_heads)
        self.feed_forward_block = FeedForwardBlock(d_attention, d_ff)

    def forward(self, x, mask):
        x = self.attention_block(x, mask)
        x = self.feed_forward_block(x)

        return x

class AttentionHead(nn.Module):
    def __init__(self, d_attention, d_qkv):
        super().__init__()
        self.Wq = nn.Linear(d_attention, d_qkv, bias=False)
        self.Wk = nn.Linear(d_attention, d_qkv, bias=False)
        self.Wv = nn.Linear(d_attention, d_qkv, bias=False)

    def forward(self, x, mask):
        """
        x: bs, seq_len, d_attention
        """
        q = self.Wq(x) # bs, seq_len, d_qkv
        k = self.Wk(x) # bs, seq_len, d_qkv
        v = self.Wv(x) # bs, seq_len, d_qkv

        bs, seq_len, d_qkv = q.shape
        # dot producst of q's and k's, scaled to retain unitary variance
        atten_dot_prods = einops.einsum(q, k, 'bs seq_q d_qkv, bs seq_k d_qkv -> bs seq_q seq_k') / np.sqrt(d_qkv)
        # causal masking, sets dot products for future tokens to -inf
        atten_dot_prods += mask
        atten_coefs = nn.functional.softmax(atten_dot_prods, dim=2)
        post_attn_embeds = einops.einsum(v, atten_coefs, 'bs seq_k d_qkv, bs seq_q seq_k -> bs seq_q d_qkv')
        
        return post_attn_embeds
    
class AttentionBlock(nn.Module):
    def __init__(self, d_attention, d_qkv, n_heads):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(d_attention, d_qkv) for _ in range(n_heads))
        self.Wo = nn.Linear(d_attention, d_attention)
        self.layern_norm = nn.LayerNorm(d_attention)

    def forward(self, x, mask):
        """
        x: batch, seq_len, d_atten
        """
        x_res = x
        x = self.layern_norm(x)

        x_heads = [head(x, mask) for head in self.heads]
        
        x_concat_heads = torch.cat(x_heads, dim=2)
        assert x_concat_heads.shape[-1] == next(self.Wo.parameters()).shape[0], 'concat heads do not add up to d_attention'

        return self.Wo(x_concat_heads) + x_res
        

class FeedForwardBlock(nn.Module):
    def __init__(self, d_attention, d_ff):
        super().__init__()
        self.Wup = nn.Linear(d_attention, d_ff)
        self.Wdp = nn.Linear(d_ff, d_attention)
        self.layer_norm = nn.LayerNorm(d_attention)

    def forward(self, x):
        """
        in: (bs, seq_len, d_attention)
        out: (bs, seq_len, d_attention)
        """
        x_res = x
        x = self.layer_norm(x)

        x = self.Wup(x)
        x = nn.functional.relu(x)
        x = self.Wdp(x)

        return x + x_res