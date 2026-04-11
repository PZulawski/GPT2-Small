import torch
from torch import nn
import math
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
        self.is_inference = False

        self.embeds = nn.Embedding(self.vocab_size, self.d_attention)
        self.pos_embeds = nn.Embedding(self.max_ctx, self.d_attention)
        self.blocks = nn.ModuleList(
            TransformerBlock(self.d_attention, self.d_qkv, self.d_ff, self.n_heads)
            for _ in range(self.n_layers)
        )
        self.vocab_project = nn.Linear(self.d_attention, self.vocab_size)
        self.unembed_layern_norm = nn.LayerNorm(d_attention)
        
        causal_mask = torch.triu(torch.zeros((self.max_ctx, self.max_ctx)) - torch.inf, diagonal=1)
        causal_mask = torch.stack(
            [causal_mask for _ in range(self.n_heads)],
            dim=-1) 
        self.register_buffer('causal_mask', causal_mask) # seq_q, seq_k -> seq_q, seq_k, n_heads

        pos_embed_range = torch.arange(self.max_ctx)
        self.register_buffer('pos_embed_range', pos_embed_range)

    def forward(self, x):
        """
        in: list of token id's 
        """
        bs, seq_len = x.shape

        # embed the input
        x = self.embeds(x) # bs, seq_len, d_attention

        # get position embeds and add to input
        assert seq_len <= self.max_ctx
        pos_embed = self.pos_embeds(self.pos_embed_range[:seq_len])
        assert x.shape[1:] == pos_embed.shape
        x = x + pos_embed # bs, seq_len, d_attention
        
        # loop sequentially through transformer layers
        for transformer_block in self.blocks:
            x = transformer_block(x, self.causal_mask, self.is_inference)
        assert x.shape == (bs, seq_len, self.d_attention)

        x = self.unembed_layern_norm(x)
        
        # project final embeddings to comput logits over full vocab
        logits = self.vocab_project(x)
        assert logits.shape == (bs, seq_len, self.vocab_size)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_attention, d_qkv, d_ff, n_heads):
        super().__init__()
        self.attention_block = AttentionBlock(d_attention, d_qkv, n_heads)
        self.feed_forward_block = FeedForwardBlock(d_attention, d_ff)

    def forward(self, x, mask, is_inference):
        # process embeddings through attention and feed-forward/MLP layers sequentially
        x = self.attention_block(x, mask, is_inference)
        x = self.feed_forward_block(x)

        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, d_attention, d_qkv, n_heads):
        super().__init__()
        self.Wo = nn.Linear(d_attention, d_attention)
        self.layern_norm = nn.LayerNorm(d_attention)
        self.Wqkv = nn.Linear(d_attention, n_heads * d_qkv * 3, bias=False)
        self.d_qkv = d_qkv
        self.n_heads = n_heads

    def forward(self, x, mask, is_inference):
        """
        Uses fused multi-head attention; Wqkv is the fused Q, K and V martrix 
        x: batch, seq_len, d_atten
        """
        x_res = x               # bs, seq_len, d_atten
        x = self.layern_norm(x) # bs, seq_len, d_atten

        qkv_cat = self.Wqkv(x)  # bs, seq_len, n_head * d_qkv * 3

        # split the result of fused projection into individual head types
        cat_head_width = self.n_heads * self.d_qkv
        q_cat = qkv_cat[..., : cat_head_width]                          # bs, seq_len, n_head * d_qkv
        k_cat = qkv_cat[..., cat_head_width : 2 * cat_head_width]       # bs, seq_len, n_head * d_qkv
        v_cat = qkv_cat[..., 2 * cat_head_width : 3 * cat_head_width]   # bs, seq_len, n_head * d_qkv

        # compute querry-key dot products, scaled by sqrt(self.d_qkv) to retain unit variance
        atten_dot_prods = einops.einsum(
            einops.rearrange(q_cat, 'bs seq_len (n_heads d_qkv) -> bs seq_len n_heads d_qkv', n_heads=self.n_heads),
            einops.rearrange(k_cat, 'bs seq_len (n_heads d_qkv) -> bs seq_len n_heads d_qkv', n_heads=self.n_heads),
            'bs seq_q n_heads d_qkv, bs seq_k n_heads d_qkv -> bs seq_q seq_k n_heads'
        ) / math.sqrt(self.d_qkv) # bs, seq_q, seq_k, n_heads

        # skip masking at inference
        if not is_inference:
            atten_dot_prods += mask

        atten_coefs = nn.functional.softmax(atten_dot_prods, dim=2) # bs, seq_q, seq_k, n_heads

        # compute output embeddings as a weighted sum of input embeddings  
        post_atten_embeds = einops.einsum(
            einops.rearrange(v_cat, 'bs seq_v (n_heads d_qkv) -> bs seq_v n_heads d_qkv', n_heads=self.n_heads),
            atten_coefs,
            'bs seq_k n_heads d_qkv, bs seq_q seq_k n_heads -> bs seq_q d_qkv n_heads'
        )

        return self.Wo(einops.rearrange(post_atten_embeds, 'bs seq d_qkv n_heads -> bs seq (d_qkv n_heads)')) + x_res
        

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
        x_res = x                   # bs, seq_len, d_atten
        x = self.layer_norm(x)      # bs, seq_len, d_atten

        x = self.Wup(x)             # bs, seq_len, d_ff
        x = nn.functional.relu(x)   # bs, seq_len, d_ff
        x = self.Wdp(x)             # bs, seq_len, d_atten

        return x + x_res