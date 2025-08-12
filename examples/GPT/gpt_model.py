from torcetti import nn
import torcetti
from examples.GPT.transformer import TransformerBlock
import numpy as np
import torch
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len = 1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(max_seq_len, embed_dim),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]),
            ln_f = nn.LayerNorm(embed_dim),
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        B, L = x.shape

        index = torcetti.arange(L, dtype=np.int32)
        token_embeddings = self.transformer.wte(x)
        position_embeddings = self.transformer.wpe(index)
        x = self.transformer.drop(token_embeddings + position_embeddings)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits
