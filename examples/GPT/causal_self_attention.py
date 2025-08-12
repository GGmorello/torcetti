from torcetti.nn.module import Module
from torcetti.nn.attention import MultiheadAttention
from torcetti.nn.dropout import Dropout
from torcetti.core.tensor import Tensor
import numpy as np


class CausalSelfAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
        self.resid_dropout = Dropout(dropout)
    
    def forward(self, x):
        b, s, d = x.shape
        mask = np.triu(np.ones((s, s), dtype=x.data.dtype), k=1) * -1e9
        y, _ = self.attn(x, x, x, attn_mask=mask)
        return self.resid_dropout(y)
