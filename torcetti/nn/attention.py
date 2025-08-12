from torcetti.nn.module import Module
from torcetti.core.tensor import Tensor
from torcetti.nn.linear import Linear
from . import functional as F


class MultiheadAttention(Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 batch_first: bool = False):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask=None,
                key_padding_mask=None):

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        return F.multi_head_attention(
            q, k, v,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_p=self.dropout,
            out_proj=self.out_proj,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_first=self.batch_first,
            training=self.training,
        )


