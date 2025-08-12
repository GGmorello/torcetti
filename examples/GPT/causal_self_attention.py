from torcetti.nn.module import Module
from torcetti.nn.attention import MultiheadAttention
from torcetti.nn.dropout import Dropout
from torcetti.core.tensor import Tensor
import torcetti
import numpy as np
import torcetti.nn.functional as F

from dataclasses import dataclass

@dataclass
class KVCache:
    """KV cache for decode mode.

    When preallocated (B, capacity, E), we append keys/values in-place during
    decode and track how many valid positions are used via `used`.

    Prefill mode does not require a cache (pass None).
    This is the only supported cache format - tuple caches are not supported.
    """
    k: Tensor
    v: Tensor
    used: int = 0

    @property
    def len(self):
        return self.used

class CausalSelfAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, batch_first=True):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=batch_first)
        self.resid_dropout = Dropout(dropout)
    
    def forward(self, x, past_kv=None, use_cache=False):
        B, L_q, E = x.shape

        q = self.attn.q_proj(x)
        k_new = self.attn.k_proj(x)
        v_new = self.attn.v_proj(x)

        if past_kv is None:
            k = k_new
            v = v_new
            past_len = 0
        else:
            assert isinstance(past_kv, KVCache), "past_kv must be a KVCache instance"
            k_past, v_past = past_kv.k, past_kv.v
            past_len = past_kv.len
            
            if (
                k_past is not None and v_past is not None and
                k_past.shape[1] >= past_len + L_q and v_past.shape[1] >= past_len + L_q
            ):
                k_past.data[:, past_len:past_len + L_q, :] = k_new.data
                v_past.data[:, past_len:past_len + L_q, :] = v_new.data
                k = k_past[:, :past_len + L_q, :]
                v = v_past[:, :past_len + L_q, :]
                if use_cache:
                    past_kv.used = past_len + L_q
            else:
                if k_past is None or v_past is None:
                    k = k_new
                    v = v_new
                    past_len = 0
                    if use_cache:
                        past_kv.used = L_q
                else:
                    k = torcetti.cat([k_past, k_new], dim=1)
                    v = torcetti.cat([v_past, v_new], dim=1)
                    past_len = k_past.shape[1]
                    if use_cache:
                        past_kv.used = past_len + L_q

        L_k = k.shape[1]


        mask = None
        if past_len == 0:
            mask = np.triu(np.ones((L_q, L_k), dtype=x.data.dtype), k=1) * (-1e9)
        elif L_q == 1 and L_k == past_len + 1:
            mask = None 
        else:
            mask = np.triu(np.ones((L_q, L_k), dtype=x.data.dtype), k=1 + past_len) * (-1e9)

        y, _ = F.multi_head_attention(
            q, k, v,
            num_heads=self.attn.num_heads,
            head_dim=self.attn.head_dim,
            dropout_p=self.attn.dropout,
            out_proj=self.attn.out_proj,
            attn_mask=mask,
            batch_first=True,
            training=self.training,
        )

        y = self.resid_dropout(y)

        if use_cache:
            return y, past_kv
        return y