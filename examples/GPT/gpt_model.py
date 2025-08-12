from torcetti import nn
import torcetti
from examples.GPT.transformer import TransformerBlock
from examples.GPT.causal_self_attention import KVCache
import numpy as np

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len = 1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(max_seq_len, embed_dim),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]),
            ln_f = nn.LayerNorm(embed_dim),
        ))
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, past_kvs=None, use_cache=False):
        B, L = x.shape

        past_len = 0
        if past_kvs is not None:
            first_kv = past_kvs[0] if len(past_kvs) > 0 else None
            if first_kv is not None:
                assert isinstance(first_kv, KVCache), "past_kvs must contain KVCache instances"
                past_len = first_kv.len

        index = torcetti.arange(past_len + L, dtype=np.int32)
        index = index[-L:]
        token_embeddings = self.transformer.wte(x)
        position_embeddings = self.transformer.wpe(index)
        x = self.transformer.drop(token_embeddings + position_embeddings)

        new_kvs = [] if use_cache else None
        for layer_idx, block in enumerate(self.transformer.h):
            past_kv = past_kvs[layer_idx] if past_kvs is not None else None
            if use_cache:
                x, layer_kv = block(x, past_kv=past_kv, use_cache=True)
                new_kvs.append(layer_kv)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, tuple(new_kvs)
        return logits
