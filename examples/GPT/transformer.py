from torcetti.nn.module import Module
from torcetti.nn.layer_norm import LayerNorm
from examples.GPT.feed_forward import FeedForward
from examples.GPT.causal_self_attention import CausalSelfAttention  


class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        self.ffn = FeedForward(embed_dim, hidden_dim=4*embed_dim)

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x   