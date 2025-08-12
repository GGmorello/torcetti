from torcetti.nn.module import Module
from torcetti.nn.linear import Linear
from torcetti.nn.activations import GELU
from torcetti.nn.dropout import Dropout

class FeedForward(Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.1):

        super().__init__()
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, embed_dim)
        self.gelu = GELU()
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear2(self.gelu(self.linear1(x))))