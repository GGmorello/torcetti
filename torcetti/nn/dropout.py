
from torcetti.nn.module import Module
from . import functional

class Dropout(Module):
    def __init__(self, p=0.5): 
        self.p = p
        super().__init__()

    def forward(self, x):
        return functional.dropout(x, p=self.p, training=self.training)