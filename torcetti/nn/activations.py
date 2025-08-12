from torcetti.nn.module import Module
from . import functional as F

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)

class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.sigmoid(x)

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.tanh(x)

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.softmax(x, dim=self.dim)
 
