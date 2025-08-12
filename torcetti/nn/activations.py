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
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x):
        return F.softmax(x, axis=self.axis)
 
