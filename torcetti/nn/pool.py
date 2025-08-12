from torcetti.nn.module import Module
from . import functional as F

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding) 