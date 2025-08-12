import numpy as np
from torcetti.core.dtype import get_default_dtype
from .module import Module
from torcetti.core.parameter import Parameter
from . import functional as F


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        
        weight_data = np.random.normal(
            0,
            std,
            (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.weight = Parameter(weight_data.astype(get_default_dtype()), requires_grad=True)
        
        if bias:
            bias_data = np.zeros(out_channels, dtype=get_default_dtype())
            self.bias = Parameter(bias_data, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def parameters(self):
        yield self.weight
        if self.bias is not None:
            yield self.bias
