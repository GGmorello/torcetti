import numpy as np
from torcetti.core.tensor import Tensor
from torcetti.core.parameter import Parameter
from torcetti.nn.module import Module
from . import functional as F
from torcetti.core.dtype import get_default_dtype


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.track_running_stats:
            default = get_default_dtype()
            self.running_mean = Tensor(np.zeros(num_features, dtype=default))
            self.running_var = Tensor(np.ones(num_features, dtype=default))
        else:
            self.running_mean = None
            self.running_var = None
        
        if self.affine:
            default = get_default_dtype()
            self.weight = Parameter(np.ones(num_features, dtype=default), requires_grad=True)
            self.bias = Parameter(np.zeros(num_features, dtype=default), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
    
    def forward(self, input):
        if self.training:
            if self.track_running_stats:
                running_mean = self.running_mean
                running_var = self.running_var
            else:
                running_mean = input.mean(axis=0)
                running_var = input.var(axis=0, ddof=0)
        else:
            if self.track_running_stats:
                running_mean = self.running_mean
                running_var = self.running_var
            else:
                running_mean = input.mean(axis=0)
                running_var = input.var(axis=0, ddof=0)
        
        output = F.batch_norm(
            input,
            running_mean,
            running_var,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training
        )
        
        if self.affine:
            output = output * self.weight + self.bias
        
        return output
    
    def extra_repr(self):
        """String representation for debugging."""
        return (
            f'num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, '
            f'affine={self.affine}, track_running_stats={self.track_running_stats}'
        )
