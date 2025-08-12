import numpy as np
from torcetti.core.dtype import get_default_dtype

from torcetti.nn.module import Module
from torcetti.core.parameter import Parameter
from torcetti.nn.functional import linear


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            np.random.uniform(-bound, bound, (in_features, out_features)).astype(get_default_dtype()), requires_grad=True
        )
        if bias:
            self.bias = Parameter(
                np.random.uniform(-bound, bound, (out_features,)).astype(get_default_dtype()), requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x):
        return linear(x, self.weight, self.bias)
