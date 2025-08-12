from torcetti.nn.module import Module
from torcetti.core.parameter import Parameter
from torcetti.core.factories import ones, zeros
from . import functional as F


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, *, elementwise_affine: bool = True, weight: bool | None = True, bias: bool | None = True):
        """See class docstring for details."""
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if not elementwise_affine:
            create_weight = False
            create_bias = False
        else:
            create_weight = True if weight is None else bool(weight)
            create_bias = True if bias is None else bool(bias)

        self.weight = Parameter(ones(self.normalized_shape), requires_grad=True) if create_weight else None
        self.bias   = Parameter(zeros(self.normalized_shape), requires_grad=True) if create_bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)