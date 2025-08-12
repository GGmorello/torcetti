from torcetti.core.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data, requires_grad=True, dtype=None):
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)
