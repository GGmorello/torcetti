import numpy as np
from torcetti.nn.module import Module
from torcetti.core.tensor import Tensor

class MSE(Module):
    def forward(self, predictions, targets):

        diff = predictions.data - targets.data
        loss_value = np.mean(diff ** 2)

        out = Tensor(
            loss_value,
            requires_grad=predictions.requires_grad,
            _children=(predictions,),
            _op='mse_loss'
        )
        
        def _backward():
            batch_size = predictions.data.size
            predictions.grad += 2 * diff / batch_size
        
        out._backward = _backward
        return out

