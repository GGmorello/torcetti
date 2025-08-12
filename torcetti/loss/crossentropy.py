import numpy as np

from torcetti.nn.module import Module
from torcetti.core.tensor import Tensor
from torcetti.core.factories import tensor
from torcetti.nn import functional as F

class CrossEntropyLoss(Module):
    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        
        log_probs_tensor = F.log_softmax(predictions, dim=1)
        
        targets_indices = targets.data.astype(int)
        
        loss_per_sample_data = -log_probs_tensor.data[np.arange(batch_size), targets_indices]
        loss_per_sample = Tensor(loss_per_sample_data, requires_grad=predictions.requires_grad, 
                                _children=(log_probs_tensor,), _op='nll_loss')
        
        def _backward():
            if loss_per_sample.grad:
                grad_contribution = np.zeros_like(log_probs_tensor.data)
                grad_contribution[np.arange(batch_size), targets_indices] = -loss_per_sample.grad.data
                log_probs_tensor.grad += grad_contribution
        
        loss_per_sample._backward = _backward
        
        loss = loss_per_sample.mean()
        
        return loss