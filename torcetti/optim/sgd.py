import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov
        }
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            
            for param in group['params']:
                grad = self._get_gradient(param, group)
                if grad is None:
                    continue
                
                grad = self._apply_weight_decay(grad, param, group)
                
                if momentum != 0:
                    param_state = self.state.setdefault(param, {})
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = grad.copy()
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf * momentum + (1 - dampening) * grad
                        param_state['momentum_buffer'] = buf
                    
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                
                param.data = param.data - lr * grad
