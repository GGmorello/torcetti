import numpy as np
from .base import Optimizer

class RMSprop(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'centered': centered
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            alpha = group['alpha']
            eps = group['eps']
            lr = group['lr']
            momentum = group['momentum']
            centered = group['centered']
            for param in group['params']:
                grad = self._get_gradient(param, group)
                if grad is None:
                    continue
                
                grad = self._apply_weight_decay(grad, param, group)
                
                if param not in self.state:
                    self.state[param] = {
                        'buffer': np.zeros_like(param.data),
                        'square_avg': np.zeros_like(param.data),
                    }
                    if centered:
                        self.state[param]['grad_avg']= np.zeros_like(param.data)

                
                state = self.state[param]
                buffer = state['buffer']
                square_avg = state['square_avg']

                square_avg *= alpha 
                square_avg += (1 - alpha) * grad**2
                v_hat = square_avg
                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg *= alpha 
                    grad_avg += (1 - alpha) * grad
                    v_hat = square_avg - (grad_avg**2)
                if momentum > 0:
                    buffer *= momentum 
                    buffer += grad / (np.sqrt(v_hat) + eps)
                    param.data -= lr * buffer
                else:
                    param.data -= lr*grad/(np.sqrt(v_hat)+eps)