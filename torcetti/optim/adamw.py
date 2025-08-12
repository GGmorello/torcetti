import numpy as np
from .base import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            amsgrad = group['amsgrad']
            lr = group['lr']
            
            for param in group['params']:
                grad = self._get_gradient(param, group)
                if grad is None:
                    continue
                
                if param not in self.state:
                    self.state[param] = {
                        'step': 0,
                        'exp_avg': np.zeros_like(param.data),
                        'exp_avg_sq': np.zeros_like(param.data)
                    }
                    if amsgrad:
                        self.state[param]['max_exp_avg_sq'] = np.zeros_like(param.data)
                
                state = self.state[param]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                state['step'] += 1
                
                exp_avg *= betas[0]
                exp_avg += (1 - betas[0]) * grad
                
                exp_avg_sq *= betas[1]
                exp_avg_sq += (1 - betas[1]) * grad ** 2
                
                bias_correction1 = 1 - betas[0] ** state['step']
                bias_correction2 = 1 - betas[1] ** state['step']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (np.sqrt(max_exp_avg_sq) / np.sqrt(bias_correction2)) + eps
                else:
                    denom = (np.sqrt(exp_avg_sq) / np.sqrt(bias_correction2)) + eps
                
                step_size = lr / bias_correction1
                
                param.data = param.data - step_size * exp_avg / denom - lr * param.data * weight_decay