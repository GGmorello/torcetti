import numpy as np

class Optimizer:
    
    def __init__(self, params, defaults):
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            self.param_groups = []
            for group in params:
                new_group = defaults.copy()
                new_group.update(group)
                self.param_groups.append(new_group)
        else:
            if hasattr(params, '__iter__') and not isinstance(params, list):
                params = list(params)
            self.param_groups = [{'params': params, **defaults}]
        
        self.state = {}
    
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.zero_grad()
    
    def step(self):
        raise NotImplementedError("Subclasses must implement step()")
    
    def _get_gradient(self, param, group):
        if param.grad and param.grad.data is not None:
            return param.grad.data.copy()
        else:
            weight_decay = group.get('weight_decay', 0)
            if weight_decay == 0:
                return None
            return np.zeros_like(param.data)
    
    def _apply_weight_decay(self, grad, param, group):
        weight_decay = group.get('weight_decay', 0)
        if weight_decay != 0:
            grad += weight_decay * param.data
        return grad 