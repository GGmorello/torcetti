from torcetti.nn.module import Module
from collections import OrderedDict

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules = []
        if modules is not None:
            for module in modules:
                self.add_module(module)

    def add_module(self, module):
        self.modules.append(module)


    def __len__(self):
        return len(self.modules)
    
    def __getitem__(self, idx):
        return self.modules[idx]
    
    def __setitem__(self, idx, module):
        self.modules[idx] = module
    
    def __iter__(self):
        return iter(self.modules)
    
    def append(self, module):
        self.modules.append(module)
    
    def extend(self, modules):
        self.modules.extend(modules)
    
    def insert(self, idx, module):
        self.modules.insert(idx, module)
    
    def pop(self, idx=-1):
        return self.modules.pop(idx)
    
    def remove(self, module):
        self.modules.remove(module)
    
    def clear(self):
        self.modules.clear()
    
    def parameters(self):
        for module in self.modules:
            yield from module.parameters()
    
    def named_parameters(self, prefix='', recurse=True):
        for idx, module in enumerate(self.modules):
            if module is not None:
                yield from module.named_parameters(prefix + f"{idx}.", recurse)
    
    def train(self, mode=True):
        """Set training mode for all modules."""
        self.training = mode
        for module in self.modules:
            module.train(mode)
        return self

    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        
        for idx, module in enumerate(self.modules):
            module.state_dict(destination, prefix + f"{idx}.")
        return destination

class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules = {}
        if modules is not None:
            for name, module in modules.items():
                self.add_module(name, module)
                
    def add_module(self, name, module):
        self.modules[name] = module
        super().__setattr__(name, module)
    
    def parameters(self):
        for module in self.modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix='', recurse=True):
        for name, module in self.modules.items():
            yield from module.named_parameters(prefix + name + '.', recurse)
    
    def train(self, mode=True):
        self.training = mode
        for module in self.modules.values():
            module.train(mode)
        return self

    def update(self, modules):
        for name, module in modules.items():
            self.add_module(name, module)
    
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.add_module(name, value)
        else:
            super().__setattr__(name, value)
    
    def __getitem__(self, name):
        return self.modules[name]
    
    def __setitem__(self, name, module):
        self.modules[name] = module
        super().__setattr__(name, module)
    
    def __delitem__(self, name):
        del self.modules[name]
        super().__delattr__(name)
    
    def __iter__(self):
        return iter(self.modules.values())
    
    def __len__(self):
        return len(self.modules)
    
    def __contains__(self, name):
        return name in self.modules
    
    def values(self):
        return self.modules.values()
    
    def items(self):
        return self.modules.items()
    
    def keys(self):
        return self.modules.keys() 

    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()

        for name, module in self.modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')

        return destination


    def __getattr__(self, item):
        if 'modules' in self.__dict__ and item in self.__dict__['modules']:
            return self.__dict__['modules'][item]
        raise AttributeError(item)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = modules

    def forward(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x
        
    def state_dict(self, destination=None, prefix=''):
        from collections import OrderedDict
        if destination is None:
            destination = OrderedDict()
        
        for idx, module in enumerate(self.layers):
            if module is not None:
                module.state_dict(destination, prefix + f"{idx}.")
        return destination
    
    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()
    
    def named_parameters(self, prefix='', recurse=True):
        for idx, module in enumerate(self.layers):
            if module is not None:
                yield from module.named_parameters(prefix + f"{idx}.", recurse)
    
    def train(self, mode=True):
        """Set training mode for all layers."""
        self.training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

