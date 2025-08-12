from collections import OrderedDict
from torcetti.core.tensor import Tensor
from torcetti.core.parameter import Parameter

class Module:
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()

    def forward(self, x):
        pass

    def _get_skip_attrs(self):
        return {'_parameters', '_buffers', '_modules', 'training'}

    def _should_skip_attr(self, name):
        return name.startswith('_') or name in self._modules or name in self._parameters

    def _iter_non_registered_modules(self):
        from torcetti.nn.containers import ModuleList, ModuleDict
        for name, value in self.__dict__.items():
            if self._should_skip_attr(name):
                continue
            if isinstance(value, (Module, ModuleList, ModuleDict)):
                yield name, value

    def parameters(self):
        yield from (p for p in self._parameters.values() if p is not None)
        yield from (p for m in self._modules.values() if m is not None for p in m.parameters())
        
        for _, module in self._iter_non_registered_modules():
            yield from module.parameters()

    def __setattr__(self, name, value):
        if name.startswith('_') or name == 'training':
            super().__setattr__(name, value)
            return
        
        for registry in ('_parameters', '_buffers', '_modules'):
            if hasattr(self, registry) and name in getattr(self, registry):
                del getattr(self, registry)[name]
        
        if isinstance(value, Parameter):
            if not hasattr(self, '_parameters'):
                super().__setattr__('_parameters', OrderedDict())
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, '_modules'):
                super().__setattr__('_modules', OrderedDict())
            self._modules[name] = value
        
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        
        for _, module in self._iter_non_registered_modules():
            module.train(mode)
        
        return self
    
    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self, destination=None, prefix=''):
        from torcetti.nn.containers import ModuleList
        if destination is None:
            destination = OrderedDict()
        
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.clone()
        
        for name, (buf, persistent) in self._buffers.items():
            if buf is not None and persistent:
                destination[prefix + name] = buf.clone()
        
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        
        for name, value in self._iter_non_registered_modules():
            if isinstance(value, Module):
                value.state_dict(destination, prefix + name + '.')
            elif isinstance(value, ModuleList):
                for idx, child in enumerate(value):
                    if child is not None:
                        child.state_dict(destination, prefix + f"{name}.{idx}.")
        
        return destination

    def _navigate_to_target(self, parts):
        target = self
        for part in parts[:-1]:
            if part.isdigit():
                idx = int(part)
                target = self._get_indexed_target(target, idx, parts)
            else:
                target = getattr(target, part)
        return target

    def _get_indexed_target(self, target, idx, parts):
        if isinstance(target, (list, tuple)) and 0 <= idx < len(target):
            return target[idx]
        elif hasattr(target, 'layers') and 0 <= idx < len(target.layers):
            return target.layers[idx]
        elif isinstance(target, dict) and str(idx) in target:
            return target[str(idx)]
        else:
            for attr_name, attr_value in target.__dict__.items():
                if (isinstance(attr_value, (list, tuple)) and 
                    0 <= idx < len(attr_value) and
                    isinstance(attr_value[idx], Module)):
                    return attr_value[idx]
            raise AttributeError(f"Cannot navigate through index '{idx}' in '{'.'.join(parts)}'")

    def _update_tensor_data(self, current_value, tensor, full_name):
        if isinstance(current_value, Tensor) and isinstance(tensor, Tensor):
            if current_value.shape != tensor.shape:
                raise RuntimeError(f"Error(s) in loading state_dict:\n"
                                 f"        size mismatch for {full_name}: copying a param with shape {tensor.shape} from checkpoint, "
                                 f"the shape in current model is {current_value.shape}.")
            current_value.data = tensor.data.copy()
            return True
        return False

    def load_state_dict(self, state_dict, strict=True):
        from torcetti.core.tensor import Tensor 
        current_keys = set(self.state_dict().keys())
        
        for full_name, tensor in state_dict.items():
            try:
                parts = full_name.split('.')
                target = self._navigate_to_target(parts)
                attr_name = parts[-1]
                current_value = getattr(target, attr_name)
                
                if not self._update_tensor_data(current_value, tensor, full_name):
                    setattr(target, attr_name, tensor.clone() if isinstance(tensor, Tensor) else tensor)
                    
            except (AttributeError, IndexError) as e:
                if strict:
                    raise ValueError(f"Error loading '{full_name}': {e}")

        missing_keys = current_keys - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - current_keys
        
        if strict and (missing_keys or unexpected_keys):
            errors = []
            if missing_keys:
                errors.append(f"Missing key(s) in state_dict: {missing_keys}")
            if unexpected_keys:
                errors.append(f"Unexpected key(s) in state_dict: {unexpected_keys}")
            raise ValueError('\n'.join(errors))
        
        class LoadResult:
            def __init__(self, missing_keys, unexpected_keys):
                self.missing_keys = missing_keys
                self.unexpected_keys = unexpected_keys
        
        return LoadResult(missing_keys, unexpected_keys)

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self._parameters.items():
            if param is not None:
                yield (prefix + name, param)
        
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_parameters(prefix + name + '.', recurse)
            
            for name, module in self._iter_non_registered_modules():
                yield from module.named_parameters(prefix + name + '.', recurse)

    def named_buffers(self, prefix='', recurse=True):
        for name, (buf, persistent) in self._buffers.items():
            if buf is not None:
                yield (prefix + name, buf)
        
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_buffers(prefix + name + '.', recurse)

    def buffers(self, recurse=True):
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is not None and module not in memo:
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    yield from module.named_modules(memo, submodule_prefix)

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def register_buffer(self, name, tensor, persistent=True):
        if tensor is not None and not isinstance(tensor, Tensor):
            raise ValueError(f"Expected a Tensor or None, but got {type(tensor)}")
        
        self._buffers[name] = (tensor, persistent)
        object.__setattr__(self, name, tensor)

    def register_parameter(self, name, param):
        if param is not None and not isinstance(param, Parameter):
            raise ValueError(f"Expected a Parameter or None, but got {type(param)}")
        
        self._parameters[name] = param
        object.__setattr__(self, name, param)
