import numpy as np
from torcetti.core.dtype import get_default_dtype
from collections import defaultdict, deque

from torcetti.autograd import _unbroadcast
from torcetti.core.grad_mode import is_grad_enabled

class Gradient:
    """Lazy gradient that initializes on first access."""
    def __init__(self, tensor):
        self.tensor = tensor
        self._grad = None
    
    def __iadd__(self, other):
        if self.tensor.requires_grad:
            if self._grad is None:
                self._grad = np.zeros_like(self.tensor.data, dtype=self.tensor.data.dtype)
            self._grad += other
        return self
    
    def __bool__(self):
        return self._grad is not None
    
    def __eq__(self, other):
        if other is None:
            return self._grad is None
        return np.array_equal(self._grad, other)
    
    @property
    def data(self):
        return self._grad
    
    @property
    def shape(self):
        """Expose gradient shape for convenience in tests and debugging.

        Returns None if the gradient has not been materialized yet.
        """
        return None if self._grad is None else self._grad.shape
    
    def copy(self):
        """Return a copy of the gradient data."""
        return self._grad.copy() if self._grad is not None else None
    
    def __repr__(self):
        return f"Gradient({self._grad})"

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=None, _children=(), _op=''):
        # Adjust requires_grad based on global grad mode for non-leaf tensors
        if _children and not is_grad_enabled():
            requires_grad = False

        if isinstance(data, Tensor):
            target_dtype = dtype if dtype is not None else data.data.dtype
            self.data = data.data.astype(target_dtype)
            self.requires_grad = requires_grad
        else:
            if dtype is None:
                if hasattr(data, 'dtype'):
                    dtype = data.dtype
                else:
                    dtype = get_default_dtype()
            self.data = np.array(data, dtype=dtype)
            self.requires_grad = requires_grad
        
        self.grad = Gradient(self)

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._size = self.data.size
    

    @staticmethod
    def _to_tensor(data, dtype=None):
        if isinstance(data, Tensor):
            return data
        return Tensor(data, dtype=dtype)

    @staticmethod
    def _promote_types(data1, dtype1, data2, dtype2):
        """
        Determines the target dtype based on PyTorch's type promotion rules
        and casts the input data arrays to that dtype.
        """
        if np.issubdtype(dtype1, np.floating) and np.issubdtype(dtype2, np.integer):
            target_dtype = dtype1
        elif np.issubdtype(dtype2, np.floating) and np.issubdtype(dtype1, np.integer):
            target_dtype = dtype2
        else:
            # For other cases (float+float, int+int, or complex), use NumPy's result_type
            target_dtype = np.result_type(dtype1, dtype2)

        return data1.astype(target_dtype), data2.astype(target_dtype), target_dtype

    def backward(self, grad=None, retain_graph=False):
        if getattr(self, "_graph_freed", False):
            raise RuntimeError("Graph already freed")

        users = defaultdict(int)

        visited = set()

        def collect(node):
            if node in visited:
                return
            visited.add(node)
            for p in node._prev:
                users[p] += 1
                collect(p)

        collect(self)

   
        for n in visited:
            if n is self or n._prev:
                n.grad._grad = None

        self.grad += grad if grad is not None else np.ones_like(self.data)

        q = deque([self])

        while q:
            v = q.popleft()

            if v is not self and users[v]:
                continue

            if v.requires_grad:
                v._backward()

            has_parents = bool(v._prev)
            
            for p in v._prev:
                users[p] -= 1
                if users[p] == 0:
                    q.append(p)

            if not retain_graph and v is not self and has_parents:
                v.grad._grad = None
                v._prev.clear()
                v._backward = lambda: None

        if not retain_graph:
            self._graph_freed = True

            

    def __add__(self, other):
        other = Tensor._to_tensor(other) 

        data1_promoted, data2_promoted, target_dtype = Tensor._promote_types(
            self.data, self.dtype, other.data, other.dtype
        )

        out_data = data1_promoted + data2_promoted
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(self, other), _op='+')

        def _backward():
            self.grad += _unbroadcast(out.grad.data, self.data.shape)
            other.grad += _unbroadcast(out.grad.data, other.data.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = Tensor._to_tensor(other) 

        data1_promoted, data2_promoted, target_dtype = Tensor._promote_types(
            self.data, self.dtype, other.data, other.dtype
        )

        out_data = data1_promoted * data2_promoted
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(self, other), _op='*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad.data, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad.data, other.data.shape)
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = Tensor._to_tensor(other)

        a_promoted, b_promoted, target_dtype = Tensor._promote_types(
            self.data, self.dtype, other.data, other.dtype
        )

        out_data = a_promoted ** b_promoted
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(self, other), _op='**')

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(b_promoted * (a_promoted ** (b_promoted - 1)) * out.grad.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(np.log(np.abs(a_promoted) + 1e-9) * (a_promoted ** b_promoted) * out.grad.data, other.data.shape)
        out._backward = _backward
        return out
   
    def dot(self, other):
        other = Tensor._to_tensor(other)

        data1_promoted, data2_promoted, target_dtype = Tensor._promote_types(
            self.data, self.dtype, other.data, other.dtype
        )

        out_data = np.matmul(data1_promoted, data2_promoted)
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(self, other), _op='dot')
        
        def _backward():
            A = data1_promoted
            B = data2_promoted
            G = out.grad.data

            # Special case: vector dot product (k,) x (k,) -> ()
            if A.ndim == 1 and B.ndim == 1:
                if self.requires_grad:
                    self.grad += _unbroadcast(B * G, self.shape)
                if other.requires_grad:
                    other.grad += _unbroadcast(A * G, other.shape)
                return

            A_mat = A
            B_mat = B
            G_mat = G

            a_added = False
            b_added = False
            if A_mat.ndim == 1:
                A_mat = A_mat.reshape(1, A_mat.shape[0])
                a_added = True
            if B_mat.ndim == 1:
                B_mat = B_mat.reshape(B_mat.shape[0], 1)
                b_added = True

            # Compute raw grads per matmul rules
            grad_A_mat = np.matmul(G_mat, np.swapaxes(B_mat, -1, -2))
            grad_B_mat = np.matmul(np.swapaxes(A_mat, -1, -2), G_mat)

            if a_added:
                grad_A_mat = np.squeeze(grad_A_mat, axis=-2)
            if b_added:
                grad_B_mat = np.squeeze(grad_B_mat, axis=-1)

            if self.requires_grad:
                self.grad += _unbroadcast(grad_A_mat, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(grad_B_mat, other.shape)

        out._backward = _backward
        return out


    def mean(self, dim=None, keepdim=False):
        axis = dim
        keepdims = keepdim
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), self.requires_grad, _children=(self,), _op='mean')
        def _backward():
            if axis is None:
                self.grad += np.full_like(self.data, 1/self.data.size) * out.grad.data
            else:
                grad = out.grad.data
                
                # Handle both single axis (int) and multiple axes (tuple)
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                
                # Normalize negative indices and calculate total number of reduced elements
                normalized_axes = tuple(ax % len(self.data.shape) for ax in axes)
                N = 1
                for ax in normalized_axes:
                    N *= self.data.shape[ax]
                
                # Expand dimensions if keepdims=False
                if not keepdims:
                    for ax in sorted(normalized_axes):
                        grad = np.expand_dims(grad, ax)
                
                # Scale gradient by 1/N and broadcast to original shape
                self.grad += np.broadcast_to(grad / N, self.data.shape)
        out._backward = _backward
        return out

    def sum(self, dim=None, keepdim=False):
        axis = dim
        keepdims = keepdim
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), self.requires_grad, _children=(self,), _op='sum')
        def _backward():
            grad = out.grad.data
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out
    
    def max(self, dim=None, keepdim=False):
        axis = dim
        keepdims = keepdim
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='max')
        def _backward():
            max_val = out.data
            if axis is not None and not keepdims:
                max_val = np.expand_dims(out.data, axis)
            mask = (self.data == max_val).astype(float)
            mask /= np.sum(mask, axis=axis, keepdims=True) + 1e-9
            grad = out.grad.data
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += mask * grad
        out._backward = _backward
        return out

    def min(self, dim=None, keepdim=False):
        axis = dim
        keepdims = keepdim
        out_data = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='min')

        def _backward():
            min_val = out.data
            if axis is not None and not keepdims:
                min_val = np.expand_dims(out.data, axis)
            mask = (self.data == min_val).astype(float)
            mask /= np.sum(mask, axis=axis, keepdims=True) + 1e-9
            grad = out.grad.data
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += mask * grad

        out._backward = _backward
        return out

    def prod(self, dim=None, keepdim=False):
        axis = dim
        keepdims = keepdim
        out_data = np.prod(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='prod')

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad.data
            if axis is None:
                total_prod = out.data
                contrib = grad * total_prod / (self.data + 1e-12)
                self.grad += contrib
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis)
                    prod_expanded = np.expand_dims(out.data, axis)
                else:
                    prod_expanded = out.data
                contrib = grad * prod_expanded / (self.data + 1e-12)
                self.grad += contrib

        out._backward = _backward
        return out

    def argmax(self, dim=None, keepdim=False):
        axis = dim
        idx = np.argmax(self.data, axis=axis)
        if keepdim and axis is not None:
            idx = np.expand_dims(idx, axis=axis)
        return Tensor(idx, requires_grad=False, dtype=np.int64)

    def argmin(self, dim=None, keepdim=False):
        axis = dim
        idx = np.argmin(self.data, axis=axis)
        if keepdim and axis is not None:
            idx = np.expand_dims(idx, axis=axis)
        return Tensor(idx, requires_grad=False, dtype=np.int64)

    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, _children=(self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad.data
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, _children=(self,), _op='log')
        def _backward():
            self.grad += (1 / self.data) * out.grad.data
        out._backward = _backward
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), self.requires_grad, _children=(self,), _op='sqrt')
        def _backward():
            self.grad += (0.5 / np.sqrt(self.data)) * out.grad.data
        out._backward = _backward
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), self.requires_grad, _children=(self,), _op='sin')
        def _backward():
            self.grad += np.cos(self.data) * out.grad.data
        out._backward = _backward
        return out

    def cos(self):
        out = Tensor(np.cos(self.data), self.requires_grad, _children=(self,), _op='cos')
        def _backward():
            self.grad += -np.sin(self.data) * out.grad.data
        out._backward = _backward
        return out

    # ---------- Element-wise operations ----------

    def abs(self):
        out_data = np.abs(self.data)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='abs')

        def _backward():
            self.grad += np.sign(self.data) * out.grad.data

        out._backward = _backward
        return out

    def clamp(self, min=None, max=None):
        out_data = np.clip(self.data, a_min=min, a_max=max)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='clamp')

        def _backward():
            mask = np.ones_like(self.data, dtype=self.data.dtype)
            if min is not None:
                mask = mask * (self.data > min)
            if max is not None:
                mask = mask * (self.data < max)
            self.grad += mask * out.grad.data

        out._backward = _backward
        return out

    @staticmethod
    def where(condition, x, y):
        if isinstance(condition, Tensor):
            condition_arr = condition.data.astype(bool)
        else:
            condition_arr = np.asarray(condition, dtype=bool)

        x = Tensor._to_tensor(x)
        y = Tensor._to_tensor(y)

        out_data = np.where(condition_arr, x.data, y.data)
        requires_grad = x.requires_grad or y.requires_grad
        out = Tensor(out_data, requires_grad, _children=(x, y), _op='where')

        def _backward():
            if x.requires_grad:
                x.grad += condition_arr.astype(x.data.dtype) * out.grad.data
            if y.requires_grad:
                y.grad += (~condition_arr).astype(y.data.dtype) * out.grad.data

        out._backward = _backward
        return out

    def floor(self):
        out = Tensor(np.floor(self.data), self.requires_grad, _children=(self,), _op='floor')

        def _backward():
            # derivative zero almost everywhere
            pass

        out._backward = _backward
        return out

    def ceil(self):
        out = Tensor(np.ceil(self.data), self.requires_grad, _children=(self,), _op='ceil')

        def _backward():
            pass

        out._backward = _backward
        return out

    def round(self):
        out = Tensor(np.round(self.data), self.requires_grad, _children=(self,), _op='round')

        def _backward():
            pass

        out._backward = _backward
        return out

    def var(self, dim=None, keepdim=False, ddof=0):
        axis = dim
        keepdims = keepdim
        out = Tensor(np.var(self.data, axis=axis, keepdims=keepdims, ddof=ddof), self.requires_grad, _children=(self,), _op='var')
        def _backward():
            if axis is None:
                mean_val = np.mean(self.data)
                N = self.data.size
                grad = (2.0 / (N - ddof)) * (self.data - mean_val) * out.grad.data
            else:
                mean_val = np.mean(self.data, axis=axis, keepdims=True)
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis
                
                normalized_axes = tuple(ax % len(self.data.shape) for ax in axes)
                N = 1
                for ax in normalized_axes:
                    N *= self.data.shape[ax]
                
                upstream_grad = out.grad.data
                if not keepdims:
                    for ax in sorted(normalized_axes):
                        upstream_grad = np.expand_dims(upstream_grad, ax)
                
                if N - ddof <= 0:
                    raise ValueError(f"Degrees of freedom <= 0 for variance gradient computation. N={N}, ddof={ddof}")
                grad = (2.0 / (N - ddof)) * (self.data - mean_val) * upstream_grad
            self.grad += grad
        out._backward = _backward
        return out

    def __len__(self):
        return len(self.data)

    @property
    def T(self):
        out = Tensor(self.data.T, self.requires_grad, _children=(self,), _op='T')
        def _backward():
            self.grad += out.grad.data.T
        out._backward = _backward
        return out


    def __matmul__(self, other):
        return self.dot(other)

    def permute(self, *dims):
        """Permutes the dimensions of this tensor."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        
        out_data = self.data.transpose(dims)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='permute')
        
        def _backward():
            inv_dims = tuple(np.argsort(dims))
            self.grad += out.grad.data.transpose(inv_dims)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __invert__(self):
        """Allows using the ~ operator for boolean negation."""
        if self.data.dtype != bool:
            raise TypeError("The ~ operator is only supported for boolean tensors")
        
        out = Tensor(~self.data, requires_grad=False, _children=(), _op='~')
        return out

    def __rsub__(self, other):
        other = Tensor._to_tensor(other)
        return other - self

    def __truediv__(self, other):
        other = Tensor._to_tensor(other)

        a_promoted, b_promoted, target_dtype = Tensor._promote_types(
            self.data, self.dtype, other.data, other.dtype
        )

        out_data = a_promoted / b_promoted
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad.data / b_promoted, self.data.shape)
            if other.requires_grad:
                self_over_b2 = -out.grad.data * a_promoted / (b_promoted ** 2)
                other.grad += _unbroadcast(self_over_b2, other.data.shape)
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        other = Tensor._to_tensor(other)
        # other / self
        a_promoted, b_promoted, target_dtype = Tensor._promote_types(
            other.data, other.dtype, self.data, self.dtype
        )

        out_data = a_promoted / b_promoted
        out = Tensor(out_data, self.requires_grad or other.requires_grad, dtype=target_dtype, _children=(other, self), _op='r/')

        def _backward():
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad.data / b_promoted, other.data.shape)
            if self.requires_grad:
                other_over_b2 = -out.grad.data * a_promoted / (b_promoted ** 2)
                self.grad += _unbroadcast(other_over_b2, self.data.shape)
        out._backward = _backward
        return out

    
    def __le__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data <= other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='<=')
    
    def __lt__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data < other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='<')
    
    def __gt__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data > other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='>')
    
    def __ge__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data >= other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='>=')
    
    def __eq__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data == other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='==')
    
    def __ne__(self, other):
        other = Tensor._to_tensor(other)
        out_data = self.data != other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='!=')
        
    def __and__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.data.dtype != bool or other.data.dtype != bool:
            raise TypeError("Logical AND (&) is only supported for boolean tensors")
        out_data = self.data & other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='&')
    
    def __or__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if self.data.dtype != bool or other.data.dtype != bool:
            raise TypeError("Logical OR (|) is only supported for boolean tensors")
        out_data = self.data | other.data
        return Tensor(out_data, requires_grad=False, dtype=bool, _children=(), _op='|')

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self._size


    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            infer_idx = shape.index(-1)
            known = 1
            for dim in shape:
                if dim != -1:
                    known *= dim
            inferred = int(self.data.size // known)
            shape = list(shape)
            shape[infer_idx] = inferred
            shape = tuple(shape)

        out = Tensor(self.data.reshape(shape), self.requires_grad, _children=(self,), _op='reshape')

        def _backward():
            self.grad += out.grad.data.reshape(self.data.shape)
        out._backward = _backward
        return out

    def view(self, *shape):
        """Alias for reshape."""
        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1):
        original_shape = self.data.shape
        rank = len(original_shape)
        if rank == 0:
            return self
        if start_dim < 0:
            start_dim = rank + start_dim
        if end_dim < 0:
            end_dim = rank + end_dim
        if start_dim >= rank:
            return self
        if end_dim < start_dim:
            # Match torch behavior: if end_dim < start_dim, no-op
            return self
        new_shape = original_shape[:start_dim] + (-1,) + original_shape[end_dim+1:]
        return self.reshape(*new_shape)


    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        elif len(repeats) == 1:
            repeats = (repeats[0],)
        data = np.tile(self.data, repeats)
        out = Tensor(data, self.requires_grad, _children=(self,), _op='repeat')

        def _backward():
            grad = out.grad.data
            reshape_dims = []
            for rep, dim in zip(repeats, self.data.shape):
                reshape_dims.extend([rep, dim])
            grad = grad.reshape(reshape_dims)
            # Sum over all repeat axes simultaneously (even indices)
            sum_axes = tuple(range(0, len(reshape_dims), 2))
            grad = grad.sum(axis=sum_axes)
            self.grad += grad
        out._backward = _backward
        return out


    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        target_shape = list(shape)
        src_shape = list(self.data.shape)
        if len(target_shape) != len(src_shape):
            raise ValueError("expand: shape rank mismatch")
        for i, (t, s) in enumerate(zip(target_shape, src_shape)):
            if t == -1:
                target_shape[i] = s
            elif s != 1 and t != s:
                raise ValueError(f"expand: cannot expand dim {i} of size {s} to size {t}")

        out_data = np.broadcast_to(self.data, tuple(target_shape))
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='expand')

        def _backward():
            self.grad += _unbroadcast(out.grad.data, self.data.shape)

        out._backward = _backward
        return out

    def unsqueeze(self, dim):
        out_data = np.expand_dims(self.data, axis=dim)
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='unsqueeze')

        def _backward():
            self.grad += np.squeeze(out.grad.data, axis=dim)

        out._backward = _backward
        return out

    def squeeze(self, dim=None):
        if dim is not None:
            if self.data.shape[dim] != 1:
                return self  # nothing to do
            out_data = np.squeeze(self.data, axis=dim)
        else:
            out_data = np.squeeze(self.data, axis=tuple(i for i,s in enumerate(self.data.shape) if s==1))

        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='squeeze')

        def _backward():
            grad = out.grad.data
            if dim is None:
                target_shape = self.data.shape
                grad = grad.reshape(target_shape)
            else:
                grad = np.expand_dims(grad, axis=dim)
            self.grad += grad

        out._backward = _backward
        return out

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def detach(self):
        return Tensor(self.data, requires_grad=False, dtype=self.dtype)

    def detach_(self):
        self.requires_grad = False
        self._prev = set()
        self._backward = lambda: None
        return self

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, _children=(self,), _op='getitem')

        def _backward():
            grad_for_self = np.zeros_like(self.data)
            np.add.at(grad_for_self, idx, out.grad.data)
            self.grad += grad_for_self
        out._backward = _backward
        return out

    def take(self, indices):

        if hasattr(indices, 'data'):
            indices_array = indices.data
        else:
            indices_array = np.asarray(indices)
            
        indices_array = indices_array.astype(np.int64)
            
        out_data = np.take(self.data, indices_array)  # flatten semantics
        out = Tensor(out_data, self.requires_grad, _children=(self,), _op='take')
        
        def _backward():
            grad_for_self = np.zeros_like(self.data)
            flat_grad = np.zeros(self.data.size, dtype=self.data.dtype)
            np.add.at(flat_grad, indices_array, out.grad.data.ravel())
            grad_for_self = flat_grad.reshape(self.data.shape)
            self.grad += grad_for_self
            
        out._backward = _backward
        return out

    def __array__(self, dtype=None):
        """Numpy array interface to allow np.asarray(Tensor)."""
        if dtype is not None:
            return self.data.astype(dtype)
        return self.data

    def zero_grad(self):
        """Zero the gradient of this tensor."""
        if self.requires_grad and self.grad:
            self.grad._grad = None

