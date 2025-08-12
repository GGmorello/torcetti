import numpy as np
from .tensor import Tensor
from .dtype import get_default_dtype

def tensor(data, requires_grad=False, dtype=None):

    if isinstance(data, Tensor):
        if dtype is not None:
            return Tensor(data.data, requires_grad=requires_grad, dtype=dtype)
        return data
    
    if dtype is None:
        if hasattr(data, 'dtype'):
            dtype = data.dtype
        elif isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
            dtype = get_default_dtype()
        else:
            # Fallback for other types
            dtype = get_default_dtype()

    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

def full(shape, fill_value, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    data = np.full(shape, fill_value, dtype=dtype if dtype is not None else get_default_dtype())
    return Tensor(data, requires_grad=requires_grad)


def full_like(tensor_like, fill_value, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    data = np.full(tensor_like.shape, fill_value, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def zeros_like(tensor_like, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    return Tensor(np.zeros_like(tensor_like.data, dtype=dtype), requires_grad=requires_grad)


def ones(shape, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def ones_like(tensor_like, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    return Tensor(np.ones_like(tensor_like.data, dtype=dtype), requires_grad=requires_grad)

def arange(start, end=None, step=1, *, dtype=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    data = np.arange(start, end, step, dtype=dtype if dtype is not None else get_default_dtype())
    return Tensor(data, requires_grad=requires_grad)


def linspace(start, end, steps, *, dtype=None, requires_grad=False):
    data = np.linspace(start, end, steps, dtype=dtype if dtype is not None else get_default_dtype())
    return Tensor(data, requires_grad=requires_grad)

def eye(n, m=None, *, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    data = np.eye(n, M=m, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def diag(v, k=0, *, dtype=None, requires_grad=False):
    if isinstance(v, Tensor):
        arr = v.data
        dtype = dtype if dtype is not None else v.dtype
    else:
        arr = np.asarray(v, dtype=dtype if dtype is not None else get_default_dtype())
    data = np.diag(arr, k=k)
    return Tensor(data, requires_grad=requires_grad)

def rand(shape, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    data = np.random.rand(*shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def rand_like(tensor_like, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    data = np.random.rand(*tensor_like.shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)


def randn(shape, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    data = np.random.randn(*shape).astype(dtype)
    data = data - data.mean()
    return Tensor(data, requires_grad=requires_grad)


def randn_like(tensor_like, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    data = np.random.randn(*tensor_like.shape).astype(dtype)
    data = data - data.mean()
    return Tensor(data, requires_grad=requires_grad)


def randint(low, high, shape, *, dtype=np.int64, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    data = np.random.randint(low, high, size=shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def normal(shape, *, mean=0.0, std=1.0, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    data = np.random.normal(loc=mean, scale=std, size=shape).astype(dtype)
    return Tensor(data, requires_grad=requires_grad)

def empty(shape, *, dtype=None, requires_grad=False):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = get_default_dtype()
    data = np.empty(shape, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def empty_like(tensor_like, *, dtype=None, requires_grad=False):
    dtype = dtype if dtype is not None else tensor_like.dtype
    data = np.empty_like(tensor_like.data, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def as_tensor(obj, *, dtype=None, requires_grad=False):
    if isinstance(obj, Tensor):
        if dtype is not None:
            return Tensor(obj.data, requires_grad=requires_grad, dtype=dtype)
        return obj
    return tensor(obj, dtype=dtype, requires_grad=requires_grad)


