import numpy as np

from .tensor import Tensor


def where(condition, x, y):
    return Tensor.where(condition, x, y)


def cat(tensors, dim=0):
    data = np.concatenate([t.data for t in tensors], axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad, _children=tensors, _op='concat')

    if requires_grad:
        def _backward():
            grad = out.grad.data
            split_indices = np.cumsum([t.shape[dim] for t in tensors[:-1]])
            grad_splits = np.split(grad, split_indices, axis=dim)
            for t, g in zip(tensors, grad_splits):
                if t.requires_grad:
                    t.grad += g
        out._backward = _backward

    return out


def stack(tensors, dim=0):
    data = np.stack([t.data for t in tensors], axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad, _children=tensors, _op='stack')

    if requires_grad:
        def _backward():
            grad = np.moveaxis(out.grad.data, dim, 0)
            for idx, t in enumerate(tensors):
                if t.requires_grad:
                    t.grad += grad[idx]
        out._backward = _backward
    return out


def meshgrid(*tensors, indexing='ij'):
    np_arrays = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    grids = np.meshgrid(*np_arrays, indexing=indexing)
    return tuple(Tensor(g, requires_grad=False) for g in grids)


concatenate = cat


def take(tensor, indices, dim=None):
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    if dim is None:
        return tensor.take(indices)
    # Implement torch.take along a dimension is torch.gather-like; our API does not support that.
    # For now, align with torch.take semantics only when dim is None (flatten). If dim provided, raise.
    raise NotImplementedError("take: only dim=None (flattened) is supported, matching torch.take")


def topk(input, k, dim=-1, largest=True, sorted=True):

    x = input if isinstance(input, Tensor) else Tensor(input)
    if k < 1:
        raise ValueError("k must be >= 1")

    ndim = x.data.ndim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim out of range (got {dim} for tensor of dimension {ndim})")

    if k > x.data.shape[dim]:
        raise ValueError(
            f"k ({k}) cannot be greater than the size of dimension {dim} ({x.data.shape[dim]})"
        )

    axis_size = x.data.shape[dim]

    if largest:
        part = np.argpartition(x.data, axis_size - k, axis=dim)
        idx_unsorted = np.take(part, indices=range(axis_size - k, axis_size), axis=dim)
    else:
        part = np.argpartition(x.data, k - 1, axis=dim)
        idx_unsorted = np.take(part, indices=range(0, k), axis=dim)

    vals_unsorted = np.take_along_axis(x.data, idx_unsorted, axis=dim)

    if sorted:
        if largest:
            order = np.argsort(-vals_unsorted, axis=dim)
        else:
            order = np.argsort(vals_unsorted, axis=dim)
        vals = np.take_along_axis(vals_unsorted, order, axis=dim)
        idx = np.take_along_axis(idx_unsorted, order, axis=dim)
    else:
        vals = vals_unsorted
        idx = idx_unsorted

    indices_tensor = Tensor(idx.astype(np.int64), requires_grad=False, dtype=np.int64)

    values_tensor = Tensor(vals, requires_grad=x.requires_grad, _children=(x,), _op='topk')

    if x.requires_grad:
        def _backward():
            upstream = values_tensor.grad.data

            x_data_moved = np.moveaxis(x.data, dim, -1)
            idx_moved = np.moveaxis(idx, dim, -1)
            grad_moved = np.moveaxis(upstream, dim, -1)

            outer_shape = x_data_moved.shape[:-1]
            n = x_data_moved.shape[-1]
            k_ = idx_moved.shape[-1]

            grad_for_x_moved = np.zeros_like(x_data_moved)

            rows = int(np.prod(outer_shape)) if outer_shape else 1
            grad_for_x_flat = grad_for_x_moved.reshape(rows, n)
            idx_flat = idx_moved.reshape(rows, k_)
            grad_flat = grad_moved.reshape(rows, k_)

            for r in range(rows):
                np.add.at(grad_for_x_flat[r], idx_flat[r], grad_flat[r])

            grad_for_x = np.moveaxis(grad_for_x_flat.reshape(*outer_shape, n), -1, dim)
            x.grad += grad_for_x

        values_tensor._backward = _backward

    return values_tensor, indices_tensor


def sum(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.sum(dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.mean(dim=dim, keepdim=keepdim)


def max(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.max(dim=dim, keepdim=keepdim)


def min(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.min(dim=dim, keepdim=keepdim)


def prod(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.prod(dim=dim, keepdim=keepdim)


def argmax(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.argmax(dim=dim, keepdim=keepdim)


def argmin(input, dim=None, keepdim=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.argmin(dim=dim, keepdim=keepdim)


def clamp(input, min=None, max=None):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.clamp(min=min, max=max)


def abs(input):  # noqa: A001
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.abs()


def exp(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.exp()


def log(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.log()


def sqrt(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.sqrt()


def floor(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.floor()


def ceil(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.ceil()


def round(input):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.round()


def var(input, dim=None, keepdim=False, ddof=0):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.var(dim=dim, keepdim=keepdim, ddof=ddof)


def permute(input, *dims):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.permute(*dims)


def reshape(input, *shape):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.reshape(*shape)


def flatten(input, start_dim=0, end_dim=-1):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


def repeat(input, *repeats):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.repeat(*repeats)


def expand(input, *shape):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.expand(*shape)


def unsqueeze(input, dim):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.unsqueeze(dim)


def squeeze(input, dim=None):
    x = input if isinstance(input, Tensor) else Tensor(input)
    return x.squeeze(dim=dim)


def multinomial(input, num_samples=1, replacement=False):
    x = input if isinstance(input, Tensor) else Tensor(input)
    
    probs = x.data / x.data.sum(axis=-1, keepdims=True)
    
    original_shape = probs.shape
    batch_shape = original_shape[:-1]
    vocab_size = original_shape[-1]
    
    probs_2d = probs.reshape(-1, vocab_size)
    batch_size = probs_2d.shape[0]
    
    if replacement:
        indices = np.zeros((batch_size, num_samples), dtype=np.int64)
        for i in range(batch_size):
            indices[i] = np.random.choice(vocab_size, size=num_samples, replace=True, p=probs_2d[i])
    else:
        indices = np.zeros((batch_size, num_samples), dtype=np.int64)
        for i in range(batch_size):
            indices[i] = np.random.choice(vocab_size, size=num_samples, replace=False, p=probs_2d[i])
    
    result_shape = batch_shape + (num_samples,)
    result_data = indices.reshape(result_shape)
    
    return Tensor(result_data, requires_grad=False, dtype=np.int64)

