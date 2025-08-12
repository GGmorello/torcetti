import numpy as np

def _unbroadcast(grad, target_shape):
    target_shape = tuple(target_shape)
    padded_shape = (1,) * (grad.ndim - len(target_shape)) + target_shape

    axes = tuple(i for i, (g, t) in enumerate(zip(grad.shape, padded_shape)) if t == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    return grad.reshape(target_shape)

