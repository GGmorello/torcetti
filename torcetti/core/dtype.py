import numpy as np

_DEFAULT_DTYPE = np.float32


def get_default_dtype():
    return _DEFAULT_DTYPE


def set_default_dtype(dtype):
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


