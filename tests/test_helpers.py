"""
Test helpers for comparing torcetti against PyTorch and other utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torcetti.core.tensor import Tensor


def to_torch(torcetti_tensor, requires_grad=None):
    """Convert torcetti Tensor to PyTorch tensor."""
    if requires_grad is None:
        requires_grad = torcetti_tensor.requires_grad
    return torch.from_numpy(torcetti_tensor.data).requires_grad_(requires_grad)


def to_torcetti(torch_tensor, requires_grad=None):
    """Convert PyTorch tensor to torcetti Tensor."""
    if requires_grad is None:
        requires_grad = torch_tensor.requires_grad
    return Tensor(torch_tensor.detach().numpy(), requires_grad=requires_grad)


def assert_tensors_close(torcetti_tensor, torch_tensor, rtol=1e-5, atol=1e-8):
    """Assert that torcetti and PyTorch tensors are close."""
    if isinstance(torch_tensor, torch.Tensor):
        torch_data = torch_tensor.detach().numpy()
    else:
        torch_data = torch_tensor
    np.testing.assert_allclose(
        torcetti_tensor.data, torch_data, rtol=rtol, atol=atol
    )


def assert_gradients_close(torcetti_tensor, torch_tensor, rtol=1e-5, atol=1e-8):
    """Assert that gradients are close."""
    torcetti_grad = torcetti_tensor.grad.data if torcetti_tensor.grad else None
    torch_grad = torch_tensor.grad.detach().numpy() if torch_tensor.grad is not None else None
    if torcetti_grad is None or torch_grad is None:
        if torcetti_grad is not None or torch_grad is not None:
            raise AssertionError("One tensor has grad, the other doesn't")
        return
    np.testing.assert_allclose(torcetti_grad, torch_grad, rtol=rtol, atol=atol)


def compare_forward_backward(torcetti_fn, torch_fn, inputs, grad_outputs=None, requires_grad=None, atol=1e-8, rtol=1e-5):
    """
    Compare forward and backward passes between torcetti and PyTorch.
    
    Args:
        torcetti_fn: Function that takes torcetti tensors
        torch_fn: Function that takes PyTorch tensors  
        inputs: List of input data (numpy arrays)
        grad_outputs: Optional gradient outputs for backward pass
        requires_grad: List of booleans indicating which inputs need gradients (default: all True)
    
    Returns:
        dict with 'torcetti_result', 'torch_result', and comparison results
    """
    if requires_grad is None:
        requires_grad = [True] * len(inputs)
    torcetti_inputs = [Tensor(inp, requires_grad=req_grad) for inp, req_grad in zip(inputs, requires_grad)]
    torch_inputs = [torch.from_numpy(inp).requires_grad_(req_grad) for inp, req_grad in zip(inputs, requires_grad)]
    torcetti_result = torcetti_fn(*torcetti_inputs)
    torch_result = torch_fn(*torch_inputs)
    assert_tensors_close(torcetti_result, torch_result, atol=atol, rtol=rtol)
    if grad_outputs is None:
        grad_outputs = [np.ones_like(torcetti_result.data)]
    torcetti_result.backward(grad_outputs[0])
    torch_result.backward(torch.from_numpy(grad_outputs[0]))
    for torcetti_inp, torch_inp, req_grad in zip(torcetti_inputs, torch_inputs, requires_grad):
        if req_grad:
            assert_gradients_close(torcetti_inp, torch_inp, atol=atol, rtol=rtol)
    return {
        'torcetti_result': torcetti_result,
        'torch_result': torch_result,
        'torcetti_inputs': torcetti_inputs,
        'torch_inputs': torch_inputs
    }


def random_tensor(shape, requires_grad=False, dtype=np.float32):
    """Generate random tensor for testing."""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def random_torch_tensor(shape, requires_grad=False, dtype=torch.float32):
    """Generate random PyTorch tensor for testing."""
    return torch.randn(shape, requires_grad=requires_grad, dtype=dtype)


