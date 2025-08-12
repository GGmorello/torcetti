import numpy as np
import pytest

from torcetti.core.tensor import Tensor


def _build_simple_graph():
    a = Tensor([2.0], requires_grad=True)
    b = a * a
    c = b * 3
    return a, b, c


def test_backward_frees_intermediate_grads():
    a, b, c = _build_simple_graph()
    c.backward()
    assert a.grad.data is not None and np.allclose(a.grad.data, np.array([12.0]))
    assert b.grad.data is None


def test_retain_graph_keeps_intermediate_grads():
    a, b, c = _build_simple_graph()
    c.backward(retain_graph=True)
    assert b.grad.data is not None
    prev_grad_a = a.grad.data.copy()
    c.backward()
    assert np.allclose(a.grad.data, prev_grad_a * 2)


def test_second_backward_without_retain_graph_raises():
    a, _b, c = _build_simple_graph()
    c.backward()
    with pytest.raises(Exception):
        c.backward()


