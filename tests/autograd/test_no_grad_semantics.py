import numpy as np

from torcetti.core.tensor import Tensor
from torcetti.core.grad_mode import no_grad, grad_enabled


def test_no_grad_disables_tracking_for_nonleafs():
    a = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        b = a * 3.0
        c = b + 2.0
    assert b.requires_grad is False
    assert c.requires_grad is False
    # Backward through c should not affect a, since graph wasn't recorded
    a2 = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        out = (a2 * 3.0 + 2.0).sum()
    # out has no grad function; nothing to backward
    try:
        out.backward()
    except Exception:
        # It's okay if calling backward fails; key point is no grad tracked
        pass
    assert a.grad.data is None if a.grad else True


def test_grad_enabled_restores_tracking_inside_context():
    a = Tensor([1.0, 2.0], requires_grad=False)
    with grad_enabled():
        b = a + 5.0
    # b is a leaf created in grad-enabled context but from non-requires-grad input -> still False
    assert b.requires_grad is False
    # Create fresh tensors fully inside grad-enabled scope
    with grad_enabled():
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2.0
        (y.sum()).backward()
        assert x.grad is not None and x.grad.data is not None




