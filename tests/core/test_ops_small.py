import unittest
import numpy as np
from torcetti.core.tensor import Tensor
import pytest
import torcetti


class TestBroadcastOps(unittest.TestCase):
    def test_expand_backward(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x.unsqueeze(0)
        z = y.expand(3, -1)
        self.assertEqual(z.shape, (3, 2))
        loss = z.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.data, np.array([3.0, 3.0]))

    def test_squeeze_unsqueeze(self):
        x = Tensor(np.ones((1, 2, 1, 3)), requires_grad=True)
        y = x.squeeze()
        self.assertEqual(y.shape, (2, 3))
        z = y.unsqueeze(0)
        self.assertEqual(z.shape, (1, 2, 3))
        loss = z.sum()
        loss.backward()
        self.assertTrue(np.all(x.grad.data == 1))


class TestElemwise(unittest.TestCase):
    def test_abs(self):
        x = Tensor([-1.0, 2.0], requires_grad=True)
        y = x.abs()
        np.testing.assert_allclose(y.data, np.array([1.0, 2.0]))
        y.sum().backward()
        np.testing.assert_allclose(x.grad.data, np.array([-1.0, 1.0]))

    def test_clamp(self):
        x = Tensor([-1.0, 0.5, 3.0], requires_grad=True)
        y = x.clamp(min=0.0, max=2.0)
        np.testing.assert_allclose(y.data, np.array([0.0, 0.5, 2.0]))
        y.sum().backward()
        np.testing.assert_allclose(x.grad.data, np.array([0.0, 1.0, 0.0]))

    def test_where(self):
        cond = np.array([True, False, True])
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        y = Tensor.where(cond, a, b)
        np.testing.assert_allclose(y.data, np.array([1.0, 5.0, 3.0]))
        y.sum().backward()
        np.testing.assert_allclose(a.grad.data, np.array([1.0, 0.0, 1.0]))
        np.testing.assert_allclose(b.grad.data, np.array([0.0, 1.0, 0.0]))

    def test_floor_ceil_round(self):
        x = Tensor([1.2, -1.7, 2.5], requires_grad=True)
        self.assertTrue(np.array_equal(x.floor().data, np.floor(x.data)))
        self.assertTrue(np.array_equal(x.ceil().data, np.ceil(x.data)))
        self.assertTrue(np.array_equal(x.round().data, np.round(x.data)))


class TestReductions(unittest.TestCase):
    def test_min_backward(self):
        x = Tensor([[1.0, 2.0], [3.0, 0.0]], requires_grad=True)
        y = x.min()
        self.assertEqual(y.data, 0.0)
        y.backward()
        grad_expected = np.zeros_like(x.data)
        grad_expected[1, 1] = 1
        np.testing.assert_allclose(x.grad.data, grad_expected)

    def test_prod_backward(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.prod()
        self.assertEqual(y.data, 6.0)
        y.backward()
        np.testing.assert_allclose(x.grad.data, np.array([6.0, 3.0, 2.0]))

    def test_argmax_min(self):
        x = Tensor([[1.0, 5.0], [7.0, 4.0]])
        idx_max = x.argmax()
        idx_min = x.argmin()
        self.assertFalse(idx_max.requires_grad)
        self.assertEqual(idx_max.dtype, np.int64)
        self.assertEqual(idx_max.data, 2)
        self.assertEqual(idx_min.data, 0)


def test_multinomial():
    """Test multinomial sampling from probability distributions."""
    # Test basic multinomial sampling
    probs = torcetti.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    samples = torcetti.multinomial(probs, num_samples=1)
    
    # Check shape
    assert samples.shape == (2, 1)
    assert samples.dtype == np.int64
    
    # Check that samples are valid indices
    assert all(0 <= sample < 3 for sample in samples.data.flatten())
    
    # Test multiple samples with replacement
    samples_multi = torcetti.multinomial(probs, num_samples=5, replacement=True)
    assert samples_multi.shape == (2, 5)
    assert all(0 <= sample < 3 for sample in samples_multi.data.flatten())
    
    # Test with replacement=False (num_samples must be <= vocab_size)
    samples_no_replace = torcetti.multinomial(probs, num_samples=2, replacement=False)
    assert samples_no_replace.shape == (2, 2)
    assert all(0 <= sample < 3 for sample in samples_no_replace.data.flatten())
    
    # Test that sampling is not differentiable
    assert not samples.requires_grad


def test_expand_with_negative_one():
    """Test expand function with -1 dimension like PyTorch."""
    x = torcetti.tensor([[1, 2, 3]])
    
    # Test expand with -1 to keep original dimension
    expanded = x.expand(3, -1)
    assert expanded.shape == (3, 3)
    assert np.array_equal(expanded.data, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    
    # Test expand with explicit dimensions
    expanded2 = x.expand(3, 3)
    assert expanded2.shape == (3, 3)
    
    # Test that -1 preserves original size
    y = torcetti.tensor([[1], [2], [3]])
    expanded_y = y.expand(-1, 4)
    assert expanded_y.shape == (3, 4)


if __name__ == '__main__':
    unittest.main()


