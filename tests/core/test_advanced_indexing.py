import numpy as np
import unittest

from torcetti.core.tensor import Tensor


class TestAdvancedIndexing(unittest.TestCase):
    def test_boolean_mask_forward(self):
        x = Tensor(np.array([1.0, -2.0, 3.0, 0.0], dtype=np.float32), requires_grad=False)
        mask = np.array([True, False, True, False])
        y = x[mask]
        np.testing.assert_array_equal(y.data, np.array([1.0, 3.0], dtype=np.float32))

    def test_boolean_mask_backward_scatter(self):
        x = Tensor(np.arange(6, dtype=np.float32), requires_grad=True)
        mask = np.array([True, False, True, False, True, False])
        y = x[mask]
        self.assertEqual(y.shape, (3,))
        (y.sum()).backward()
        expected = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(x.grad.data, expected)

    def test_mixed_indexing_and_ellipsis(self):
        x = Tensor(np.arange(2*3*4).reshape(2,3,4).astype(np.float32), requires_grad=True)
        y = x[1, ..., 2]
        self.assertEqual(y.shape, (3,))
        (y.sum()).backward()
        # Gradient accumulates only on the selected slice (index 2 on last dim of batch 1)
        grad = np.zeros_like(x.data)
        grad[1, :, 2] = 1.0
        np.testing.assert_allclose(x.grad.data, grad)


if __name__ == '__main__':
    unittest.main()




