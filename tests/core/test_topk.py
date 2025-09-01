import numpy as np
import unittest

import torcetti
from torcetti.core.tensor import Tensor


class TestTopK(unittest.TestCase):
    def test_topk_basic_largest_sorted(self):
        x = Tensor(np.array([[1.0, 3.0, 2.0], [4.0, -1.0, 0.5]], dtype=np.float32), requires_grad=True)
        vals, idx = torcetti.topk(x, k=2, dim=1, largest=True, sorted=True)
        np.testing.assert_array_equal(idx.data, np.array([[1, 2], [0, 2]]))
        np.testing.assert_allclose(vals.data, np.array([[3.0, 2.0], [4.0, 0.5]]))

    def test_topk_smallest_unsorted(self):
        rng = np.random.default_rng(0)
        x = Tensor(rng.standard_normal((4, 5)).astype(np.float32), requires_grad=True)
        vals, idx = torcetti.topk(x, k=3, dim=-1, largest=False, sorted=False)
        self.assertEqual(vals.shape, (4, 3))
        self.assertEqual(idx.shape, (4, 3))
        # Values correspond to the indices along the axis
        gathered = np.take_along_axis(x.data, idx.data, axis=-1)
        np.testing.assert_allclose(vals.data, gathered)

    def test_topk_gradient_routed_to_selected(self):
        x = Tensor(np.array([1.0, 5.0, 2.0, 4.0], dtype=np.float32), requires_grad=True)
        vals, idx = torcetti.topk(x, k=2, dim=0, largest=True, sorted=True)
        (vals.sum()).backward()
        # The top-2 are 5.0 (idx 1) and 4.0 (idx 3)
        expected = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(x.grad.data, expected)

    def test_topk_k_equals_dimension(self):
        x = Tensor(np.array([2.0, -1.0, 3.0], dtype=np.float32), requires_grad=True)
        vals, idx = torcetti.topk(x, k=3, dim=0, largest=True, sorted=True)
        # Should return a permutation of indices and all values sorted desc
        np.testing.assert_allclose(vals.data, np.array([3.0, 2.0, -1.0], dtype=np.float32))
        np.testing.assert_array_equal(np.sort(idx.data), np.array([0, 1, 2]))

    def test_topk_invalid_k_raises(self):
        x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            torcetti.topk(x, k=0)
        with self.assertRaises(ValueError):
            torcetti.topk(x, k=4)


if __name__ == '__main__':
    unittest.main()




