import numpy as np
import unittest
from torcetti.core.tensor import Tensor
import torcetti


class TestVarDdof(unittest.TestCase):
    def test_var_ddof_backward_axis_none(self):
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        for ddof in [0, 1, 2]:
            with self.subTest(ddof=ddof):
                x = Tensor(data.copy(), requires_grad=True)
                var_result = x.var(ddof=ddof)
                var_result.backward()
                mean_val = np.mean(data)
                N = data.size
                expected_grad = (2.0 / (N - ddof)) * (data - mean_val)
                np.testing.assert_array_almost_equal(x.grad.data, expected_grad, decimal=6)

    def test_var_ddof_backward_with_axis(self):
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        for ddof in [0, 1]:
            for axis in [0, 1]:
                with self.subTest(ddof=ddof, axis=axis):
                    x = Tensor(data.copy(), requires_grad=True)
                    var_result = x.var(axis=axis, ddof=ddof)
                    upstream_grad = np.ones_like(var_result.data)
                    var_result.grad += upstream_grad
                    var_result.backward()
                    mean_val = np.mean(data, axis=axis, keepdims=True)
                    N = data.shape[axis]
                    expected_grad = (2.0 / (N - ddof)) * (data - mean_val)
                    np.testing.assert_array_almost_equal(x.grad.data, expected_grad, decimal=6)

    def test_var_ddof_backward_keepdims(self):
        np.random.seed(42)
        data = np.random.randn(2, 3, 4).astype(np.float32)
        for ddof in [0, 1]:
            for keepdims in [True, False]:
                with self.subTest(ddof=ddof, keepdims=keepdims):
                    x = Tensor(data.copy(), requires_grad=True)
                    var_result = x.var(axis=1, ddof=ddof, keepdims=keepdims)
                    var_result.backward()
                    mean_val = np.mean(data, axis=1, keepdims=True)
                    N = data.shape[1]
                    expected_grad = (2.0 / (N - ddof)) * (data - mean_val)
                    np.testing.assert_array_almost_equal(x.grad.data, expected_grad, decimal=6)

    def test_var_ddof_consistency_with_numpy(self):
        np.random.seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        for ddof in [0, 1, 2]:
            with self.subTest(ddof=ddof):
                x = Tensor(data.copy())
                torcetti_var = x.var(ddof=ddof)
                numpy_var = np.var(data, ddof=ddof)
                np.testing.assert_array_almost_equal(torcetti_var.data, numpy_var, decimal=6)

    def test_var_ddof_axis_consistency(self):
        np.random.seed(42)
        data = np.random.randn(3, 4, 5).astype(np.float32)
        for ddof in [0, 1]:
            for axis in [0, 1, 2]:
                with self.subTest(ddof=ddof, axis=axis):
                    x = Tensor(data.copy())
                    torcetti_var = x.var(axis=axis, ddof=ddof)
                    numpy_var = np.var(data, axis=axis, ddof=ddof)
                    np.testing.assert_array_almost_equal(torcetti_var.data, numpy_var, decimal=6)

    def test_var_ddof_gradient_scaling_difference(self):
        np.random.seed(42)
        data = np.random.randn(4, 4).astype(np.float32)
        gradients = {}
        for ddof in [0, 1, 2]:
            x = Tensor(data.copy(), requires_grad=True)
            var_result = x.var(ddof=ddof)
            var_result.backward()
            gradients[ddof] = x.grad.data.copy()
        self.assertFalse(np.allclose(gradients[0], gradients[1]))
        self.assertFalse(np.allclose(gradients[1], gradients[2]))
        grad_magnitude_ddof0 = np.mean(np.abs(gradients[0]))
        grad_magnitude_ddof1 = np.mean(np.abs(gradients[1]))
        grad_magnitude_ddof2 = np.mean(np.abs(gradients[2]))
        self.assertGreater(grad_magnitude_ddof1, grad_magnitude_ddof0)
        self.assertGreater(grad_magnitude_ddof2, grad_magnitude_ddof1)

    def test_var_ddof_edge_case_single_element(self):
        data = np.array([1.0, 2.0], dtype=np.float32)
        x = Tensor(data.copy(), requires_grad=True)
        var_result = x.var(ddof=1)
        var_result.backward()
        mean_val = np.mean(data)
        expected_grad = (2.0 / 1) * (data - mean_val)
        np.testing.assert_array_almost_equal(x.grad.data, expected_grad, decimal=6)

    def test_var_ddof_multiple_axes(self):
        np.random.seed(42)
        data = np.random.randn(2, 3, 4).astype(np.float32)
        for ddof in [0, 1]:
            with self.subTest(ddof=ddof):
                x = Tensor(data.copy(), requires_grad=True)
                var_result = x.var(axis=(1, 2), ddof=ddof)
                var_result.backward()
                mean_val = np.mean(data, axis=(1, 2), keepdims=True)
                N = data.shape[1] * data.shape[2]
                expected_grad = (2.0 / (N - ddof)) * (data - mean_val)
                np.testing.assert_array_almost_equal(x.grad.data, expected_grad, decimal=6)


class TestTake(unittest.TestCase):
    def test_take_basic_flatten(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
        indices = [0, 2, 4]
        result = x.take(indices)
        expected = np.array([1, 3, 5], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_take_with_tensor_indices(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), requires_grad=True)
        indices = Tensor([1, 3, 5])
        result = x.take(indices)
        expected = np.array([2, 4, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_take_axis_specific(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), requires_grad=True)
        result = x.take([0, 2], axis=0)
        expected = np.array([[1, 2, 3], [7, 8, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_take_backward_flatten(self):
        x = Tensor(np.array([1, 2, 3, 4, 5], dtype=np.float32), requires_grad=True)
        result = x.take([1, 3, 4])
        loss = result.sum(); loss.backward()
        expected_grad = np.array([0, 1, 0, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_take_backward_axis(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), requires_grad=True)
        result = x.take([0, 2], axis=0)
        loss = result.sum(); loss.backward()
        expected_grad = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_take_factory_function(self):
        x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        result = torcetti.take(x, [0, 2, 4])
        expected = np.array([1, 3, 5], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_take_factories_with_array(self):
        x_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = torcetti.take(x_array, [1, 3, 5])
        expected = np.array([2, 4, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_take_duplicate_indices(self):
        x = Tensor(np.array([1, 2, 3, 4], dtype=np.float32), requires_grad=True)
        result = x.take([1, 1, 3, 1])
        expected = np.array([2, 2, 4, 2], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
        loss = result.sum(); loss.backward()
        expected_grad = np.array([0, 3, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.data, expected_grad)

    def test_take_negative_indices(self):
        x = Tensor(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        result = x.take([-1, -2])
        expected = np.array([5, 4], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)


if __name__ == '__main__':
    unittest.main()


