"""
Consolidated tensor tests using PyTorch for reference comparisons.
"""

import unittest
import numpy as np
import torch
import torcetti
from torcetti.core.tensor import Tensor
from tests.test_helpers import (
    assert_tensors_close, assert_gradients_close,
    compare_forward_backward, random_tensor, to_torch
)


class TestTensorCreation(unittest.TestCase):
    def test_from_list(self):
        data = [1, 2, 3, 4]
        torcetti_tensor = Tensor(data)
        torch_tensor = torch.from_numpy(np.array(data, dtype=np.float32))
        assert_tensors_close(torcetti_tensor, torch_tensor)
        self.assertEqual(torcetti_tensor.shape, torch_tensor.shape)
        self.assertEqual(torcetti_tensor.dtype, np.float32)
        self.assertFalse(torcetti_tensor.requires_grad)

    def test_from_numpy(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.int64)
        torcetti_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.from_numpy(data)
        assert_tensors_close(torcetti_tensor, torch_tensor)
        self.assertTrue(torcetti_tensor.requires_grad)
        self.assertEqual(torcetti_tensor.dtype, np.int64)

    def test_tensor_factory(self):
        data = [1, 2, 3]
        t1 = torcetti.tensor(data)
        torch_ref = torch.from_numpy(np.array(data, dtype=np.float32))
        assert_tensors_close(t1, torch_ref)
        t2 = torcetti.tensor(data, dtype=np.int32)
        torch_ref2 = torch.from_numpy(np.array(data, dtype=np.int32))
        assert_tensors_close(t2, torch_ref2)
        t3 = torcetti.tensor(data, requires_grad=True)
        torch_ref3 = torch.from_numpy(np.array(data, dtype=np.float32)).requires_grad_(True)
        assert_tensors_close(t3, torch_ref3)
        self.assertTrue(t3.requires_grad)

    def test_dtype_promotion(self):
        t_f32 = Tensor([1.0], dtype=np.float32)
        t_f64 = Tensor([2.0], dtype=np.float64)
        t_i32 = Tensor([3], dtype=np.int32)
        result = t_f32 + t_f64; self.assertEqual(result.dtype, np.float64)
        result = t_f32 + t_i32; self.assertEqual(result.dtype, np.float32)
        torch_f32 = torch.from_numpy(np.array([1.0], dtype=np.float32))
        torch_f64 = torch.from_numpy(np.array([2.0], dtype=np.float64))
        torch_i32 = torch.from_numpy(np.array([3], dtype=np.int32))
        torch_result_f32_f64 = torch_f32 + torch_f64
        self.assertEqual((t_f32 + t_f64).dtype, torch_result_f32_f64.detach().numpy().dtype)
        torch_result_f32_i32 = torch_f32 + torch_i32
        self.assertEqual((t_f32 + t_i32).dtype, torch_result_f32_i32.detach().numpy().dtype)


class TestTensorOperations(unittest.TestCase):
    def test_addition(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a, b: a + b, lambda a, b: a + b, [a_data, b_data])
        compare_forward_backward(lambda a: a + 5.0, lambda a: a + 5.0, [a_data])
        compare_forward_backward(lambda a: 5.0 + a, lambda a: 5.0 + a, [a_data])

    def test_subtraction(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a, b: a - b, lambda a, b: a - b, [a_data, b_data])

    def test_rsub(self):
        np.random.seed(42); a_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a: 5.0 - a, lambda a: 5.0 - a, [a_data])
        compare_forward_backward(lambda a: -2.5 - a, lambda a: -2.5 - a, [a_data])
        compare_forward_backward(lambda a: 0.0 - a, lambda a: 0.0 - a, [a_data])

    def test_multiplication(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a, b: a * b, lambda a, b: a * b, [a_data, b_data])
        compare_forward_backward(lambda a: a * 3.0, lambda a: a * 3.0, [a_data])

    def test_division(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3) + 1.0; b_data = np.random.randn(2, 3) + 1.0
        compare_forward_backward(lambda a, b: a / b, lambda a, b: a / b, [a_data, b_data])

    def test_power(self):
        np.random.seed(42)
        a_data = np.abs(np.random.randn(2, 3)) + 0.1
        compare_forward_backward(lambda a: a ** 2.0, lambda a: a ** 2.0, [a_data])

    def test_negation(self):
        np.random.seed(42); a_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a: -a, lambda a: -a, [a_data])

    def test_dot_product(self):
        np.random.seed(42); a_data = np.random.randn(5); b_data = np.random.randn(5)
        compare_forward_backward(lambda a, b: a.dot(b), lambda a, b: torch.dot(a, b), [a_data, b_data])

    def test_matrix_multiplication(self):
        np.random.seed(42); a_data = np.random.randn(2, 3); b_data = np.random.randn(3, 4)
        compare_forward_backward(lambda a, b: a @ b, lambda a, b: a @ b, [a_data, b_data])

    def test_transpose(self):
        np.random.seed(42); a_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a: a.T, lambda a: a.T, [a_data])

    def test_sum(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.sum(), lambda a: a.sum(), [a_data])
        compare_forward_backward(lambda a: a.sum(dim=1), lambda a: a.sum(dim=1), [a_data])

    def test_mean(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.mean(), lambda a: a.mean(), [a_data])

    def test_mean_axis_backward_bug(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.mean(dim=1), lambda a: a.mean(dim=1), [a_data])
        compare_forward_backward(lambda a: a.mean(dim=0), lambda a: a.mean(dim=0), [a_data])
        compare_forward_backward(lambda a: a.mean(dim=2), lambda a: a.mean(dim=2), [a_data])

    def test_mean_axis_keepdims(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.mean(dim=1, keepdim=True), lambda a: a.mean(dim=1, keepdim=True), [a_data])
        compare_forward_backward(lambda a: a.mean(dim=0, keepdim=True), lambda a: a.mean(dim=0, keepdim=True), [a_data])
        compare_forward_backward(lambda a: a.mean(dim=-1, keepdim=False), lambda a: a.mean(dim=-1, keepdim=False), [a_data])

    def test_max(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.max(dim=1), lambda a: a.max(dim=1)[0], [a_data])

    def test_exp(self):
        np.random.seed(42); a_data = np.random.randn(2, 3) * 0.1
        compare_forward_backward(lambda a: a.exp(), lambda a: a.exp(), [a_data])

    def test_log(self):
        np.random.seed(42); a_data = np.abs(np.random.randn(2, 3)) + 0.1
        compare_forward_backward(lambda a: a.log(), lambda a: a.log(), [a_data])


class TestTensorBroadcasting(unittest.TestCase):
    def test_broadcasting_add(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(3)
        compare_forward_backward(lambda a, b: a + b, lambda a, b: a + b, [a_data, b_data])

    def test_broadcasting_mul(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 1)
        compare_forward_backward(lambda a, b: a * b, lambda a, b: a * b, [a_data, b_data])


class TestTensorIndexing(unittest.TestCase):
    def test_basic_indexing(self):
        np.random.seed(42); data = np.random.randn(3, 4, 5)
        torcetti_tensor = Tensor(data); torch_tensor = torch.from_numpy(data)
        test_cases = [(0, 0), (slice(None), slice(1, 3)), (1, slice(None)), (slice(0, 2), slice(1, 4))]
        for idx in test_cases:
            with self.subTest(idx=idx):
                torcetti_result = torcetti_tensor[idx]; torch_result = torch_tensor[idx]
                assert_tensors_close(torcetti_result, torch_result)


class TestComplexOperations(unittest.TestCase):
    def test_complex_computation_graph(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 3); c_data = np.random.randn(2, 3)
        def complex_fn(a, b, c):
            return (a + b) * c - a**2 + b.sum()
        compare_forward_backward(complex_fn, complex_fn, [a_data, b_data, c_data])

    def test_nested_operations(self):
        np.random.seed(42); x_data = np.random.randn(2, 3) * 0.1
        def nested_fn(x): return (x.exp() + 1).log() - x
        compare_forward_backward(nested_fn, nested_fn, [x_data])


class TestTensorReshapeFlatten(unittest.TestCase):
    def test_reshape(self):
        np.random.seed(42); a_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a: a.reshape(6), lambda a: a.reshape(6), [a_data])

    def test_view(self):
        np.random.seed(42); a_data = np.random.randn(2, 3)
        compare_forward_backward(lambda a: a.view(6), lambda a: a.view(6), [a_data])

    def test_flatten(self):
        np.random.seed(42); a_data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.flatten(), lambda a: a.flatten(), [a_data])


class TestTensorConcat(unittest.TestCase):
    def test_concat_dim0(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(4, 3)
        compare_forward_backward(lambda a, b: torcetti.cat([a, b], dim=0), lambda a, b: torch.cat([a, b], dim=0), [a_data, b_data])

    def test_concat_dim1(self):
        np.random.seed(42)
        a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 4)
        compare_forward_backward(lambda a, b: torcetti.cat([a, b], dim=1), lambda a, b: torch.cat([a, b], dim=1), [a_data, b_data])


class TestTensorSliceBackward(unittest.TestCase):
    def test_slice_backward(self):
        np.random.seed(42); a_data = np.random.randn(3, 4)
        compare_forward_backward(lambda a: a[1:3].sum(), lambda a: a[1:3].sum(), [a_data])


class TestTensorReshapeFlattenEdgeCases(unittest.TestCase):
    def test_reshape_infer_dim(self):
        np.random.seed(0); data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.reshape(6, -1), lambda a: a.reshape(6, -1), [data])

    def test_reshape_noop(self):
        np.random.seed(1); data = np.random.randn(5, 5)
        compare_forward_backward(lambda a: a.reshape(5, 5), lambda a: a.reshape(5, 5), [data])

    def test_flatten_partial_dims(self):
        np.random.seed(2); data = np.random.randn(2, 3, 4)
        compare_forward_backward(lambda a: a.flatten(start_dim=1), lambda a: a.flatten(start_dim=1), [data])


class TestTensorConcatEdgeCases(unittest.TestCase):
    def test_concat_multiple_tensors(self):
        np.random.seed(3)
        a, b, c = (np.random.randn(1, 4), np.random.randn(2, 4), np.random.randn(3, 4))
        compare_forward_backward(lambda x, y, z: torcetti.cat([x, y, z], dim=0), lambda x, y, z: torch.cat([x, y, z], dim=0), [a, b, c])

    def test_concat_negative_axis(self):
        np.random.seed(4); a_data = np.random.randn(2, 3); b_data = np.random.randn(2, 2)
        compare_forward_backward(lambda a, b: torcetti.cat([a, b], dim=-1), lambda a, b: torch.cat([a, b], dim=-1), [a_data, b_data])

    def test_concat_shape_mismatch_error(self):
        a_data = np.random.randn(2, 3); b_data = np.random.randn(3, 3)
        a = Tensor(a_data); b = Tensor(b_data)
        with self.assertRaises(Exception):
            torcetti.cat([a, b], dim=1)


class TestTensorSliceEdgeCases(unittest.TestCase):
    def test_slice_with_step(self):
        np.random.seed(5); data = np.random.randn(6, 4)
        compare_forward_backward(lambda a: a[::2].sum(), lambda a: a[::2].sum(), [data])

    def test_slice_negative_indices(self):
        np.random.seed(6); data = np.random.randn(4, 5)
        compare_forward_backward(lambda a: a[-2:].sum(), lambda a: a[-2:].sum(), [data])


if __name__ == '__main__':
    unittest.main()


