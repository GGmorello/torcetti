import unittest
import numpy as np
import torch
import torcetti
from torcetti.core.tensor import Tensor
from tests.test_helpers import assert_tensors_close


class TestBasicFactoryHelpers(unittest.TestCase):
    def test_zeros(self):
        t = torcetti.zeros((2, 3))
        reference = torch.zeros((2, 3), dtype=torch.float32)
        self.assertFalse(t.requires_grad)
        assert_tensors_close(t, reference)

    def test_ones(self):
        t = torcetti.ones((4,), dtype=np.float64, requires_grad=True)
        reference = torch.ones((4,), dtype=torch.float64, requires_grad=True)
        self.assertTrue(t.requires_grad)
        assert_tensors_close(t, reference)
        self.assertEqual(t.dtype, np.float64)

    def test_zeros_like(self):
        base = Tensor(np.random.randn(3, 2).astype(np.float32), requires_grad=True)
        z = torcetti.zeros_like(base)
        reference = torch.zeros_like(torch.from_numpy(base.data))
        self.assertFalse(z.requires_grad)
        assert_tensors_close(z, reference)
        self.assertEqual(z.shape, base.shape)

    def test_ones_like(self):
        base = Tensor(np.random.randint(0, 5, size=(2, 2)).astype(np.int32))
        o = torcetti.ones_like(base, requires_grad=True)
        reference = torch.ones_like(torch.from_numpy(base.data), dtype=torch.int32)
        self.assertTrue(o.requires_grad)
        assert_tensors_close(o, reference)
        self.assertEqual(o.shape, base.shape)
        self.assertEqual(o.dtype, base.dtype)

    def test_randn(self):
        np.random.seed(0)
        torch.manual_seed(0)
        r = torcetti.randn((5, 1), dtype=np.float32)
        self.assertEqual(r.shape, (5, 1))
        self.assertEqual(r.dtype, np.float32)
        self.assertFalse(r.requires_grad)
        self.assertTrue(abs(r.data.mean()) < 0.5)
        self.assertTrue(0.5 < r.data.std() < 2.0)


class TestConstantFactories(unittest.TestCase):
    def test_full(self):
        t = torcetti.full((2, 3), 3.14, dtype=np.float32)
        ref = torch.full((2, 3), 3.14, dtype=torch.float32)
        assert_tensors_close(t, ref)
        self.assertEqual(t.dtype, np.float32)

    def test_full_like(self):
        base = Tensor(np.zeros((4, 1), dtype=np.int64))
        t = torcetti.full_like(base, 7)
        ref = torch.full_like(torch.from_numpy(base.data), 7)
        assert_tensors_close(t, ref)
        self.assertEqual(t.dtype, base.dtype)


class TestRangeFactories(unittest.TestCase):
    def test_arange_basic(self):
        t = torcetti.arange(5)
        ref = torch.arange(5, dtype=torch.float32)
        assert_tensors_close(t, ref)

    def test_arange_step(self):
        t = torcetti.arange(2, 10, 2, dtype=np.int32)
        ref = torch.arange(2, 10, 2, dtype=torch.int32)
        assert_tensors_close(t, ref)
        self.assertEqual(t.dtype, np.int32)

    def test_linspace(self):
        t = torcetti.linspace(0.0, 1.0, 5)
        ref = torch.linspace(0.0, 1.0, 5)
        assert_tensors_close(t, ref)


class TestMatrixFactories(unittest.TestCase):
    def test_eye(self):
        t = torcetti.eye(4)
        ref = torch.eye(4)
        assert_tensors_close(t, ref)

    def test_diag(self):
        v = torcetti.arange(1, 4)
        t = torcetti.diag(v)
        ref = torch.diag(torch.arange(1, 4))
        assert_tensors_close(t, ref)


class TestRandomFactories(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

    def test_rand(self):
        t = torcetti.rand((3, 2))
        self.assertEqual(t.shape, (3, 2))
        self.assertTrue((t.data >= 0).all() and (t.data < 1).all())

    def test_rand_like(self):
        base = Tensor(np.empty((2, 2)))
        t = torcetti.rand_like(base)
        self.assertEqual(t.shape, base.shape)

    def test_randn_like(self):
        base = Tensor(np.empty((2, 3)))
        t = torcetti.randn_like(base)
        self.assertEqual(t.shape, base.shape)

    def test_randint(self):
        t = torcetti.randint(0, 5, (4,))
        self.assertTrue(((t.data >= 0) & (t.data < 5)).all())

    def test_normal(self):
        t = torcetti.normal((1000,), mean=2.0, std=0.5)
        self.assertAlmostEqual(t.data.mean(), 2.0, delta=0.2)
        self.assertAlmostEqual(t.data.std(), 0.5, delta=0.2)


class TestEmptyFactories(unittest.TestCase):
    def test_empty(self):
        t = torcetti.empty((2, 2))
        self.assertEqual(t.shape, (2, 2))

    def test_empty_like(self):
        base = Tensor(np.zeros((3, 1)))
        t = torcetti.empty_like(base)
        self.assertEqual(t.shape, base.shape)


class TestCompositionHelpers(unittest.TestCase):
    def test_stack(self):
        a = torcetti.ones((2,))
        b = torcetti.zeros((2,))
        s = torcetti.stack([a, b], axis=0)
        ref = torch.stack([torch.ones(2), torch.zeros(2)], dim=0)
        assert_tensors_close(s, ref)
        self.assertEqual(s.shape, (2, 2))

    def test_meshgrid(self):
        x = torcetti.arange(3)
        y = torcetti.arange(4)
        gx, gy = torcetti.meshgrid(x, y)
        rx, ry = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
        assert_tensors_close(gx, rx)
        assert_tensors_close(gy, ry)

    def test_repeat_tile(self):
        a = torcetti.arange(3)
        r = a.repeat(2)
        ref = torch.arange(3).repeat(2)
        assert_tensors_close(r, ref)


class TestCopyHelpers(unittest.TestCase):
    def test_as_tensor(self):
        arr = [1, 2, 3]
        t = torcetti.as_tensor(arr, dtype=np.int32)
        ref = torch.as_tensor(arr, dtype=torch.int32)
        assert_tensors_close(t, ref)

    def test_clone(self):
        original = torcetti.ones((2, 2), requires_grad=True)
        c = original.clone()
        assert_tensors_close(original, c)
        self.assertIsNot(original.data, c.data)
        c.backward(np.ones_like(c.data))
        self.assertIsNone(original.grad.data)

    def test_detach(self):
        x = torcetti.ones((1,), requires_grad=True)
        y = x.detach()
        self.assertFalse(y.requires_grad)
        assert_tensors_close(x, y)


class TestShapeParameterFlexibility(unittest.TestCase):
    def test_zeros_with_int_shape(self):
        t = torcetti.zeros(5)
        reference = torch.zeros(5)
        assert_tensors_close(t, reference)
        self.assertEqual(t.shape, (5,))

    def test_ones_with_int_shape(self):
        t = torcetti.ones(3)
        reference = torch.ones(3)
        assert_tensors_close(t, reference)
        self.assertEqual(t.shape, (3,))

    def test_randn_with_int_shape(self):
        np.random.seed(0)
        torch.manual_seed(0)
        t = torcetti.randn(4)
        reference = torch.randn(4)
        self.assertEqual(t.shape, (4,))
        self.assertTrue(abs(t.data.mean()) < 0.5)
        self.assertTrue(0.5 < t.data.std() < 2.0)

    def test_rand_with_int_shape(self):
        np.random.seed(0)
        torch.manual_seed(0)
        t = torcetti.rand(6)
        reference = torch.rand(6)
        self.assertEqual(t.shape, (6,))
        self.assertTrue((t.data >= 0).all() and (t.data < 1).all())

    def test_empty_with_int_shape(self):
        t = torcetti.empty(7)
        reference = torch.empty(7)
        self.assertEqual(t.shape, (7,))

    def test_full_with_int_shape(self):
        t = torcetti.full(4, 2.5)
        reference = torch.full((4,), 2.5)
        assert_tensors_close(t, reference)
        self.assertEqual(t.shape, (4,))

    def test_normal_with_int_shape(self):
        np.random.seed(0)
        torch.manual_seed(0)
        t = torcetti.normal(100, mean=1.0, std=0.5)
        reference = torch.normal(mean=1.0, std=0.5, size=(100,))
        self.assertEqual(t.shape, (100,))
        self.assertAlmostEqual(t.data.mean(), 1.0, delta=0.2)

    def test_randint_with_int_shape(self):
        np.random.seed(0)
        torch.manual_seed(0)
        t = torcetti.randint(0, 10, 8)
        reference = torch.randint(0, 10, (8,))
        self.assertEqual(t.shape, (8,))
        self.assertTrue(((t.data >= 0) & (t.data < 10)).all())


class TestDtypePolicy(unittest.TestCase):
    def test_tensor_preserve_numpy_dtypes(self):
        test_cases = [
            (np.int32, [1, 2, 3]),
            (np.int64, [1, 2, 3]),
            (np.float32, [1.0, 2.0, 3.0]),
            (np.float64, [1.0, 2.0, 3.0]),
            (np.bool_, [True, False, True]),
        ]
        for numpy_dtype, data in test_cases:
            with self.subTest(dtype=numpy_dtype):
                np_array = np.array(data, dtype=numpy_dtype)
                tensor = torcetti.tensor(np_array)
                self.assertEqual(tensor.dtype, numpy_dtype)

    def test_tensor_python_lists_default_float32(self):
        t1 = torcetti.tensor([1, 2, 3])
        self.assertEqual(t1.dtype, np.float32)
        t2 = torcetti.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t2.dtype, np.float32)
        t3 = torcetti.tensor([1, 2.0, 3])
        self.assertEqual(t3.dtype, np.float32)

    def test_tensor_explicit_dtype_override(self):
        np_int = np.array([1, 2, 3], dtype=np.int64)
        tensor = torcetti.tensor(np_int, dtype=np.float32)
        self.assertEqual(tensor.dtype, np.float32)
        tensor2 = torcetti.tensor([1, 2, 3], dtype=np.int32)
        self.assertEqual(tensor2.dtype, np.int32)

    def test_as_tensor_same_behavior_as_tensor(self):
        np_array = np.array([1, 2, 3], dtype=np.int64)
        t1 = torcetti.tensor(np_array)
        t2 = torcetti.as_tensor(np_array)
        self.assertEqual(t1.dtype, t2.dtype)
        py_list = [1.0, 2.0, 3.0]
        t3 = torcetti.tensor(py_list)
        t4 = torcetti.as_tensor(py_list)
        self.assertEqual(t3.dtype, t4.dtype)
        t5 = torcetti.tensor(np_array, dtype=np.float32)
        t6 = torcetti.as_tensor(np_array, dtype=np.float32)
        self.assertEqual(t5.dtype, t6.dtype)

    def test_tensor_constructor_preserves_dtype(self):
        test_arrays = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([True, False], dtype=np.bool_),
        ]
        for np_array in test_arrays:
            with self.subTest(dtype=np_array.dtype):
                tensor = Tensor(np_array)
                self.assertEqual(tensor.dtype, np_array.dtype)

    def test_like_functions_preserve_reference_dtype(self):
        ref_int = torcetti.tensor(np.array([1, 2], dtype=np.int64))
        ref_float = torcetti.tensor(np.array([1.0, 2.0], dtype=np.float64))
        z1 = torcetti.zeros_like(ref_int)
        self.assertEqual(z1.dtype, np.int64)
        z2 = torcetti.zeros_like(ref_float)
        self.assertEqual(z2.dtype, np.float64)
        o1 = torcetti.ones_like(ref_int)
        self.assertEqual(o1.dtype, np.int64)
        o2 = torcetti.ones_like(ref_float)
        self.assertEqual(o2.dtype, np.float64)
        e1 = torcetti.empty_like(ref_int)
        self.assertEqual(e1.dtype, np.int64)
        e2 = torcetti.empty_like(ref_float)
        self.assertEqual(e2.dtype, np.float64)

    def test_like_functions_explicit_dtype_override(self):
        ref_tensor = torcetti.tensor(np.array([1, 2], dtype=np.int64))
        z = torcetti.zeros_like(ref_tensor, dtype=np.float32)
        self.assertEqual(z.dtype, np.float32)
        o = torcetti.ones_like(ref_tensor, dtype=np.float32)
        self.assertEqual(o.dtype, np.float32)

    def test_creation_functions_default_float32(self):
        functions_to_test = [
            (torcetti.zeros, (3,)),
            (torcetti.ones, (3,)),
            (torcetti.randn, (3,)),
            (torcetti.rand, (3,)),
            (torcetti.empty, (3,)),
            (torcetti.arange, (0, 5)),
            (torcetti.linspace, (0, 1, 5)),
            (torcetti.normal, (3,)),
        ]
        for func, args in functions_to_test:
            with self.subTest(function=func.__name__):
                result = func(*args)
                self.assertEqual(result.dtype, np.float32)

    def test_integer_creation_functions(self):
        r = torcetti.randint(0, 10, (3,))
        self.assertEqual(r.dtype, np.int64)
        e = torcetti.eye(3)
        self.assertEqual(e.dtype, np.float32)

    def test_full_functions_explicit_dtype(self):
        f1 = torcetti.full((3,), 5.0, dtype=np.int32)
        self.assertEqual(f1.dtype, np.int32)
        ref_tensor = torcetti.zeros(3)
        f2 = torcetti.full_like(ref_tensor, 5.0, dtype=np.int32)
        self.assertEqual(f2.dtype, np.int32)

    def test_tensor_from_tensor_preservation(self):
        original = Tensor(np.array([1, 2, 3], dtype=np.int64))
        t1 = torcetti.tensor(original)
        self.assertEqual(t1.dtype, np.int64)
        t2 = torcetti.tensor(original, dtype=np.float32)
        self.assertEqual(t2.dtype, np.float32)

    def test_complex_dtype_scenarios(self):
        empty_arr = np.array([], dtype=np.float64)
        t1 = torcetti.tensor(empty_arr)
        self.assertEqual(t1.dtype, np.float64)
        single_arr = np.array([42], dtype=np.int32)
        t2 = torcetti.tensor(single_arr)
        self.assertEqual(t2.dtype, np.int32)
        multi_arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        t3 = torcetti.tensor(multi_arr)
        self.assertEqual(t3.dtype, np.float64)

    def test_backward_compatibility(self):
        t1 = torcetti.tensor([1, 2, 3], dtype=np.int64)
        self.assertEqual(t1.dtype, np.int64)
        z1 = torcetti.zeros(3, dtype=np.float64)
        self.assertEqual(z1.dtype, np.float64)
        np_arr = np.array([1.0, 2.0], dtype=np.float64)
        t2 = torcetti.tensor(np_arr)
        t3 = t2 + 1.0
        self.assertEqual(t3.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()


