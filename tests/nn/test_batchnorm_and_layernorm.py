import unittest
import numpy as np
import torch
import torcetti

from torcetti.core.tensor import Tensor
from torcetti.nn import functional as F
from torcetti.nn.layer_norm import LayerNorm


class TestBatchNormFunctional(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4; self.features = 3; self.eps = 1e-5; self.momentum = 0.1
        np.random.seed(42)
        self.input_data = np.random.randn(self.batch_size, self.features).astype(np.float32)
        self.running_mean = np.zeros(self.features, dtype=np.float32)
        self.running_var = np.ones(self.features, dtype=np.float32)

    def test_batch_norm_training_mode(self):
        input_tensor = Tensor(self.input_data, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy())
        running_var = Tensor(self.running_var.copy())
        output = F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertEqual(output.shape, input_tensor.shape)
        output_mean = np.mean(output.data, axis=0)
        output_var = np.var(output.data, axis=0, ddof=0)
        np.testing.assert_allclose(output_mean, 0, atol=1e-5)
        np.testing.assert_allclose(output_var, 1, atol=1e-4)
        self.assertFalse(np.allclose(running_mean.data, self.running_mean))
        self.assertFalse(np.allclose(running_var.data, self.running_var))

    def test_batch_norm_eval_mode(self):
        input_tensor = Tensor(self.input_data, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy())
        running_var = Tensor(self.running_var.copy())
        F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        new_input_data = self.input_data * 2 + 1
        new_input_tensor = Tensor(new_input_data, requires_grad=True)
        output = F.batch_norm(new_input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=False)
        expected = (new_input_data - running_mean.data) / np.sqrt(running_var.data + self.eps)
        np.testing.assert_allclose(output.data, expected, rtol=1e-5, atol=1e-5)

    def test_batch_norm_backward(self):
        input_tensor = Tensor(self.input_data, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy())
        running_var = Tensor(self.running_var.copy())
        output = F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        grad_output = np.random.randn(*output.shape).astype(np.float32)
        output.backward(grad_output)
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.data.shape, input_tensor.shape)

    def test_batch_norm_momentum_update(self):
        input_tensor = Tensor(self.input_data, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy())
        running_var = Tensor(self.running_var.copy())
        initial_mean = running_mean.data.copy(); initial_var = running_var.data.copy()
        F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertFalse(np.allclose(running_mean.data, initial_mean))
        self.assertFalse(np.allclose(running_var.data, initial_var))

    def test_batch_norm_eps_parameter(self):
        input_tensor = Tensor(self.input_data, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy())
        running_var = Tensor(self.running_var.copy())
        for eps in [1e-3, 1e-5, 1e-8]:
            output = F.batch_norm(input_tensor, running_mean, running_var, eps=eps, momentum=self.momentum, training=True)
            self.assertTrue(np.all(np.isfinite(output.data)))

    def test_batch_norm_requires_grad(self):
        input_tensor = Tensor(self.input_data, requires_grad=False)
        running_mean = Tensor(self.running_mean.copy()); running_var = Tensor(self.running_var.copy())
        output = F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertFalse(output.requires_grad)
        input_tensor = Tensor(self.input_data, requires_grad=True)
        output = F.batch_norm(input_tensor, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertTrue(output.requires_grad)

    def test_batch_norm_edge_cases(self):
        single_input = Tensor(self.input_data[:1], requires_grad=True)
        running_mean = Tensor(self.running_mean.copy()); running_var = Tensor(self.running_var.copy())
        output = F.batch_norm(single_input, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertEqual(output.shape, single_input.shape)
        small_var_input = Tensor(np.ones((2, 3)) * 1e-8, requires_grad=True)
        running_mean = Tensor(self.running_mean.copy()); running_var = Tensor(self.running_var.copy())
        output = F.batch_norm(small_var_input, running_mean, running_var, eps=self.eps, momentum=self.momentum, training=True)
        self.assertTrue(np.all(np.isfinite(output.data)))


class TestBatchNorm(unittest.TestCase):
    """BatchNorm1d forward statistics and running stats."""

    def setUp(self):
        self.features = 4
        self.batch_size = 16
        self.eps = 1e-5
        np.random.seed(0)
        self.x_data = np.random.randn(self.batch_size, self.features).astype(np.float32)

    def test_batchnorm_training_stats(self):
        bn = torcetti.nn.BatchNorm1d(self.features, eps=self.eps, momentum=0.1)
        x = Tensor(self.x_data)
        y = bn(x)
        self.assertTrue(np.allclose(y.data.mean(axis=0), 0, atol=1e-5))
        self.assertTrue(np.allclose(y.data.var(axis=0, ddof=0), 1, atol=1e-4))
        self.assertFalse(np.allclose(bn.running_mean.data, 0))
        self.assertFalse(np.allclose(bn.running_var.data, 1))

    def test_batchnorm_eval_uses_running(self):
        bn = torcetti.nn.BatchNorm1d(self.features, eps=self.eps, momentum=0.1)
        _ = bn(Tensor(self.x_data))
        bn.eval()
        x_new = Tensor(self.x_data * 2 + 1)
        y_eval = bn(x_new)
        running_mean = bn.running_mean.data
        running_var = bn.running_var.data
        expected = (x_new.data - running_mean) / np.sqrt(running_var + self.eps)
        np.testing.assert_allclose(y_eval.data, expected, rtol=1e-6, atol=1e-6)


class TestLayerNorm(unittest.TestCase):
    def test_layer_norm_initialization(self):
        layer_norm = LayerNorm(normalized_shape=4)
        self.assertEqual(layer_norm.weight.shape, (4,))
        self.assertEqual(layer_norm.bias.shape, (4,))
        self.assertTrue(layer_norm.weight.requires_grad)
        self.assertTrue(layer_norm.bias.requires_grad)
        layer_norm = LayerNorm(normalized_shape=(3, 4))
        self.assertEqual(layer_norm.weight.shape, (3, 4))
        self.assertEqual(layer_norm.bias.shape, (3, 4))
        layer_norm = LayerNorm(normalized_shape=4, eps=1e-6)
        self.assertEqual(layer_norm.eps, 1e-6)

    def test_layer_norm_forward_1d(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x = Tensor(np.random.randn(4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(output.requires_grad)
        self.assertAlmostEqual(abs(np.mean(output.data)), 0, delta=1.0)
        self.assertTrue(0.1 < np.std(output.data) < 10.0)

    def test_layer_norm_forward_2d(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (3, 4))
        for i in range(3):
            sample_output = output.data[i]
            self.assertAlmostEqual(abs(np.mean(sample_output)), 0, delta=2.0)
            self.assertTrue(0.1 < np.std(sample_output) < 10.0)

    def test_layer_norm_forward_3d(self):
        layer_norm = LayerNorm(normalized_shape=(2, 4))
        x = Tensor(np.random.randn(2, 3, 2, 4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (2, 3, 2, 4))

    def test_layer_norm_backward(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        grad_output = Tensor(np.ones_like(output.data), requires_grad=False)
        output.grad += grad_output.data
        output.backward()
        self.assertIsNotNone(x.grad); self.assertIsNotNone(x.grad.data)
        self.assertIsNotNone(layer_norm.weight.grad); self.assertIsNotNone(layer_norm.bias.grad)
        self.assertEqual(x.grad.data.shape, x.data.shape)
        self.assertEqual(layer_norm.weight.grad.data.shape, layer_norm.weight.data.shape)
        self.assertEqual(layer_norm.bias.grad.data.shape, layer_norm.bias.data.shape)

    def test_layer_norm_parameters(self):
        layer_norm = LayerNorm(normalized_shape=4)
        params = list(layer_norm.parameters())
        self.assertEqual(len(params), 2)
        self.assertIn(layer_norm.weight, params)
        self.assertIn(layer_norm.bias, params)

    def test_layer_norm_no_bias(self):
        layer_norm = LayerNorm(normalized_shape=4, bias=False)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (3, 4))
        self.assertIsNone(layer_norm.bias)
        params = list(layer_norm.parameters())
        self.assertEqual(len(params), 1)
        self.assertIn(layer_norm.weight, params)

    def test_layer_norm_no_weight(self):
        layer_norm = LayerNorm(normalized_shape=4, weight=False)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (3, 4))
        self.assertIsNone(layer_norm.weight)
        params = list(layer_norm.parameters())
        self.assertEqual(len(params), 1)
        self.assertIn(layer_norm.bias, params)

    def test_layer_norm_no_weight_no_bias(self):
        layer_norm = LayerNorm(normalized_shape=4, weight=False, bias=False)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (3, 4))
        self.assertIsNone(layer_norm.weight)
        self.assertIsNone(layer_norm.bias)
        params = list(layer_norm.parameters())
        self.assertEqual(len(params), 0)

    def test_layer_norm_elementwise_affine(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), requires_grad=True)
        output = layer_norm(x)
        self.assertEqual(output.shape, (2, 4))
        self.assertFalse(np.allclose(output.data, x.data))

    def test_layer_norm_different_eps(self):
        layer_norm_small_eps = LayerNorm(normalized_shape=4, eps=1e-8)
        layer_norm_large_eps = LayerNorm(normalized_shape=4, eps=1e-3)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output_small = layer_norm_small_eps(x)
        output_large = layer_norm_large_eps(x)
        self.assertFalse(np.allclose(output_small.data, output_large.data))

    def test_layer_norm_gradient_flow(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer_norm(x)
        loss = (output ** 2).sum(); loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(x.grad.data)
        self.assertFalse(np.allclose(x.grad.data, 0))
        self.assertIsNotNone(layer_norm.weight.grad)
        self.assertIsNotNone(layer_norm.bias.grad)
        self.assertFalse(np.allclose(layer_norm.weight.grad.data, 0))
        self.assertFalse(np.allclose(layer_norm.bias.grad.data, 0))

    def test_layer_norm_shape_handling(self):
        layer_norm = LayerNorm(normalized_shape=4)
        x1 = Tensor(np.random.randn(4), requires_grad=True)
        output1 = layer_norm(x1); self.assertEqual(output1.shape, (4,))
        x2 = Tensor(np.random.randn(3, 4), requires_grad=True)
        output2 = layer_norm(x2); self.assertEqual(output2.shape, (3, 4))
        x3 = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        output3 = layer_norm(x3); self.assertEqual(output3.shape, (2, 3, 4))
        layer_norm_tuple = LayerNorm(normalized_shape=(3, 4))
        x4 = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        output4 = layer_norm_tuple(x4); self.assertEqual(output4.shape, (2, 3, 4))


if __name__ == '__main__':
    unittest.main()


