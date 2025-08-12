import unittest
import numpy as np
import torch
import torch.nn as nn

import torcetti
from torcetti.core.tensor import Tensor
from tests.test_helpers import assert_tensors_close, compare_forward_backward


class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):
        np.random.seed(42)
        input_data = np.random.randn(2, 3)
        torcetti_fn = lambda x: torcetti.nn.functional.relu(x)
        torch_fn = lambda x: torch.nn.functional.relu(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])
        torcetti_relu = torcetti.nn.ReLU(); torch_relu = nn.ReLU()
        torcetti_fn = lambda x: torcetti_relu(x); torch_fn = lambda x: torch_relu(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_gelu(self):
        np.random.seed(0)
        input_data = np.random.randn(4, 5).astype(np.float32)
        # Functional GELU
        torcetti_fn = lambda x: torcetti.nn.functional.gelu(x)
        torch_fn = lambda x: torch.nn.functional.gelu(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data], atol=1e-6, rtol=1e-4)
        # Module GELU
        np.random.seed(1)
        input_data = np.random.randn(3, 7).astype(np.float32)
        torcetti_gelu = torcetti.nn.GELU(); torch_gelu = nn.GELU()
        torcetti_fn = lambda x: torcetti_gelu(x); torch_fn = lambda x: torch_gelu(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data], atol=1e-6, rtol=1e-4)

    def test_sigmoid(self):
        np.random.seed(42); input_data = np.random.randn(2, 3)
        torcetti_fn = lambda x: torcetti.nn.functional.sigmoid(x)
        torch_fn = lambda x: torch.nn.functional.sigmoid(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])
        torcetti_sigmoid = torcetti.nn.Sigmoid(); torch_sigmoid = nn.Sigmoid()
        torcetti_fn = lambda x: torcetti_sigmoid(x); torch_fn = lambda x: torch_sigmoid(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_tanh(self):
        np.random.seed(42); input_data = np.random.randn(2, 3)
        torcetti_fn = lambda x: torcetti.nn.functional.tanh(x)
        torch_fn = lambda x: torch.nn.functional.tanh(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])
        torcetti_tanh = torcetti.nn.Tanh(); torch_tanh = nn.Tanh()
        torcetti_fn = lambda x: torcetti_tanh(x); torch_fn = lambda x: torch_tanh(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_softmax(self):
        np.random.seed(42); input_data = np.random.randn(2, 3)
        torcetti_fn = lambda x: torcetti.nn.functional.softmax(x, axis=1)
        torch_fn = lambda x: torch.nn.functional.softmax(x, dim=1)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])
        torcetti_softmax = torcetti.nn.Softmax(axis=1); torch_softmax = nn.Softmax(dim=1)
        torcetti_fn = lambda x: torcetti_softmax(x); torch_fn = lambda x: torch_softmax(x)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_softmax_different_axes(self):
        np.random.seed(42); input_data = np.random.randn(2, 3, 4)
        torcetti_fn = lambda x: torcetti.nn.functional.softmax(x, axis=0)
        torch_fn = lambda x: torch.nn.functional.softmax(x, dim=0)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])
        torcetti_fn = lambda x: torcetti.nn.functional.softmax(x, axis=2)
        torch_fn = lambda x: torch.nn.functional.softmax(x, dim=2)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_log_softmax(self):
        np.random.seed(42); input_data = np.random.randn(2, 3)
        torcetti_fn = lambda x: torcetti.nn.functional.log_softmax(x, axis=1)
        torch_fn = lambda x: torch.nn.functional.log_softmax(x, dim=1)
        compare_forward_backward(torcetti_fn, torch_fn, [input_data])

    def test_activation_properties(self):
        np.random.seed(42); input_data = np.random.randn(100)
        torcetti_tensor = Tensor(input_data)
        relu_output = torcetti.nn.functional.relu(torcetti_tensor)
        self.assertTrue(np.all(relu_output.data >= 0))
        positive_mask = input_data > 0
        np.testing.assert_array_equal(relu_output.data[positive_mask], input_data[positive_mask])
        sigmoid_output = torcetti.nn.functional.sigmoid(torcetti_tensor)
        self.assertTrue(np.all(sigmoid_output.data > 0)); self.assertTrue(np.all(sigmoid_output.data < 1))
        tanh_output = torcetti.nn.functional.tanh(torcetti_tensor)
        self.assertTrue(np.all(tanh_output.data > -1)); self.assertTrue(np.all(tanh_output.data < 1))
        softmax_input = Tensor(input_data.reshape(10, 10))
        softmax_output = torcetti.nn.functional.softmax(softmax_input, axis=1)
        row_sums = np.sum(softmax_output.data, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), rtol=1e-6)
        self.assertTrue(np.all(softmax_output.data > 0))


class TestConv2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.batch_size = 2; self.in_channels = 3; self.out_channels = 4; self.height = 8; self.width = 8; self.kernel_size = 3

    def test_conv2d_forward(self):
        input_data = np.random.randn(self.batch_size, self.in_channels, self.height, self.width).astype(np.float32)
        torcetti_conv = torcetti.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=True)
        torch_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=True)
        weight_data = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).astype(np.float32)
        bias_data = np.random.randn(self.out_channels).astype(np.float32)
        torcetti_conv.weight.data = weight_data; torcetti_conv.bias.data = bias_data
        torch_conv.weight.data = torch.from_numpy(weight_data); torch_conv.bias.data = torch.from_numpy(bias_data)
        compare_forward_backward(lambda x: torcetti_conv(x), lambda x: torch_conv(x), [input_data])

    def test_conv2d_no_bias(self):
        input_data = np.random.randn(self.batch_size, self.in_channels, self.height, self.width).astype(np.float32)
        torcetti_conv = torcetti.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False)
        torch_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False)
        weight_data = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).astype(np.float32)
        torcetti_conv.weight.data = weight_data; torch_conv.weight.data = torch.from_numpy(weight_data)
        compare_forward_backward(lambda x: torcetti_conv(x), lambda x: torch_conv(x), [input_data])

    def test_conv2d_padding(self):
        input_data = np.random.randn(self.batch_size, self.in_channels, self.height, self.width).astype(np.float32)
        torcetti_conv = torcetti.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=1, bias=True)
        torch_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=1, bias=True)
        weight_data = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).astype(np.float32)
        bias_data = np.random.randn(self.out_channels).astype(np.float32)
        torcetti_conv.weight.data = weight_data; torcetti_conv.bias.data = bias_data
        torch_conv.weight.data = torch.from_numpy(weight_data); torch_conv.bias.data = torch.from_numpy(bias_data)
        compare_forward_backward(lambda x: torcetti_conv(x), lambda x: torch_conv(x), [input_data])

    def test_conv2d_stride(self):
        input_data = np.random.randn(self.batch_size, self.in_channels, self.height, self.width).astype(np.float32)
        torcetti_conv = torcetti.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=2, bias=True)
        torch_conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=2, bias=True)
        weight_data = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size).astype(np.float32)
        bias_data = np.random.randn(self.out_channels).astype(np.float32)
        torcetti_conv.weight.data = weight_data; torcetti_conv.bias.data = bias_data
        torch_conv.weight.data = torch.from_numpy(weight_data); torch_conv.bias.data = torch.from_numpy(bias_data)
        compare_forward_backward(lambda x: torcetti_conv(x), lambda x: torch_conv(x), [input_data])


class TestConv2DFunctional(unittest.TestCase):
    def test_conv2d_functional(self):
        input_data = np.random.randn(2, 3, 8, 8).astype(np.float32)
        weight_data = np.random.randn(4, 3, 3, 3).astype(np.float32)
        bias_data = np.random.randn(4).astype(np.float32)
        compare_forward_backward(
            lambda x, w, b: torcetti.nn.functional.conv2d(x, w, b),
            lambda x, w, b: torch.nn.functional.conv2d(x, w, b),
            [input_data, weight_data, bias_data]
        )


class TestPooling(unittest.TestCase):
    def test_avgpool2d_forward_backward(self):
        x = Tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), requires_grad=True)
        pool = torcetti.nn.AvgPool2d(2)
        out = pool(x)
        expected = np.array([[[[2.5, 4.5], [10.5, 12.5]]]], dtype=np.float32)
        np.testing.assert_allclose(out.data, expected)
        loss = out.sum(); loss.backward()
        np.testing.assert_allclose(x.grad.data, np.ones_like(x.data) * 0.25)

    def test_maxpool2d_forward_backward(self):
        data = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
        x = Tensor(data, requires_grad=True)
        pool = torcetti.nn.MaxPool2d(2)
        out = pool(x)
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
        np.testing.assert_allclose(out.data, expected)
        loss = out.sum(); loss.backward()
        grad_expected = np.zeros_like(data)
        grad_expected[0, 0, 1, 1] = 1; grad_expected[0, 0, 1, 3] = 1; grad_expected[0, 0, 3, 1] = 1; grad_expected[0, 0, 3, 3] = 1
        np.testing.assert_allclose(x.grad.data, grad_expected)


class TestNNBasics(unittest.TestCase):
    def test_linear_layer(self):
        np.random.seed(42)
        input_size, output_size = 5, 3; batch_size = 4
        weight_data = np.random.randn(input_size, output_size).astype(np.float32)
        bias_data = np.random.randn(output_size).astype(np.float32)
        torcetti_linear = torcetti.nn.Linear(input_size, output_size)
        torch_linear = nn.Linear(input_size, output_size)
        torcetti_linear.weight.data = weight_data; torcetti_linear.bias.data = bias_data
        torch_linear.weight.data = torch.from_numpy(weight_data.T); torch_linear.bias.data = torch.from_numpy(bias_data)
        input_data = np.random.randn(batch_size, input_size).astype(np.float32)
        compare_forward_backward(lambda x: torcetti_linear(x), lambda x: torch_linear(x), [input_data])

    def test_sequential_network(self):
        np.random.seed(42)
        input_size, hidden_size, output_size = 4, 6, 2; batch_size = 3
        w1_data = np.random.randn(input_size, hidden_size).astype(np.float32)
        b1_data = np.random.randn(hidden_size).astype(np.float32)
        w2_data = np.random.randn(hidden_size, output_size).astype(np.float32)
        b2_data = np.random.randn(output_size).astype(np.float32)
        torcetti_net = torcetti.nn.Sequential(
            torcetti.nn.Linear(input_size, hidden_size),
            torcetti.nn.ReLU(),
            torcetti.nn.Linear(hidden_size, output_size)
        )
        torcetti_net.layers[0].weight.data = w1_data; torcetti_net.layers[0].bias.data = b1_data
        torcetti_net.layers[2].weight.data = w2_data; torcetti_net.layers[2].bias.data = b2_data
        torch_net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
        torch_net[0].weight.data = torch.from_numpy(w1_data.T); torch_net[0].bias.data = torch.from_numpy(b1_data)
        torch_net[2].weight.data = torch.from_numpy(w2_data.T); torch_net[2].bias.data = torch.from_numpy(b2_data)
        input_data = np.random.randn(batch_size, input_size).astype(np.float32)
        compare_forward_backward(lambda x: torcetti_net(x), lambda x: torch_net(x), [input_data])

    def test_network_properties(self):
        np.random.seed(42)
        net = torcetti.nn.Sequential(torcetti.nn.Linear(3, 5), torcetti.nn.ReLU(), torcetti.nn.Linear(5, 2))
        input_data = np.random.randn(4, 3)
        input_tensor = Tensor(input_data, requires_grad=True)
        output = net(input_tensor)
        self.assertEqual(output.shape, (4, 2))
        loss = output.sum(); loss.backward()
        self.assertIsNotNone(net.layers[0].weight.grad); self.assertIsNotNone(net.layers[0].bias.grad)
        self.assertIsNotNone(net.layers[2].weight.grad); self.assertIsNotNone(net.layers[2].bias.grad)
        self.assertIsNotNone(input_tensor.grad)

    def test_parameter_initialization(self):
        np.random.seed(42)
        linear = torcetti.nn.Linear(10, 5)
        self.assertEqual(linear.weight.shape, (10, 5)); self.assertEqual(linear.bias.shape, (5,))
        self.assertFalse(np.allclose(linear.weight.data, 0)); self.assertFalse(np.allclose(linear.bias.data, 0))
        self.assertLess(np.std(linear.weight.data), 1.0); self.assertLess(np.std(linear.bias.data), 1.0)

    def test_batch_processing(self):
        np.random.seed(42)
        linear = torcetti.nn.Linear(3, 2)
        single_input = np.random.randn(1, 3)
        single_output = linear(Tensor(single_input))
        batch_input = np.tile(single_input, (5, 1))
        batch_output = linear(Tensor(batch_input))
        for i in range(5):
            np.testing.assert_allclose(batch_output.data[i], single_output.data[0], rtol=1e-6)

    def test_gradient_accumulation(self):
        np.random.seed(42)
        linear = torcetti.nn.Linear(3, 2)
        input_data = np.random.randn(2, 3)
        output1 = linear(Tensor(input_data, requires_grad=True)); loss1 = output1.sum(); loss1.backward()
        weight_grad_1 = linear.weight.grad.copy(); bias_grad_1 = linear.bias.grad.copy()
        output2 = linear(Tensor(input_data, requires_grad=True)); loss2 = output2.sum(); loss2.backward()
        expected_weight_grad = 2 * weight_grad_1; expected_bias_grad = 2 * bias_grad_1
        np.testing.assert_allclose(linear.weight.grad.data, expected_weight_grad, rtol=1e-6)
        np.testing.assert_allclose(linear.bias.grad.data, expected_bias_grad, rtol=1e-6)


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.p = 0.5
        self.x_data = np.ones((1000,), dtype=np.float32)
        np.random.seed(0)

    def test_dropout_training_zero_ratio(self):
        out = torcetti.nn.functional.dropout(Tensor(self.x_data), p=self.p, training=True)
        zeros = (out.data == 0).sum()
        ratio = zeros / out.data.size
        self.assertTrue(abs(ratio - self.p) < 0.05)
        if ratio < 1.0:
            non_zero_vals = out.data[out.data != 0]
            self.assertTrue(np.allclose(non_zero_vals, 1.0 / (1 - self.p)))

    def test_dropout_eval_identity(self):
        module = torcetti.nn.Dropout(p=self.p)
        module.eval()
        x = Tensor(self.x_data)
        out = module(x)
        assert_tensors_close(out, x)

    def test_dropout_backward_mask(self):
        x = Tensor(self.x_data, requires_grad=True)
        out = torcetti.nn.functional.dropout(x, p=self.p, training=True)
        out.backward(np.ones_like(out.data))
        mask = (out.data != 0).astype(np.float32)
        expected_grad = mask * (1.0 / (1 - self.p))
        assert_tensors_close(x.grad, expected_grad)


if __name__ == '__main__':
    unittest.main()


