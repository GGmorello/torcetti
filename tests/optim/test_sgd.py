import unittest
import numpy as np
import torch

from torcetti.core.tensor import Tensor
from torcetti.optim.sgd import SGD
from torcetti.nn.linear import Linear


class TestSGDOptimizer(unittest.TestCase):
    def _assert_tensors_close(self, torcetti_tensor, torch_tensor, rtol=1e-6, atol=1e-8):
        np.testing.assert_allclose(torcetti_tensor.data, torch_tensor.detach().numpy(), rtol=rtol, atol=atol)

    def test_single_parameter_step(self):
        np.random.seed(42)
        data = np.random.randn(3, 3).astype(np.float32)
        param = Tensor(data.copy(), requires_grad=True)
        loss = param.sum()
        loss.backward()
        grad_copy = param.grad.copy()
        optimizer = SGD([param], lr=0.1)
        optimizer.step()
        expected = data - 0.1 * grad_copy
        np.testing.assert_allclose(param.data, expected, rtol=1e-6)

    def test_zero_grad(self):
        np.random.seed(0)
        param = Tensor(np.random.randn(5), requires_grad=True)
        (param.sum()).backward()
        optimizer = SGD([param], lr=0.01)
        optimizer.zero_grad()
        self.assertFalse(bool(param.grad))

    def test_module_parameters_update(self):
        np.random.seed(0)
        layer = Linear(4, 2)
        inp = Tensor(np.random.randn(3, 4), requires_grad=True)
        output = layer(inp)
        loss = output.sum()
        loss.backward()
        optimizer = SGD(layer.parameters(), lr=0.05)
        w_before = layer.weight.data.copy()
        b_before = layer.bias.data.copy()
        optimizer.step()
        self.assertFalse(np.array_equal(layer.weight.data, w_before))
        self.assertFalse(np.array_equal(layer.bias.data, b_before))

    def test_multiple_steps(self):
        np.random.seed(42)
        param = Tensor(np.array([5.0, -3.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        loss1 = (param ** 2).sum()
        loss1.backward()
        grad1 = param.grad.copy()
        optimizer.step(); optimizer.zero_grad()
        loss2 = (param ** 2).sum()
        loss2.backward()
        grad2 = param.grad.copy()
        optimizer.step()
        self.assertFalse(np.allclose(grad1, grad2))
        self.assertLess(loss2.data, loss1.data)

    def test_no_gradients_skip(self):
        np.random.seed(42)
        param1 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        param2 = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        loss = param1.sum(); loss.backward()
        optimizer = SGD([param1, param2], lr=0.1)
        param2_before = param2.data.copy()
        optimizer.step()
        np.testing.assert_array_equal(param2.data, param2_before)
        self.assertFalse(np.array_equal(param1.data, np.array([1.0, 2.0])))

    def test_momentum_basic(self):
        np.random.seed(42)
        param = Tensor(np.array([2.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.9)
        initial_value = param.data.copy()
        loss1 = param ** 2; loss1.backward(); grad1 = param.grad.copy(); optimizer.step(); step1_value = param.data.copy(); optimizer.zero_grad()
        loss2 = param ** 2; loss2.backward(); grad2 = param.grad.copy(); optimizer.step(); step2_value = param.data.copy()
        pure_gradient_step = step1_value - 0.1 * grad2
        actual_step2 = step2_value
        self.assertGreater(abs(step1_value - actual_step2), abs(step1_value - pure_gradient_step))

    def test_momentum_direction_consistency(self):
        np.random.seed(42)
        param = Tensor(np.array([10.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.01, momentum=0.9)
        steps = []
        for _ in range(5):
            loss = param ** 2
            loss.backward()
            param_before = param.data.copy()
            optimizer.step()
            step_size = abs(param.data - param_before)
            steps.append(step_size[0])
            optimizer.zero_grad()
        self.assertGreater(steps[-1], steps[0])

    def test_weight_decay(self):
        np.random.seed(42)
        param = Tensor(np.array([2.0, -1.5]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, weight_decay=0.01)
        loss = Tensor(0.0); loss.backward()
        param_before = param.data.copy(); optimizer.step()
        expected = param_before - 0.1 * 0.01 * param_before
        np.testing.assert_allclose(param.data, expected, rtol=1e-6)

    def test_weight_decay_with_gradients(self):
        np.random.seed(42)
        param = Tensor(np.array([1.0, -2.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1, weight_decay=0.01)
        loss = param.sum(); loss.backward()
        param_before = param.data.copy(); grad_before = param.grad.copy(); optimizer.step()
        expected_grad = grad_before + 0.01 * param_before
        expected = param_before - 0.1 * expected_grad
        np.testing.assert_allclose(param.data, expected, rtol=1e-6)

    def test_nesterov_momentum(self):
        np.random.seed(42)
        param1 = Tensor(np.array([3.0]), requires_grad=True)
        param2 = Tensor(np.array([3.0]), requires_grad=True)
        optimizer1 = SGD([param1], lr=0.1, momentum=0.9, nesterov=False)
        optimizer2 = SGD([param2], lr=0.1, momentum=0.9, nesterov=True)
        for _ in range(3):
            loss1 = param1 ** 2; loss2 = param2 ** 2
            loss1.backward(); loss2.backward()
            optimizer1.step(); optimizer2.step()
            optimizer1.zero_grad(); optimizer2.zero_grad()
        self.assertFalse(np.allclose(param1.data, param2.data, rtol=1e-6))

    def test_dampening(self):
        np.random.seed(42)
        param1 = Tensor(np.array([2.0]), requires_grad=True)
        param2 = Tensor(np.array([2.0]), requires_grad=True)
        optimizer1 = SGD([param1], lr=0.1, momentum=0.9, dampening=0.0)
        optimizer2 = SGD([param2], lr=0.1, momentum=0.9, dampening=0.5)
        for _ in range(3):
            loss1 = param1 ** 2; loss2 = param2 ** 2
            loss1.backward(); loss2.backward(); optimizer1.step(); optimizer2.step(); optimizer1.zero_grad(); optimizer2.zero_grad()
        self.assertFalse(np.allclose(param1.data, param2.data, rtol=1e-6))

    def test_learning_rate_effect(self):
        np.random.seed(42)
        param1 = Tensor(np.array([1.0]), requires_grad=True)
        param2 = Tensor(np.array([1.0]), requires_grad=True)
        optimizer1 = SGD([param1], lr=0.01)
        optimizer2 = SGD([param2], lr=0.1)
        loss1 = param1 ** 2; loss2 = param2 ** 2
        loss1.backward(); loss2.backward()
        param1_before = param1.data.copy(); param2_before = param2.data.copy()
        optimizer1.step(); optimizer2.step()
        step1_size = abs(param1.data - param1_before); step2_size = abs(param2.data - param2_before)
        self.assertGreater(step2_size, step1_size)
        self.assertAlmostEqual(step2_size[0] / step1_size[0], 10.0, places=5)

    def test_gradient_accumulation_reset(self):
        np.random.seed(42)
        param = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        optimizer = SGD([param], lr=0.1)
        loss1 = param.sum(); loss1.backward(); grad_after_first = param.grad.copy()
        loss2 = (param * 2).sum(); loss2.backward(); grad_after_second = param.grad.copy()
        expected_accumulated = grad_after_first + np.array([2.0, 2.0])
        np.testing.assert_allclose(grad_after_second, expected_accumulated)
        optimizer.zero_grad(); self.assertFalse(bool(param.grad))
        loss3 = (param * 3).sum(); loss3.backward(); grad_after_reset = param.grad.copy()
        expected_fresh = np.array([3.0, 3.0])
        np.testing.assert_allclose(grad_after_reset, expected_fresh)

    def test_empty_parameter_list(self):
        optimizer = SGD([], lr=0.1)
        optimizer.step(); optimizer.zero_grad()

    def test_generator_parameter_input(self):
        np.random.seed(42)
        layer = Linear(2, 1)
        param_generator = layer.parameters()
        optimizer = SGD(param_generator, lr=0.1)
        inp = Tensor(np.random.randn(1, 2), requires_grad=True)
        output = layer(inp); loss = output.sum(); loss.backward()
        w_before = layer.weight.data.copy(); b_before = layer.bias.data.copy()
        optimizer.step()
        self.assertFalse(np.array_equal(layer.weight.data, w_before))
        self.assertFalse(np.array_equal(layer.bias.data, b_before))

    def test_sgd_parity_no_momentum(self):
        np.random.seed(0)
        data = np.random.randn(4, 4).astype(np.float32)
        param_torcetti = Tensor(data.copy(), requires_grad=True)
        loss_torcetti = (param_torcetti ** 2).sum(); loss_torcetti.backward(); opt_torcetti = SGD([param_torcetti], lr=0.1); opt_torcetti.step()
        param_torch = torch.from_numpy(data.copy()).requires_grad_(True)
        loss_torch = (param_torch ** 2).sum(); loss_torch.backward(); opt_torch = torch.optim.SGD([param_torch], lr=0.1); opt_torch.step()
        self._assert_tensors_close(param_torcetti, param_torch)

    def test_sgd_parity_momentum_weight_decay_nesterov(self):
        np.random.seed(123)
        data = np.random.randn(5).astype(np.float32)
        lr = 0.05; momentum = 0.9; weight_decay = 0.01
        p_t = Tensor(data.copy(), requires_grad=True)
        loss_t = (p_t ** 2).sum(); loss_t.backward(); opt_t = SGD([p_t], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        loss_pt = (p_pt ** 2).sum(); loss_pt.backward(); opt_pt = torch.optim.SGD([p_pt], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True); opt_pt.step()
        self._assert_tensors_close(p_t, p_pt)

    def test_zero_learning_rate(self):
        np.random.seed(7)
        param = Tensor(np.random.randn(3), requires_grad=True)
        original = param.data.copy()
        (param.sum()).backward(); optimizer = SGD([param], lr=0.0); optimizer.step()
        np.testing.assert_array_equal(param.data, original)

    def test_weight_decay_only_repeated(self):
        np.random.seed(11)
        param = Tensor(np.array([10.0]), requires_grad=True)
        weight_decay = 0.1; lr = 0.05
        optimizer = SGD([param], lr=lr, weight_decay=weight_decay)
        expected = param.data.copy()
        for _ in range(5):
            optimizer.step()
            expected = expected - lr * weight_decay * expected
            np.testing.assert_allclose(param.data, expected, rtol=1e-6)

    def test_double_step_same_gradient_parity(self):
        np.random.seed(13)
        data = np.random.randn(4).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        loss_t = (p_t ** 2).sum(); loss_t.backward(); opt_t = SGD([p_t], lr=0.1, momentum=0.9); opt_t.step(); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        loss_pt = (p_pt ** 2).sum(); loss_pt.backward(); opt_pt = torch.optim.SGD([p_pt], lr=0.1, momentum=0.9); opt_pt.step(); opt_pt.step()
        self._assert_tensors_close(p_t, p_pt)


if __name__ == "__main__":
    unittest.main()


