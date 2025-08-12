import unittest
import numpy as np
import torch

from torcetti.core.tensor import Tensor
from torcetti.optim.adamw import AdamW
from torcetti.nn.linear import Linear


class TestAdamWOptimizer(unittest.TestCase):
    def _assert_close(self, torcetti_tensor, torch_tensor, rtol=1e-5, atol=1e-7):
        np.testing.assert_allclose(
            torcetti_tensor.data,
            torch_tensor.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )

    def test_adamw_basic_step(self):
        np.random.seed(0); torch.manual_seed(0)
        data = np.random.randn(4, 3).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        loss_t = (p_t ** 2).sum(); loss_t.backward(); opt_t = AdamW([p_t], lr=0.001, weight_decay=0.0); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        loss_pt = (p_pt ** 2).sum(); loss_pt.backward(); opt_pt = torch.optim.AdamW([p_pt], lr=0.001, weight_decay=0.0); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_adamw_weight_decay(self):
        np.random.seed(42); torch.manual_seed(42)
        data = np.random.randn(3, 4).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        (p_t ** 2).sum().backward(); opt_t = AdamW([p_t], lr=0.01, weight_decay=0.1); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        (p_pt ** 2).sum().backward(); opt_pt = torch.optim.AdamW([p_pt], lr=0.01, weight_decay=0.1); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_adamw_betas_eps(self):
        np.random.seed(123); torch.manual_seed(123)
        data = np.random.randn(2, 5).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        (p_t ** 2).sum().backward(); opt_t = AdamW([p_t], lr=0.005, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.05); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        (p_pt ** 2).sum().backward(); opt_pt = torch.optim.AdamW([p_pt], lr=0.005, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.05); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_adamw_multiple_steps(self):
        np.random.seed(999); torch.manual_seed(999)
        data = np.random.randn(3, 3).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        opt_t = AdamW([p_t], lr=0.01, weight_decay=0.1)
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        opt_pt = torch.optim.AdamW([p_pt], lr=0.01, weight_decay=0.1)
        for _ in range(5):
            opt_t.zero_grad(); (p_t ** 2).sum().backward(); opt_t.step()
            opt_pt.zero_grad(); (p_pt ** 2).sum().backward(); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_adamw_parameter_groups(self):
        np.random.seed(456); torch.manual_seed(456)
        data1 = np.random.randn(2, 3).astype(np.float32)
        data2 = np.random.randn(3, 2).astype(np.float32)
        p1_t = Tensor(data1.copy(), requires_grad=True)
        p2_t = Tensor(data2.copy(), requires_grad=True)
        opt_t = AdamW([
            {'params': [p1_t], 'lr': 0.01, 'weight_decay': 0.1},
            {'params': [p2_t], 'lr': 0.02, 'weight_decay': 0.05}
        ])
        p1_pt = torch.from_numpy(data1.copy()).requires_grad_(True)
        p2_pt = torch.from_numpy(data2.copy()).requires_grad_(True)
        opt_pt = torch.optim.AdamW([
            {'params': [p1_pt], 'lr': 0.01, 'weight_decay': 0.1},
            {'params': [p2_pt], 'lr': 0.02, 'weight_decay': 0.05}
        ])
        loss_t = (p1_t ** 2).sum() + (p2_t ** 2).sum(); loss_t.backward(); opt_t.step()
        loss_pt = (p1_pt ** 2).sum() + (p2_pt ** 2).sum(); loss_pt.backward(); opt_pt.step()
        self._assert_close(p1_t, p1_pt); self._assert_close(p2_t, p2_pt)

    def test_adamw_with_module(self):
        np.random.seed(789); torch.manual_seed(789)
        input_data = np.random.randn(4, 5).astype(np.float32)
        weight_init = np.random.randn(5, 3).astype(np.float32)
        bias_init = np.random.randn(3).astype(np.float32)
        layer_t = Linear(5, 3); layer_t.weight.data = weight_init.copy(); layer_t.bias.data = bias_init.copy()
        opt_t = AdamW(layer_t.parameters(), lr=0.01, weight_decay=0.1)
        layer_pt = torch.nn.Linear(5, 3); layer_pt.weight.data = torch.from_numpy(weight_init.T); layer_pt.bias.data = torch.from_numpy(bias_init)
        opt_pt = torch.optim.AdamW(layer_pt.parameters(), lr=0.01, weight_decay=0.1)
        out_t = layer_t(Tensor(input_data, requires_grad=True)); (out_t ** 2).sum().backward(); opt_t.step()
        out_pt = layer_pt(torch.from_numpy(input_data).requires_grad_(True)); (out_pt ** 2).sum().backward(); opt_pt.step()
        self._assert_close(layer_t.weight, layer_pt.weight.T); self._assert_close(layer_t.bias, layer_pt.bias)

    def test_adamw_zero_grad(self):
        np.random.seed(111)
        data = np.random.randn(3, 3).astype(np.float32)
        p = Tensor(data, requires_grad=True); opt = AdamW([p], lr=0.01)
        loss = (p ** 2).sum(); loss.backward(); self.assertTrue(p.grad.data is not None)
        opt.zero_grad(); self.assertFalse(bool(p.grad))

    def test_adamw_amsgrad(self):
        np.random.seed(888); torch.manual_seed(888)
        data = np.random.randn(2, 4).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True); opt_t = AdamW([p_t], lr=0.01, amsgrad=True, weight_decay=0.1)
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True); opt_pt = torch.optim.AdamW([p_pt], lr=0.01, amsgrad=True, weight_decay=0.1)
        for _ in range(3):
            opt_t.zero_grad(); (p_t ** 2).sum().backward(); opt_t.step()
            opt_pt.zero_grad(); (p_pt ** 2).sum().backward(); opt_pt.step()
        self._assert_close(p_t, p_pt)


if __name__ == '__main__':
    unittest.main()


