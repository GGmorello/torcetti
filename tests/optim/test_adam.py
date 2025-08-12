import unittest
import numpy as np
import torch

from torcetti.core.tensor import Tensor
from torcetti.optim.adam import Adam
from torcetti.nn.linear import Linear


class TestAdamOptimizer(unittest.TestCase):
    def _assert_close(self, torcetti_tensor, torch_tensor, rtol=1e-5, atol=1e-7):
        np.testing.assert_allclose(
            torcetti_tensor.data,
            torch_tensor.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )

    def test_adam_basic_step(self):
        np.random.seed(0)
        torch.manual_seed(0)
        data = np.random.randn(4, 3).astype(np.float32)
        p_t = Tensor(data.copy(), requires_grad=True)
        loss_t = (p_t ** 2).sum(); loss_t.backward(); opt_t = Adam([p_t], lr=0.001); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        loss_pt = (p_pt ** 2).sum(); loss_pt.backward(); opt_pt = torch.optim.Adam([p_pt], lr=0.001); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_adam_weight_decay_betas(self):
        np.random.seed(42)
        torch.manual_seed(42)
        data = np.random.randn(5).astype(np.float32)
        lr = 0.01; betas = (0.8, 0.888); weight_decay = 0.05
        p_t = Tensor(data.copy(), requires_grad=True)
        (p_t ** 2).sum().backward(); opt_t = Adam([p_t], lr=lr, betas=betas, weight_decay=weight_decay); opt_t.step()
        p_pt = torch.from_numpy(data.copy()).requires_grad_(True)
        (p_pt ** 2).sum().backward(); opt_pt = torch.optim.Adam([p_pt], lr=lr, betas=betas, weight_decay=weight_decay); opt_pt.step()
        self._assert_close(p_t, p_pt)

    def test_multiple_param_groups(self):
        np.random.seed(1); torch.manual_seed(1)
        layer = Linear(3, 2)
        params_t = [ {"params": [layer.weight], "lr": 0.01}, {"params": [layer.bias], "lr": 0.1} ]
        params_pt = [ {"params": [torch.from_numpy(layer.weight.data.T.astype(np.float32).copy()).requires_grad_(True)], "lr": 0.01}, {"params": [torch.from_numpy(layer.bias.data.astype(np.float32).copy()).requires_grad_(True)], "lr": 0.1} ]
        inp = Tensor(np.random.randn(4, 3), requires_grad=True)
        out_t = layer(inp); loss_t = out_t.sum(); loss_t.backward(); opt_t = Adam(params_t); opt_t.step()
        w_pt, b_pt = params_pt[0]["params"][0], params_pt[1]["params"][0]
        def torch_forward(x): return x @ w_pt.T + b_pt
        x_pt = torch.from_numpy(inp.data.astype(np.float32).copy()).requires_grad_(True)
        out_pt = torch_forward(x_pt); loss_pt = out_pt.sum(); loss_pt.backward(); opt_pt = torch.optim.Adam(params_pt); opt_pt.step()
        self._assert_close(layer.weight, w_pt.T); self._assert_close(layer.bias, b_pt)

    def test_zero_grad_does_not_reset_state(self):
        np.random.seed(5); torch.manual_seed(5)
        param = Tensor(np.random.randn(3), requires_grad=True)
        opt = Adam([param], lr=0.01)
        (param.sum()).backward(); opt.step(); opt.zero_grad()
        (param.sum()).backward(); before = param.data.copy(); opt.step(); after = param.data.copy()
        self.assertFalse(np.array_equal(before, after))


if __name__ == "__main__":
    unittest.main()


