"""Tests for Parameter class, Module state, and parameter traversal."""

import unittest
import numpy as np
from torcetti.core.tensor import Tensor
from torcetti.core.parameter import Parameter
from torcetti.nn.linear import Linear
from torcetti.nn.module import Module


class TestParameter(unittest.TestCase):
    def test_requires_grad_default(self):
        p = Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertTrue(p.requires_grad)
        self.assertIsInstance(p, Tensor)

    def test_requires_grad_override(self):
        p = Parameter(np.array([1.0, 2.0], dtype=np.float32), requires_grad=False)
        self.assertFalse(p.requires_grad)


class TestStateDict(unittest.TestCase):
    def test_linear_state_dict_roundtrip(self):
        np.random.seed(0)
        layer_src = Linear(4, 3)
        weight_src = layer_src.weight.data.copy(); bias_src = layer_src.bias.data.copy()
        state = layer_src.state_dict()
        layer_tgt = Linear(4, 3)
        layer_tgt.weight.data = np.zeros_like(layer_tgt.weight.data)
        layer_tgt.bias.data = np.zeros_like(layer_tgt.bias.data)
        layer_tgt.load_state_dict(state)
        np.testing.assert_allclose(layer_tgt.weight.data, weight_src)
        np.testing.assert_allclose(layer_tgt.bias.data, bias_src)

    

    def test_buffers_and_nonregistered_tensors(self):
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                # Parameters must be Parameter to be included in state_dict (PyTorch behavior)
                self.weight = Parameter(np.random.randn(2, 2).astype(np.float32), requires_grad=True)
                self.bias = Parameter(np.random.randn(2).astype(np.float32), requires_grad=True)
                # Plain attribute tensor (not registered) should NOT appear
                self.tmp_tensor = Tensor(np.random.randn(2), requires_grad=False)
                # Properly registered buffer SHOULD appear
                self.register_buffer('running_mean', Tensor(np.random.randn(2), requires_grad=False))

        module = TestModule()
        state = module.state_dict()
        self.assertIn('weight', state)
        self.assertIn('bias', state)
        self.assertIn('running_mean', state)
        self.assertNotIn('tmp_tensor', state)

    def test_parameter_saved_even_if_requires_grad_false(self):
        np.random.seed(0)
        layer = Linear(3, 2)
        # Turn off grad on parameters; they should still be saved
        layer.weight.requires_grad = False
        layer.bias.requires_grad = False
        state = layer.state_dict()
        self.assertIn('weight', state)
        self.assertIn('bias', state)

    def test_load_state_dict_preserves_object_identity(self):
        layer = Linear(2, 2)
        original_weight = layer.weight; original_bias = layer.bias
        new_state = { 'weight': Tensor(np.ones((2, 2)), requires_grad=True), 'bias': Tensor(np.zeros(2), requires_grad=True) }
        layer.load_state_dict(new_state)
        self.assertIs(layer.weight, original_weight); self.assertIs(layer.bias, original_bias)
        np.testing.assert_allclose(layer.weight.data, np.ones((2, 2)))
        np.testing.assert_allclose(layer.bias.data, np.zeros(2))

    def test_load_state_dict_incompatible_keys_and_strict(self):
        layer = Linear(2, 2)
        # Unexpected key shouldn't raise when strict=False; should be reported in unexpected_keys
        result = layer.load_state_dict({'nonexistent': Tensor(np.zeros(1))}, strict=False)
        # Expect structure with missing_keys and unexpected_keys (list-like)
        if isinstance(result, dict):
            unexpected = result.get('unexpected_keys', [])
            missing = result.get('missing_keys', [])
        else:
            unexpected = getattr(result, 'unexpected_keys')
            missing = getattr(result, 'missing_keys')
        self.assertIn('nonexistent', unexpected)
        # Missing key when partial loading
        state = {'weight': layer.weight.clone()}
        result2 = layer.load_state_dict(state, strict=False)
        if isinstance(result2, dict):
            missing2 = result2.get('missing_keys', [])
        else:
            missing2 = getattr(result2, 'missing_keys')
        self.assertIn('bias', missing2)

    def test_shape_mismatch_raises(self):
        layer = Linear(3, 2)
        bad_state = {
            'weight': Tensor(np.zeros((99, 99), dtype=np.float32), requires_grad=True),
            'bias': layer.bias.clone(),
        }
        with self.assertRaises(RuntimeError):
            layer.load_state_dict(bad_state)

    def test_parameter_inheritance(self):
        data = np.random.randn(3, 4)
        tensor = Tensor(data, requires_grad=True)
        param = Parameter(data, requires_grad=True)
        tensor_result = tensor + tensor
        param_result = param + param
        np.testing.assert_allclose(tensor_result.data, param_result.data)
        self.assertEqual(tensor_result.requires_grad, param_result.requires_grad)
class TestModuleParameterTraversal(unittest.TestCase):
    def test_parameters_yield_only_parameters(self):
        class M(Module):
            def __init__(self):
                super().__init__()
                self.p = Parameter(np.random.randn(2, 2).astype(np.float32), requires_grad=False)
                self.t = Tensor(np.random.randn(3, 3), requires_grad=True)
            def forward(self, x):
                return x
        m = M()
        params = list(m.parameters())
        # Only the Parameter should be yielded
        self.assertEqual(params, [m.p])

    


if __name__ == '__main__':
    unittest.main()


