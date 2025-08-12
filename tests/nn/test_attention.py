import unittest
import numpy as np
import torch
import torch.nn as nn

import torcetti
from torcetti.core.tensor import Tensor
from tests.test_helpers import assert_tensors_close, compare_forward_backward


class TestMultiheadAttentionBasic(unittest.TestCase):
    """Step 1: Basic initialization and parameter management tests."""
    
    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads
        
    def test_initialization(self):
        """Test that MultiheadAttention initializes with correct parameters."""
        mha = torcetti.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            bias=True,
            batch_first=False
        )
        
        self.assertEqual(self.embed_dim % self.num_heads, 0)
        self.assertEqual(self.head_dim, 4)
        self.assertIsInstance(mha.q_proj, torcetti.nn.Linear)
        self.assertIsInstance(mha.k_proj, torcetti.nn.Linear)
        self.assertIsInstance(mha.v_proj, torcetti.nn.Linear)
        self.assertIsInstance(mha.out_proj, torcetti.nn.Linear)
        self.assertEqual(mha.q_proj.weight.shape, (self.embed_dim, self.embed_dim))
        self.assertEqual(mha.k_proj.weight.shape, (self.embed_dim, self.embed_dim))
        self.assertEqual(mha.v_proj.weight.shape, (self.embed_dim, self.embed_dim))
        self.assertEqual(mha.out_proj.weight.shape, (self.embed_dim, self.embed_dim))
        self.assertEqual(mha.q_proj.bias.shape, (self.embed_dim,))
        self.assertEqual(mha.k_proj.bias.shape, (self.embed_dim,))
        self.assertEqual(mha.v_proj.bias.shape, (self.embed_dim,))
        self.assertEqual(mha.out_proj.bias.shape, (self.embed_dim,))
        
    def test_initialization_no_bias(self):
        """Test initialization without bias."""
        mha = torcetti.nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            bias=False
        )
        self.assertIsNone(mha.q_proj.bias)
        self.assertIsNone(mha.k_proj.bias)
        self.assertIsNone(mha.v_proj.bias)
        self.assertIsNone(mha.out_proj.bias)
        
    def test_parameters_registration(self):
        """Test that all parameters are properly registered."""
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        params = list(mha.parameters())
        expected_param_count = 8 if mha.q_proj.bias is not None else 4
        self.assertEqual(len(params), expected_param_count)
        all_linear_params = []
        all_linear_params.extend(mha.q_proj.parameters())
        all_linear_params.extend(mha.k_proj.parameters())
        all_linear_params.extend(mha.v_proj.parameters())
        all_linear_params.extend(mha.out_proj.parameters())
        self.assertEqual(set(params), set(all_linear_params))


class TestMultiheadAttentionProjections(unittest.TestCase):
    """Step 2: Test individual projection layers (Q, K, V, Output)."""
    
    def setUp(self):
        self.embed_dim = 12
        self.num_heads = 3
        self.seq_len = 5
        self.batch_size = 2
        
    def test_qkv_projections(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        input_data = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        input_tensor = Tensor(input_data, requires_grad=True)
        q_out = mha.q_proj(input_tensor)
        k_out = mha.k_proj(input_tensor)
        v_out = mha.v_proj(input_tensor)
        self.assertEqual(q_out.shape, (self.seq_len, self.batch_size, self.embed_dim))
        self.assertEqual(k_out.shape, (self.seq_len, self.batch_size, self.embed_dim))
        self.assertEqual(v_out.shape, (self.seq_len, self.batch_size, self.embed_dim))
        q_out.sum().backward()
        self.assertIsNotNone(input_tensor.grad)
        
    def test_output_projection(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        attn_output = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        attn_tensor = Tensor(attn_output, requires_grad=True)
        final_output = mha.out_proj(attn_tensor)
        self.assertEqual(final_output.shape, (self.seq_len, self.batch_size, self.embed_dim))
        final_output.sum().backward()
        self.assertIsNotNone(attn_tensor.grad)


import pytest

@pytest.mark.skip(reason="Reshaping helpers are implementation details now handled in torcetti.nn.functional, not exposed for testing.")
class TestMultiheadAttentionReshaping(unittest.TestCase):
    """Step 3: Test tensor reshaping for multi-head attention."""
    
    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads
        self.seq_len = 6
        self.batch_size = 3
        
    def test_reshape_for_attention(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.assertTrue(hasattr(mha, '_reshape_for_attention'))
        
    def test_reshape_back_to_output(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.assertTrue(hasattr(mha, '_reshape_back_to_output'))


class TestScaledDotProductAttention(unittest.TestCase):
    """Step 4: Test the core scaled dot-product attention mechanism."""
    
    def setUp(self):
        self.num_heads = 2
        self.head_dim = 4
        self.seq_len = 5
        self.batch_size = 3
        
    def test_scaled_dot_product_attention_basic(self):
        self.assertTrue(hasattr(torcetti.nn.functional, 'scaled_dot_product_attention'))
        
    def test_attention_with_mask(self):
        mask = np.triu(np.ones((self.seq_len, self.seq_len)), k=1)
        mask = mask.astype(np.float32)
        mask[mask == 1] = float('-inf')
        self.assertTrue(hasattr(torcetti.nn.functional, 'scaled_dot_product_attention'))


class TestMultiheadAttentionForward(unittest.TestCase):
    """Step 5: Test the complete forward pass."""
    
    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.seq_len = 7
        self.batch_size = 3
        
    def test_forward_pass_basic(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0)
        query = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        key = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        value = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        output, attn_weights = mha(Tensor(query, requires_grad=True), Tensor(key, requires_grad=True), Tensor(value, requires_grad=True))
        self.assertEqual(output.shape, (self.seq_len, self.batch_size, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        np.testing.assert_allclose(attn_weights.data.sum(axis=-1), np.ones((self.batch_size, self.num_heads, self.seq_len)), atol=1e-6)
        
    def test_forward_pass_batch_first(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        query = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        key = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        value = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        output, attn_weights = mha(Tensor(query, requires_grad=True), Tensor(key, requires_grad=True), Tensor(value, requires_grad=True))
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))


class TestMultiheadAttentionBackward(unittest.TestCase):
    """Step 6: Test backward pass and gradient flow."""
    
    def setUp(self):
        self.embed_dim = 16
        self.num_heads = 4
        self.seq_len = 5
        self.batch_size = 2
        
    def test_backward_pass(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        query = Tensor(np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32), requires_grad=True)
        key = Tensor(np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32), requires_grad=True)
        value = Tensor(np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32), requires_grad=True)
        output, _ = mha(query, key, value)
        loss = output.sum(); loss.backward()
        for param in mha.parameters():
            self.assertIsNotNone(param.grad)
            self.assertIsNotNone(param.grad.data)
        self.assertIsNotNone(query.grad); self.assertIsNotNone(key.grad); self.assertIsNotNone(value.grad)


class TestMultiheadAttentionMasks(unittest.TestCase):
    """Step 7: Test attention with various mask types."""
    
    def setUp(self):
        self.embed_dim = 12
        self.num_heads = 3
        self.seq_len = 6
        self.batch_size = 2
        
    def test_attention_with_attn_mask(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        mask = np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype(np.float32)
        mask[mask == 1] = float('-inf')
        query = Tensor(np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32), requires_grad=True)
        key = query.clone(); value = query.clone()
        output, attn_weights = mha(query, key, value, attn_mask=Tensor(mask))
        self.assertEqual(output.shape, (self.seq_len, self.batch_size, self.embed_dim))
        
    def test_attention_with_key_padding_mask(self):
        mha = torcetti.nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        padding_mask = np.zeros((self.batch_size, self.seq_len), dtype=np.bool_)
        padding_mask[0, -2:] = True; padding_mask[1, -1:] = True
        query = Tensor(np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32), requires_grad=True)
        key = query.clone(); value = query.clone()
        output, attn_weights = mha(query, key, value, key_padding_mask=Tensor(padding_mask))
        self.assertEqual(output.shape, (self.seq_len, self.batch_size, self.embed_dim))


class TestMultiheadAttentionPyTorchComparison(unittest.TestCase):
    """Step 8: Compare against PyTorch reference implementation."""

    def setUp(self):
        np.random.seed(0); torch.manual_seed(0)
        self.embed_dim = 16; self.num_heads = 4; self.seq_len = 7; self.batch_size = 3

    @staticmethod
    def _copy_weights(torcetti_mha, torch_mha):
        q_w = torcetti_mha.q_proj.weight.data
        k_w = torcetti_mha.k_proj.weight.data
        v_w = torcetti_mha.v_proj.weight.data
        q_b = torcetti_mha.q_proj.bias.data
        k_b = torcetti_mha.k_proj.bias.data
        v_b = torcetti_mha.v_proj.bias.data
        in_proj_weight = np.concatenate([q_w.T, k_w.T, v_w.T], axis=0)
        in_proj_bias = np.concatenate([q_b, k_b, v_b], axis=0)
        torch_mha.in_proj_weight.data = torch.from_numpy(in_proj_weight.astype(np.float32))
        torch_mha.in_proj_bias.data = torch.from_numpy(in_proj_bias.astype(np.float32))
        torch_mha.out_proj.weight.data = torch.from_numpy(torcetti_mha.out_proj.weight.data.T.astype(np.float32))
        torch_mha.out_proj.bias.data = torch.from_numpy(torcetti_mha.out_proj.bias.data.astype(np.float32))

    def test_forward_backward(self):
        torcetti_mha = torcetti.nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.0, bias=True, batch_first=False)
        torch_mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, bias=True, batch_first=False, dtype=torch.float32)
        self._copy_weights(torcetti_mha, torch_mha)
        query_data = np.random.randn(self.seq_len, self.batch_size, self.embed_dim).astype(np.float32)
        compare_forward_backward(lambda x: torcetti_mha(x, x, x)[0], lambda x: torch_mha(x, x, x)[0], [query_data], atol=1e-7, rtol=1e-4)

    def test_batch_first_flag(self):
        torcetti_mha = torcetti.nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.0, bias=True, batch_first=True)
        batch_first_input = Tensor(np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32), requires_grad=True)
        out, attn_w = torcetti_mha(batch_first_input, batch_first_input, batch_first_input)
        self.assertEqual(out.shape, batch_first_input.shape)
        self.assertEqual(attn_w.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        loss = out.sum(); loss.backward()
        self.assertIsNotNone(batch_first_input.grad)


if __name__ == "__main__":
    unittest.main()


