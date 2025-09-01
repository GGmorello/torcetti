import unittest
import numpy as np
import torcetti
from torcetti.core.tensor import Tensor
from torcetti.nn import functional as F


class TestWindowedAttentionMasks(unittest.TestCase):
    """Test windowed attention mask generation."""
    
    def test_create_causal_window_mask(self):
        """Test creation of causal window mask for autoregressive models."""
        seq_len = 8
        window_size = 3
        
        # Should create an additive mask where positions outside window get -inf
        # and positions within window get 0.0
        mask = F.create_window_mask(seq_len, window_size, causal=True)
        
        # Shape should be [seq_len, seq_len]
        self.assertEqual(mask.shape, (seq_len, seq_len), msg="Mask should be of shape [seq_len, seq_len]")
        
        # Should be float dtype for additive masking
        self.assertTrue(np.issubdtype(mask.dtype, np.floating), msg="Mask should be float dtype for additive masking")
        
        # Position 0 should only attend to itself (0.0), mask others (-inf)
        expected_0 = [0.0, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9]
        np.testing.assert_allclose(mask.data[0], expected_0, err_msg="Position 0 should only attend to itself")
        
        # Position 3 should attend to positions 1, 2, 3 (window_size = 3)
        expected_3 = [-1e9, 0.0, 0.0, 0.0, -1e9, -1e9, -1e9, -1e9]
        np.testing.assert_allclose(mask.data[3], expected_3, err_msg="Position 3 should attend to positions 1, 2, 3")
        
        # Last position should attend to last 3 positions  
        expected_7 = [-1e9, -1e9, -1e9, -1e9, -1e9, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(mask.data[7], expected_7, err_msg="Last position should attend to last 3 positions")
    
    def test_create_bidirectional_window_mask(self):
        """Test creation of bidirectional window mask."""
        seq_len = 8
        window_size = 3
        
        mask = F.create_window_mask(seq_len, window_size, causal=False)
        
        # Shape should be [seq_len, seq_len]
        self.assertEqual(mask.shape, (seq_len, seq_len), msg="Mask should be of shape [seq_len, seq_len]")
        
        # Position 0 should attend to positions 0, 1, 2 (window extends right to maintain size)
        expected_0 = [0.0, 0.0, 0.0, -1e9, -1e9, -1e9, -1e9, -1e9]
        np.testing.assert_allclose(mask.data[0], expected_0, err_msg="Position 0 should attend to positions 0, 1, 2")
        
        # Position 3 should attend to positions 2, 3, 4 (centered window)
        expected_3 = [-1e9, -1e9, 0.0, 0.0, 0.0, -1e9, -1e9, -1e9]
        np.testing.assert_allclose(mask.data[3], expected_3, err_msg="Position 3 should attend to positions 2, 3, 4")
        
        # Last position should attend to positions 5, 6, 7 (window extends left to maintain size)
        expected_7 = [-1e9, -1e9, -1e9, -1e9, -1e9, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(mask.data[7], expected_7, err_msg="Last position should attend to positions 5, 6, 7")
    
    def test_window_mask_edge_cases(self):
        """Test edge cases for window mask creation."""
        # Window size larger than sequence length
        mask = F.create_window_mask(4, 10, causal=True)
        # Should create lower triangular mask (standard causal) - no masking within triangle
        expected = np.triu(np.full((4, 4), -1e9, dtype=np.float32), k=1)  # Upper triangle gets -1e9
        np.testing.assert_allclose(mask.data, expected, err_msg="Window size larger than sequence length should create causal mask")
        
        # Window size of 1  
        mask = F.create_window_mask(4, 1, causal=True)
        # Should create diagonal mask - only attend to self
        expected = np.full((4, 4), -1e9, dtype=np.float32)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(mask.data, expected, err_msg="Window size of 1 should create diagonal mask")


class TestWindowedAttentionUsage(unittest.TestCase):
    """Test using windowed attention with the existing scaled_dot_product_attention function."""
    
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 8
        self.head_dim = 16
        self.window_size = 3
        
        # Create Q, K, V tensors
        self.q = torcetti.randn(self.batch_size * self.num_heads, self.seq_len, self.head_dim)
        self.k = torcetti.randn(self.batch_size * self.num_heads, self.seq_len, self.head_dim)
        self.v = torcetti.randn(self.batch_size * self.num_heads, self.seq_len, self.head_dim)
    
    def test_windowed_attention_with_causal_mask(self):
        """Test using create_window_mask with scaled_dot_product_attention for causal windowing."""
        # Create causal window mask
        window_mask = F.create_window_mask(self.seq_len, self.window_size, causal=True)
        
        # Use existing scaled_dot_product_attention with the window mask
        output, attn_weights = F.scaled_dot_product_attention(
            self.q, self.k, self.v,
            num_heads=self.num_heads,
            attn_mask=window_mask
        )
        
        # Output should have same shape as input
        self.assertEqual(output.shape, self.q.shape, 
                        msg=f"Windowed attention output shape {output.shape} should match input shape {self.q.shape}")
        
        # Attention weights should be [B*H, L, L]
        expected_attn_shape = (self.batch_size * self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(attn_weights.shape, expected_attn_shape,
                        msg=f"Attention weights shape {attn_weights.shape} should be {expected_attn_shape}")
    
    def test_windowed_attention_sparsity(self):
        """Test that windowed masks create sparse attention patterns."""
        window_mask = F.create_window_mask(self.seq_len, self.window_size, causal=True)
        
        output, attn_weights = F.scaled_dot_product_attention(
            self.q, self.k, self.v,
            num_heads=self.num_heads,
            attn_mask=window_mask
        )
        
        # Check that attention weights are sparse (many zeros)
        non_zero_entries = np.count_nonzero(attn_weights.data[0] > 1e-10)
        
        # With window size 3 and causal=True, we expect roughly:
        # 1 + 2 + 3 + 3 + 3 + 3 + 3 + 3 = 21 non-zero entries per head
        expected_non_zero = 1 + 2 + min(3, 3) + min(4, 3) + min(5, 3) + min(6, 3) + min(7, 3) + min(8, 3)
        self.assertLessEqual(non_zero_entries, expected_non_zero + 5,  # Allow some tolerance
                            msg=f"Windowed attention should be sparse: found {non_zero_entries} non-zero entries, expected ≤ {expected_non_zero + 5}")
    
    def test_windowed_vs_full_attention_equivalence(self):
        """Test that large window is equivalent to full attention."""
        # Create small tensors for exact comparison
        small_seq_len = 4
        q_small = torcetti.randn(1, small_seq_len, self.head_dim)
        k_small = torcetti.randn(1, small_seq_len, self.head_dim)
        v_small = torcetti.randn(1, small_seq_len, self.head_dim)
        
        # Full attention (no mask)
        full_output, full_weights = F.scaled_dot_product_attention(
            q_small, k_small, v_small, num_heads=1
        )
        
        # Large bidirectional window (should be equivalent to full attention)
        large_window_mask = F.create_window_mask(small_seq_len, small_seq_len, causal=False)
        windowed_output, windowed_weights = F.scaled_dot_product_attention(
            q_small, k_small, v_small,
            num_heads=1,
            attn_mask=large_window_mask
        )
        
        # Should be approximately equal
        np.testing.assert_allclose(full_output.data, windowed_output.data, rtol=1e-5,
                                  err_msg="Large window should match full attention output")
        np.testing.assert_allclose(full_weights.data, windowed_weights.data, rtol=1e-5,
                                  err_msg="Large window should match full attention weights")
    
    def test_combining_window_and_padding_masks(self):
        """Test combining windowed masks with other masks (like padding)."""
        # Create window mask
        window_mask = F.create_window_mask(self.seq_len, self.window_size, causal=True)
        
        # Create padding mask (block last position)
        padding_mask = torcetti.zeros((self.seq_len, self.seq_len))
        padding_mask.data[:, -1] = -1e9  # Block TO last position
        
        # Combine masks
        combined_mask = window_mask + padding_mask
        
        # Use combined mask
        output, attn_weights = F.scaled_dot_product_attention(
            self.q, self.k, self.v,
            num_heads=self.num_heads,
            attn_mask=combined_mask
        )
        
        # Verify attention to last position is zero
        self.assertLess(np.max(attn_weights.data[0, :, -1]), 1e-5,
                       msg="Combined mask should zero out attention to last position")
        
        # Verify attention weights still sum to 1 for each query position
        for i in range(self.seq_len):
            attn_sum = np.sum(attn_weights.data[0, i, :])
            self.assertAlmostEqual(attn_sum, 1.0, places=5,
                                 msg=f"Attention weights for position {i} should sum to 1")
    
    def test_bidirectional_windowed_attention(self):
        """Test bidirectional windowed attention."""
        # Create bidirectional window mask
        window_mask = F.create_window_mask(self.seq_len, self.window_size, causal=False)
        
        output, attn_weights = F.scaled_dot_product_attention(
            self.q, self.k, self.v,
            num_heads=self.num_heads,
            attn_mask=window_mask
        )
        
        # Verify windowing is working - positions outside window should have zero attention
        for i in range(self.seq_len):
            # For bidirectional, compute window bounds
            start = max(0, min(i - self.window_size // 2, self.seq_len - self.window_size))
            end = min(self.seq_len - 1, start + self.window_size - 1)
            start = max(0, end - self.window_size + 1)
            
            for j in range(self.seq_len):
                if j < start or j > end:
                    # Outside bidirectional window - should be zero
                    self.assertLess(attn_weights.data[0, i, j], 1e-10,
                                   msg=f"Position {i} should not attend to position {j} (outside bidirectional window [{start}:{end}])")
        
        # Output should have correct shape
        self.assertEqual(output.shape, self.q.shape)


class TestWindowedAttentionGradients(unittest.TestCase):
    """Test that windowed attention properly handles gradients."""
    
    def test_windowed_attention_backward(self):
        """Test that gradients flow through windowed attention with masks."""
        seq_len = 6
        head_dim = 8
        window_size = 3
        
        q = torcetti.randn(1, seq_len, head_dim, requires_grad=True)
        k = torcetti.randn(1, seq_len, head_dim, requires_grad=True)
        v = torcetti.randn(1, seq_len, head_dim, requires_grad=True)
        
        # Create window mask and use with existing attention function
        window_mask = F.create_window_mask(seq_len, window_size, causal=True)
        output, _ = F.scaled_dot_product_attention(
            q, k, v,
            num_heads=1,
            attn_mask=window_mask
        )
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are reasonable
        self.assertIsNotNone(q.grad, msg="Query tensor should have gradients after windowed attention backward")
        self.assertIsNotNone(k.grad, msg="Key tensor should have gradients after windowed attention backward")
        self.assertIsNotNone(v.grad, msg="Value tensor should have gradients after windowed attention backward")
        
        # Gradients should be non-zero (attention is differentiable)
        q_grad_magnitude = np.abs(q.grad.data).sum()
        k_grad_magnitude = np.abs(k.grad.data).sum()
        v_grad_magnitude = np.abs(v.grad.data).sum()
        
        self.assertGreater(q_grad_magnitude, 1e-6,
                          msg=f"Query gradients should be non-trivial, got magnitude {q_grad_magnitude}")
        self.assertGreater(k_grad_magnitude, 1e-6,
                          msg=f"Key gradients should be non-trivial, got magnitude {k_grad_magnitude}")
        self.assertGreater(v_grad_magnitude, 1e-6,
                          msg=f"Value gradients should be non-trivial, got magnitude {v_grad_magnitude}")


class TestWindowedAttentionEfficiency(unittest.TestCase):
    """Test efficiency characteristics of windowed attention."""
    
    def test_windowed_vs_full_attention_comparison(self):
        """Test windowed vs full attention for efficiency comparison."""
        # Small sequence - both should work
        small_seq = 16
        window_size = 4
        head_dim = 32
        
        q_small = torcetti.randn(1, small_seq, head_dim)
        k_small = torcetti.randn(1, small_seq, head_dim)
        v_small = torcetti.randn(1, small_seq, head_dim)
        
        # Full attention
        full_output, _ = F.scaled_dot_product_attention(
            q_small, k_small, v_small, num_heads=1
        )
        
        # Windowed attention using mask
        window_mask = F.create_window_mask(small_seq, window_size, causal=True)
        windowed_output, _ = F.scaled_dot_product_attention(
            q_small, k_small, v_small,
            num_heads=1,
            attn_mask=window_mask
        )
        
        # Both should produce valid outputs
        self.assertEqual(full_output.shape, windowed_output.shape,
                        msg=f"Full and windowed attention outputs should have same shape: {full_output.shape} vs {windowed_output.shape}")
        self.assertFalse(np.any(np.isnan(full_output.data)),
                         msg="Full attention output should not contain NaN values")
        self.assertFalse(np.any(np.isnan(windowed_output.data)),
                         msg="Windowed attention output should not contain NaN values")


if __name__ == '__main__':
    unittest.main()
