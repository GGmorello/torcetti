import numpy as np

import torcetti
from torcetti.core.tensor import Tensor


def _manual_l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.sqrt((x ** 2).sum(axis=-1, keepdims=True)) + eps
    return x / norm


def _manual_rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms


class TestQKNorm:
    def test_scaled_dot_product_attention_qk_norm_rms_matches_manual(self):
        # Setup random Q, K, V with shape (B*H, L, D)
        B, H, L, D = 2, 3, 5, 4
        rng = np.random.default_rng(3)
        q = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        k = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        v = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)

        # Reference using numpy with RMS-normalized Q and K per head
        qn = _manual_rms_normalize(q.data)
        kn = _manual_rms_normalize(k.data)
        scores = (qn @ kn.transpose(0, 2, 1)) / np.sqrt(D)
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        ref_out = w @ v.data

        # Target API: pass qk_norm="rms" to normalize Q, K internally
        out, _ = torcetti.nn.functional.scaled_dot_product_attention(
            q, k, v, num_heads=H, qk_norm="rms", dropout_p=0.0, training=False
        )

        np.testing.assert_allclose(out.data, ref_out, rtol=1e-5, atol=1e-6)

    def test_scaled_dot_product_attention_qk_norm_l2_matches_manual(self):
        # Setup random Q, K, V with shape (B*H, L, D)
        B, H, L, D = 2, 3, 5, 4
        rng = np.random.default_rng(0)
        q = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        k = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        v = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)

        # Reference using numpy with L2-normalized Q and K per head
        qn = _manual_l2_normalize(q.data)
        kn = _manual_l2_normalize(k.data)
        scores = (qn @ kn.transpose(0, 2, 1)) / np.sqrt(D)
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        ref_out = w @ v.data

        # Target API: pass qk_norm="l2" to normalize Q, K internally
        out, _ = torcetti.nn.functional.scaled_dot_product_attention(
            q, k, v, num_heads=H, qk_norm="l2", dropout_p=0.0, training=False
        )

        np.testing.assert_allclose(out.data, ref_out, rtol=1e-5, atol=1e-6)

    def test_multi_head_attention_qk_norm_shapes_and_prob_sums(self):
        # Use functional MHA directly to avoid depending on module wiring
        embed_dim, num_heads = 16, 4
        L, B = 7, 3
        D = embed_dim // num_heads
        rng = np.random.default_rng(1)

        # Create synthetic Q/K/V already in model space (batch_first)
        x = Tensor(rng.standard_normal((B, L, embed_dim)).astype(np.float32), requires_grad=True)

        # Make a simple linear-like projection by fixed random matrices to get q/k/v
        Wq = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32)
        Wk = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32)
        Wv = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32)

        q = Tensor(x.data @ Wq, requires_grad=True)
        k = Tensor(x.data @ Wk, requires_grad=True)
        v = Tensor(x.data @ Wv, requires_grad=True)

        out, w = torcetti.nn.functional.multi_head_attention(
            q, k, v,
            num_heads=num_heads,
            head_dim=D,
            dropout_p=0.0,
            out_proj=lambda y: y,  # identity to focus on attention core
            attn_mask=None,
            key_padding_mask=None,
            batch_first=True,
            training=False,
            qk_norm="l2",
        )

        assert out.shape == (B, L, embed_dim)
        assert w.shape == (B, num_heads, L, L)
        # Attention probabilities along last dim should sum to 1
        np.testing.assert_allclose(w.data.sum(axis=-1), np.ones((B, num_heads, L)), atol=1e-6)

    def test_qk_norm_reduces_scale_sensitivity(self):
        # Scaling Q without QK norm should change outputs; with QK norm it should not
        B, H, L, D = 1, 2, 4, 8
        rng = np.random.default_rng(2)
        q = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        k = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)
        v = Tensor(rng.standard_normal((B * H, L, D)).astype(np.float32), requires_grad=True)

        out_no_norm, _ = torcetti.nn.functional.scaled_dot_product_attention(
            q * 3.0, k, v, num_heads=H, dropout_p=0.0, training=False
        )
        out_with_norm_scaled, _ = torcetti.nn.functional.scaled_dot_product_attention(
            q * 3.0, k, v, num_heads=H, qk_norm="l2", dropout_p=0.0, training=False
        )
        out_with_norm, _ = torcetti.nn.functional.scaled_dot_product_attention(
            q, k, v, num_heads=H, qk_norm="l2", dropout_p=0.0, training=False
        )

        # With QK norm on, scaling q should barely change the output
        np.testing.assert_allclose(out_with_norm_scaled.data, out_with_norm.data, rtol=1e-4, atol=1e-5)
        # Without QK norm, scaling q changes output noticeably
        assert not np.allclose(out_no_norm.data, out_with_norm.data, rtol=1e-4, atol=1e-5)


