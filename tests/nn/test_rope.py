import numpy as np

import torcetti
from torcetti.core.tensor import Tensor
from tests.test_helpers import assert_tensors_close


def _numpy_build_rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0, dtype=np.float32):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    pos = np.arange(max_seq_len, dtype=dtype)  # [L]
    inv_freq = base ** (-np.arange(0, half, dtype=dtype) / half)  # [D/2]
    angles = np.outer(pos, inv_freq)  # [L, D/2]
    cos = np.cos(angles)
    sin = np.sin(angles)
    # Duplicate along the last dim so cos/sin have shape [L, D] and
    # even/odd positions share the same cosine/sine value
    cos_full = np.repeat(cos, 2, axis=-1)
    sin_full = np.repeat(sin, 2, axis=-1)
    return cos_full.astype(dtype), sin_full.astype(dtype)


def _numpy_apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, position_ids: np.ndarray):
    # x: [B, H, L, D]; cos/sin: [Lk, D]; position_ids: [L] or [B, L]
    B, H, L, D = x.shape
    if position_ids.ndim == 1:
        pos = position_ids
        # Broadcast to [B, L]
        pos = np.broadcast_to(pos.reshape(1, L), (B, L))
    else:
        pos = position_ids  # [B, L]

    # Gather cos/sin for each (b, l)
    cos_g = cos[pos]  # [B, L, D]
    sin_g = sin[pos]  # [B, L, D]

    # Broadcast across heads
    cos_g = np.expand_dims(cos_g, 1)  # [B, 1, L, D]
    sin_g = np.expand_dims(sin_g, 1)  # [B, 1, L, D]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    c_even = cos_g[..., 0::2]
    s_even = sin_g[..., 0::2]
    # Because we repeated cos/sin, c_odd == c_even and s_odd == s_even

    rot_even = x_even * c_even - x_odd * s_even
    rot_odd = x_odd * c_even + x_even * s_even

    out = np.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class TestRoPE:
    def test_build_rope_cache_shapes_and_values(self):
        L, D = 32, 16
        base = 10000.0
        # Target API: implement these in torcetti.nn.functional
        cos, sin = torcetti.nn.functional.build_rope_cache(
            max_seq_len=L, head_dim=D, base=base
        )
        assert cos.shape == (L, D)
        assert sin.shape == (L, D)

        # Compare to NumPy reference
        ref_cos, ref_sin = _numpy_build_rope_cache(L, D, base=base, dtype=cos.dtype)
        assert_tensors_close(cos, ref_cos)
        assert_tensors_close(sin, ref_sin)

        # Spot-check first position
        np.testing.assert_allclose(cos.data[0], np.ones((D,), dtype=cos.dtype), atol=1e-6)
        np.testing.assert_allclose(sin.data[0], np.zeros((D,), dtype=sin.dtype), atol=1e-6)

        # Even/odd pairs should match per pair
        np.testing.assert_allclose(cos.data[:, 0::2], cos.data[:, 1::2], atol=1e-6)
        np.testing.assert_allclose(sin.data[:, 0::2], sin.data[:, 1::2], atol=1e-6)

    def test_apply_rotary_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        B, H, L, D = 2, 3, 5, 8
        x = rng.standard_normal((B, H, L, D)).astype(np.float32)
        cos, sin = _numpy_build_rope_cache(max_seq_len=L, head_dim=D)
        pos = np.arange(L, dtype=np.int32)

        ref = _numpy_apply_rope(x, cos, sin, pos)

        x_t = Tensor(x, requires_grad=True)
        cos_t = Tensor(cos)
        sin_t = Tensor(sin)
        pos_t = Tensor(pos)

        # Target API: implement apply_rotary_pos_emb in torcetti.nn.functional
        out = torcetti.nn.functional.apply_rotary_pos_emb(x_t, cos_t, sin_t, position_ids=pos_t)
        assert out.shape == x_t.shape
        assert_tensors_close(out, ref, atol=1e-6, rtol=1e-5)

        # Backward should flow through x
        out.sum().backward()
        assert x_t.grad is not None and x_t.grad.data is not None

    def test_rope_for_q_and_k_can_use_different_positions(self):
        # In decode, Q uses positions [past_len .. past_len+L_q-1] while K uses [0 .. L_k-1]
        rng = np.random.default_rng(1)
        B, H, L_q, L_k, D = 2, 2, 3, 7, 12
        xq = rng.standard_normal((B, H, L_q, D)).astype(np.float32)
        xk = rng.standard_normal((B, H, L_k, D)).astype(np.float32)

        cos, sin = _numpy_build_rope_cache(max_seq_len=max(L_q + L_k, L_k), head_dim=D)
        past_len = 4
        q_pos = np.arange(past_len, past_len + L_q, dtype=np.int32)
        k_pos = np.arange(L_k, dtype=np.int32)

        q_ref = _numpy_apply_rope(xq, cos, sin, q_pos)
        k_ref = _numpy_apply_rope(xk, cos, sin, k_pos)

        xq_t = Tensor(xq, requires_grad=True)
        xk_t = Tensor(xk, requires_grad=True)
        cos_t = Tensor(cos)
        sin_t = Tensor(sin)
        q_pos_t = Tensor(q_pos)
        k_pos_t = Tensor(k_pos)

        q_out = torcetti.nn.functional.apply_rotary_pos_emb(xq_t, cos_t, sin_t, position_ids=q_pos_t)
        k_out = torcetti.nn.functional.apply_rotary_pos_emb(xk_t, cos_t, sin_t, position_ids=k_pos_t)

        assert_tensors_close(q_out, q_ref, atol=1e-6, rtol=1e-5)
        assert_tensors_close(k_out, k_ref, atol=1e-6, rtol=1e-5)




