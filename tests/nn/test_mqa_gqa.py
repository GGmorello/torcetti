import numpy as np
import unittest

import torcetti
from torcetti.core.tensor import Tensor


class TestMQA_GQA(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.embed_dim = 16
        self.seq_len = 5
        self.batch = 2

    def _inputs(self, batch_first=False):
        shape = (self.seq_len, self.batch, self.embed_dim) if not batch_first else (self.batch, self.seq_len, self.embed_dim)
        x = np.random.randn(*shape).astype(np.float32)
        t = Tensor(x, requires_grad=True)
        return t.clone(), t.clone(), t.clone()

    def test_mqa_shapes_and_backward(self):
        mha = torcetti.nn.MultiheadAttention(self.embed_dim, num_heads=4, num_kv_heads=1, dropout=0.0, batch_first=False)
        q, k, v = self._inputs(batch_first=False)
        out, attn_w = mha(q, k, v)
        self.assertEqual(out.shape, (self.seq_len, self.batch, self.embed_dim))
        self.assertEqual(attn_w.shape, (self.batch, 4, self.seq_len, self.seq_len))
        (out.sum()).backward()
        for p in mha.parameters():
            self.assertIsNotNone(p.grad)
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_gqa_shapes_and_backward(self):
        mha = torcetti.nn.MultiheadAttention(self.embed_dim, num_heads=8, num_kv_heads=2, dropout=0.0, batch_first=True)
        q, k, v = self._inputs(batch_first=True)
        out, attn_w = mha(q, k, v)
        self.assertEqual(out.shape, (self.batch, self.seq_len, self.embed_dim))
        self.assertEqual(attn_w.shape, (self.batch, 8, self.seq_len, self.seq_len))
        (out.sum()).backward()
        for p in mha.parameters():
            self.assertIsNotNone(p.grad)
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)

    def test_mqa_equivalence_to_replicated_kv(self):
        num_heads = 4
        mha_mqa = torcetti.nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, num_kv_heads=1, dropout=0.0, batch_first=False)
        mha_ref = torcetti.nn.MultiheadAttention(self.embed_dim, num_heads=num_heads, num_kv_heads=num_heads, dropout=0.0, batch_first=False)

        # Align projections: copy weights so only KV sharing differs
        mha_ref.q_proj.weight.data[...] = mha_mqa.q_proj.weight.data
        mha_ref.q_proj.bias.data[...] = mha_mqa.q_proj.bias.data
        mha_ref.out_proj.weight.data[...] = mha_mqa.out_proj.weight.data
        mha_ref.out_proj.bias.data[...] = mha_mqa.out_proj.bias.data
        
        # Replicate MQA's single K/V head to match MHA's multiple heads
        # MQA: (embed_dim, head_dim * 1), MHA: (embed_dim, head_dim * num_heads)
        # We need to repeat the single head's weights for each head position
        mqa_k_weight = mha_mqa.k_proj.weight  # Shape: (16, 4)
        mqa_k_bias = mha_mqa.k_proj.bias      # Shape: (4,)
        mqa_v_weight = mha_mqa.v_proj.weight  # Shape: (16, 4) 
        mqa_v_bias = mha_mqa.v_proj.bias      # Shape: (4,)
        
        # Replicate to fill all head positions
        replicated_k_weight = torcetti.cat([mqa_k_weight] * num_heads, dim=1)  # (16, 16)
        replicated_k_bias = torcetti.cat([mqa_k_bias] * num_heads, dim=0)      # (16,)
        replicated_v_weight = torcetti.cat([mqa_v_weight] * num_heads, dim=1)  # (16, 16)
        replicated_v_bias = torcetti.cat([mqa_v_bias] * num_heads, dim=0)      # (16,)
        
        mha_ref.k_proj.weight.data[...] = replicated_k_weight.data
        mha_ref.k_proj.bias.data[...] = replicated_k_bias.data
        mha_ref.v_proj.weight.data[...] = replicated_v_weight.data
        mha_ref.v_proj.bias.data[...] = replicated_v_bias.data

        q, k, v = self._inputs(batch_first=False)
        out_mqa, _ = mha_mqa(q, k, v)
        out_ref, _ = mha_ref(q, k, v)

        np.testing.assert_allclose(out_mqa.data, out_ref.data, rtol=1e-5, atol=1e-5)

    def test_qk_norm_variants(self):
        for norm in ["l2", "rms"]:
            mha = torcetti.nn.MultiheadAttention(self.embed_dim, num_heads=4, num_kv_heads=1, dropout=0.0, batch_first=False, qk_norm=norm)
            q, k, v = self._inputs(batch_first=False)
            out, attn_w = mha(q, k, v)
            self.assertEqual(out.shape, (self.seq_len, self.batch, self.embed_dim))
            self.assertEqual(attn_w.shape, (self.batch, 4, self.seq_len, self.seq_len))


if __name__ == '__main__':
    unittest.main()




