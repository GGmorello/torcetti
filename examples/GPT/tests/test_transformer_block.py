import numpy as np
from torcetti.core.tensor import Tensor
import sys
import os

from examples.GPT.transformer import TransformerBlock


class TestTransformerBlock:
    def test_forward_pass(self):
        batch_size = 2
        seq_len = 8
        embed_dim = 32
        num_heads = 4

        block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
        x = Tensor(np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32), requires_grad=True)
        out = block(x)

        # Output should keep the same shape as input
        assert out.shape == (batch_size, seq_len, embed_dim)

        # Gradients should flow
        out.sum().backward()
        assert x.grad is not None

