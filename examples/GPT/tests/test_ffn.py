import numpy as np
from torcetti.core.tensor import Tensor
from examples.GPT.feed_forward import FeedForward


class TestFeedForward:
    def test_forward_and_backward(self):
        batch_size = 2
        seq_len = 8
        embed_dim = 32
        hidden_dim = 4 * embed_dim

        ffn = FeedForward(embed_dim, hidden_dim)
        x = Tensor(np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32), requires_grad=True)
        out = ffn(x)
        # Shape should be preserved
        assert out.shape == (batch_size, seq_len, embed_dim)

        out.sum().backward()
        # Input tensor should now have gradients
        assert x.grad is not None
        # All parameters should have gradients
        for p in ffn.parameters():
            assert p.grad is not None

