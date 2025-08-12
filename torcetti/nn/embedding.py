from torcetti.nn.module import Module
from torcetti.core.parameter import Parameter
from torcetti.core.factories import rand
from torcetti.core.dtype import get_default_dtype
from torcetti.nn import functional as F


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        # Initialise embeddings with random values in the same style as torch.nn.Embedding
        self.weight = Parameter(
            rand((num_embeddings, embedding_dim), dtype=get_default_dtype()).data, requires_grad=True
        )
    
    def forward(self, x):
        return F.embedding(x, self.weight)
