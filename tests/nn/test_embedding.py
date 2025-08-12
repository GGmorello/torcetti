import numpy as np
import pytest
from torcetti.core.tensor import Tensor
from torcetti.nn.embedding import Embedding


class TestEmbedding:
    def test_embedding_initialization(self):
        embedding = Embedding(num_embeddings=1000, embedding_dim=256)
        assert embedding.weight.shape == (1000, 256)
        assert embedding.weight.requires_grad is True
        embedding = Embedding(num_embeddings=500, embedding_dim=128)
        assert embedding.weight.shape == (500, 128)
        assert not np.allclose(embedding.weight.data, 0)

    def test_embedding_forward_basic(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor([3], requires_grad=False)
        output = embedding(indices)
        assert output.shape == (1, 4)
        assert np.allclose(output.data, embedding.weight.data[3])
        indices = Tensor([1, 3, 5], requires_grad=False)
        output = embedding(indices)
        assert output.shape == (3, 4)
        assert np.allclose(output.data[0], embedding.weight.data[1])
        assert np.allclose(output.data[1], embedding.weight.data[3])
        assert np.allclose(output.data[2], embedding.weight.data[5])

    def test_embedding_forward_2d(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor([[1, 2], [3, 4]], requires_grad=False)
        output = embedding(indices)
        assert output.shape == (2, 2, 4)
        assert np.allclose(output.data[0, 0], embedding.weight.data[1])
        assert np.allclose(output.data[0, 1], embedding.weight.data[2])
        assert np.allclose(output.data[1, 0], embedding.weight.data[3])
        assert np.allclose(output.data[1, 1], embedding.weight.data[4])

    def test_embedding_forward_3d(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=False)
        output = embedding(indices)
        assert output.shape == (2, 2, 2, 4)
        assert np.allclose(output.data[0, 0, 0], embedding.weight.data[1])
        assert np.allclose(output.data[0, 0, 1], embedding.weight.data[2])
        assert np.allclose(output.data[1, 1, 1], embedding.weight.data[8])

    def test_embedding_backward(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor([1, 3, 5], requires_grad=False)
        output = embedding(indices)
        grad_output = Tensor(np.ones_like(output.data), requires_grad=False)
        output.grad += grad_output.data
        output.backward()
        assert embedding.weight.grad is not None and embedding.weight.grad.data is not None
        assert np.allclose(embedding.weight.grad.data[1], 1.0)
        assert np.allclose(embedding.weight.grad.data[3], 1.0)
        assert np.allclose(embedding.weight.grad.data[5], 1.0)
        assert np.allclose(embedding.weight.grad.data[0], 0.0)
        assert np.allclose(embedding.weight.grad.data[2], 0.0)
        assert np.allclose(embedding.weight.grad.data[4], 0.0)

    def test_embedding_parameters(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        params = list(embedding.parameters())
        assert len(params) == 1
        assert params[0] is embedding.weight

    def test_embedding_out_of_bounds(self):
        embedding = Embedding(num_embeddings=5, embedding_dim=4)
        indices = Tensor([6], requires_grad=False)
        with pytest.raises(IndexError):
            embedding(indices)

    def test_embedding_negative_indices(self):
        embedding = Embedding(num_embeddings=5, embedding_dim=4)
        indices = Tensor([-1], requires_grad=False)
        with pytest.raises(IndexError):
            embedding(indices)

    def test_embedding_gradient_accumulation(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices = Tensor([1, 1, 1], requires_grad=False)
        output = embedding(indices)
        grad_output = Tensor(np.ones_like(output.data), requires_grad=False)
        output.grad += grad_output.data
        output.backward()
        assert np.allclose(embedding.weight.grad.data[1], 3.0)

    def test_embedding_different_dtypes(self):
        embedding = Embedding(num_embeddings=10, embedding_dim=4)
        indices_int32 = Tensor([1, 2, 3], requires_grad=False, dtype=np.int32)
        output_int32 = embedding(indices_int32)
        assert output_int32.shape == (3, 4)
        indices_int64 = Tensor([1, 2, 3], requires_grad=False, dtype=np.int64)
        output_int64 = embedding(indices_int64)
        assert output_int64.shape == (3, 4)
        assert np.allclose(output_int32.data, output_int64.data)


