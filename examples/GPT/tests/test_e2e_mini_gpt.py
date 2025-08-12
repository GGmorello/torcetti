import numpy as np
import torch
import torch.nn as nn
import pytest

from torcetti.core.tensor import Tensor
from torcetti.optim.adamw import AdamW
from torcetti.loss.crossentropy import CrossEntropyLoss
from examples.GPT.gpt_model import GPT

# Mapping from PyTorch dtypes to numpy dtypes
TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.bool: np.bool_,
}

from tests.test_helpers import compare_forward_backward, assert_tensors_close
from torcetti.nn.attention import MultiheadAttention  # type: ignore


class TorchMiniGPT(nn.Module):
    """A minimal PyTorch reference implementation that mirrors the tiny GPT used in
    this repo. It is intentionally *very* small and only supports the few
    features required by the tests below (batch-first tensors, causal mask,
    standard transformer blocks, tied / untied head, etc.).
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):  # x: (B, L)
        device = x.device
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        # Apply causal mask to match Torcetti's CausalSelfAttention
        s = h.size(1)
        mask = torch.triu(torch.full((s, s), float('-inf'), device=device), diagonal=1)
        h = self.transformer(h, mask=mask)
        h = self.ln_f(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _copy_linear(torcetti_linear, torch_linear):
    """Copy weights/bias from torcetti.nn.Linear -> torch.nn.Linear."""
    torch_linear.weight.data = torch.from_numpy(torcetti_linear.weight.data.T.astype(TORCH_TO_NUMPY_DTYPE[torch_linear.weight.dtype]))
    if torcetti_linear.bias is not None:
        torch_linear.bias.data = torch.from_numpy(torcetti_linear.bias.data.astype(TORCH_TO_NUMPY_DTYPE[torch_linear.bias.dtype]))


def _copy_embedding(torcetti_emb, torch_emb):
    torch_emb.weight.data = torch.from_numpy(torcetti_emb.weight.data.astype(TORCH_TO_NUMPY_DTYPE[torch_emb.weight.dtype]))


def _copy_layernorm(torcetti_ln, torch_ln):
    torch_ln.weight.data = torch.from_numpy(torcetti_ln.weight.data.astype(TORCH_TO_NUMPY_DTYPE[torch_ln.weight.dtype]))
    torch_ln.bias.data = torch.from_numpy(torcetti_ln.bias.data.astype(TORCH_TO_NUMPY_DTYPE[torch_ln.bias.dtype]))


def _copy_mha(torcetti_mha: MultiheadAttention, torch_mha: nn.MultiheadAttention):
    """Copy parameters from torcetti's MultiheadAttention -> torch."""
    q_w = torcetti_mha.q_proj.weight.data
    k_w = torcetti_mha.k_proj.weight.data
    v_w = torcetti_mha.v_proj.weight.data
    q_b = torcetti_mha.q_proj.bias.data if torcetti_mha.q_proj.bias is not None else None
    k_b = torcetti_mha.k_proj.bias.data if torcetti_mha.k_proj.bias is not None else None
    v_b = torcetti_mha.v_proj.bias.data if torcetti_mha.v_proj.bias is not None else None

    in_proj_weight = np.concatenate([q_w.T, k_w.T, v_w.T], axis=0)
    torch_mha.in_proj_weight.data = torch.from_numpy(in_proj_weight.astype(np.float32))

    if q_b is not None:
        in_proj_bias = np.concatenate([q_b, k_b, v_b], axis=0)
        torch_mha.in_proj_bias.data = torch.from_numpy(in_proj_bias.astype(np.float32))
    else:
        torch_mha.in_proj_bias = None  # type: ignore

    # Out proj
    _copy_linear(torcetti_mha.out_proj, torch_mha.out_proj)


def _copy_ffn(torcetti_ffn, torch_layer):
    _copy_linear(torcetti_ffn.linear1, torch_layer.linear1)
    _copy_linear(torcetti_ffn.linear2, torch_layer.linear2)


# NOTE: Copying the full transformer stack (especially attention) would require
# a lot of boilerplate. For the purpose of *unit* testing we only copy the easy
# layers (embeddings + final head). This is enough to make the very first forward
# bytes identical which is sufficient for a smoke-test style comparison.

def _partial_weight_copy(torcetti_gpt, torch_gpt):
    # Embeddings + final head (prefer new locations under transformer)
    if hasattr(torcetti_gpt, "transformer") and hasattr(torcetti_gpt.transformer, "wte"):
        _copy_embedding(torcetti_gpt.transformer.wte, torch_gpt.token_emb)
    elif hasattr(torcetti_gpt, "embed"):
        _copy_embedding(torcetti_gpt.embed, torch_gpt.token_emb)
    elif hasattr(torcetti_gpt, "token_emb"):
        _copy_embedding(torcetti_gpt.token_emb, torch_gpt.token_emb)

    if hasattr(torcetti_gpt, "transformer") and hasattr(torcetti_gpt.transformer, "wpe") and hasattr(torch_gpt, "pos_emb"):
        _copy_embedding(torcetti_gpt.transformer.wpe, torch_gpt.pos_emb)
    elif hasattr(torcetti_gpt, "pos_emb") and hasattr(torch_gpt, "pos_emb"):
        _copy_embedding(torcetti_gpt.pos_emb, torch_gpt.pos_emb)
    elif hasattr(torcetti_gpt, "pos_embed") and hasattr(torch_gpt, "pos_emb"):
        _copy_embedding(torcetti_gpt.pos_embed.pos_emb, torch_gpt.pos_emb)

    if hasattr(torcetti_gpt, "lm_head"):
        _copy_linear(torcetti_gpt.lm_head, torch_gpt.lm_head)

    # Transformer blocks (prefer modules under transformer.h)
    tor_layers = None
    if hasattr(torcetti_gpt, "transformer") and hasattr(torcetti_gpt.transformer, "h"):
        tor_layers = torcetti_gpt.transformer.h
    elif hasattr(torcetti_gpt, "layers"):
        tor_layers = torcetti_gpt.layers

    if tor_layers is not None:
        for tor_layer, torch_layer in zip(tor_layers, torch_gpt.transformer.layers):
            # LayerNorm 1 & 2
            _copy_layernorm(tor_layer.ln1, torch_layer.norm1)
            _copy_layernorm(tor_layer.ln2, torch_layer.norm2)
            # Attention
            _copy_mha(tor_layer.attn.attn, torch_layer.self_attn)
            # FeedForward
            _copy_ffn(tor_layer.ffn, torch_layer)


# ---------------------------------------------------------------------------
# Extended test-suite
# ---------------------------------------------------------------------------

class TestMiniGPTExtended:
    def _build_models(self, vocab_size=50, embed_dim=32, num_heads=4, num_layers=2, max_seq_len=8):
        """Utility to build a torcetti GPT + matching torch reference model."""
        # -- Torcetti
        torcetti_model = GPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        # -- Torch
        torch_model = TorchMiniGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        _partial_weight_copy(torcetti_model, torch_model)
        torcetti_model.eval(); torch_model.eval()
        return torcetti_model, torch_model

    def test_forward_backward_against_pytorch(self):
        """Compare a single forward/backward pass against a reference PyTorch implementation."""
        batch_size, seq_len, vocab_size = 2, 8, 50
        embed_dim, num_heads, num_layers = 32, 4, 2
        torcetti_gpt, torch_gpt = self._build_models(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=seq_len,
        )
        # Random input ids
        np.random.seed(0); torch.manual_seed(0)
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        compare_forward_backward(
            lambda x: torcetti_gpt(x),
            lambda x: torch_gpt(x.long()),
            [input_ids],
            requires_grad=[False],
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("seq_len", [1, 4, 8])
    def test_variable_sequence_lengths(self, seq_len):
        """Ensure model works for various sequence lengths up to max_seq_len."""
        batch_size = 2; vocab_size = 30; embed_dim = 16; num_heads = 4; num_layers = 1
        torcetti_gpt, _ = self._build_models(vocab_size, embed_dim, num_heads, num_layers, max_seq_len=8)
        inputs = Tensor(np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32), requires_grad=False)
        outputs = torcetti_gpt(inputs)
        assert outputs.shape == (batch_size, seq_len, vocab_size)

    def test_training_loss_decreases(self):
        """Run a few optimisation steps and assert that the loss decreases."""
        batch_size, seq_len, vocab_size = 4, 8, 60
        embed_dim, num_heads, num_layers = 32, 4, 2
        model = GPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=seq_len,
        )
        optimizer = AdamW(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        np.random.seed(42)
        inputs_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        targets_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
        inputs = Tensor(inputs_np, requires_grad=False)
        targets = Tensor(targets_np, requires_grad=False)

        losses = []
        for _ in range(5):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            losses.append(loss.data.item() if hasattr(loss.data, 'item') else loss.data)
            loss.backward(); optimizer.step(); optimizer.zero_grad()  # type: ignore[attr-defined]
        # Check that the loss has gone down at least a little
        assert losses[-1] < losses[0], f"Expected training loss to decrease, but got {losses}"

