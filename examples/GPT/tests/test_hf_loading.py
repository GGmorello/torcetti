import numpy as np
import pytest
import torch

from torcetti.core.tensor import Tensor


transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel


def _build_matching_models(vocab_size=64, embed_dim=32, num_heads=4, num_layers=2, max_seq_len=16):
    """Utility: build an HF GPT-2 and an HFGPT (to be implemented by user) with matching hyperparams."""
    # HF model from config (offline, no weights download)
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_seq_len,
        n_embd=embed_dim,
        n_head=num_heads,
        n_layer=num_layers,
        n_inner=4 * embed_dim,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    hf_model = GPT2LMHeadModel(hf_config)
    hf_model.eval()

    # Import user-provided class that should extend examples.GPT.gpt_model.GPT
    from examples.GPT.hf_compat import HFGPT  # type: ignore

    tor_model = HFGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    tor_model.eval()

    return tor_model, hf_model


@pytest.mark.parametrize("batch_size,seq_len", [(1, 4), (2, 8)])
def test_can_load_from_hf_gpt2_and_match_forward(batch_size, seq_len):
    """
    Expectation/specification:
    - There exists a class examples.GPT.hf_compat.HFGPT that extends GPT
    - It implements a method `.load_from_hf_gpt2(hf_model)` which loads weights from a transformers.GPT2LMHeadModel
    - After loading, a forward pass on the same input matches HF logits (within tolerance)
    """
    vocab_size = 127
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    max_seq_len = 32

    # Build models with matching shapes
    tor_model, hf_model = _build_matching_models(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    # The user class must expose a weight-loading method that accepts a HF GPT-2 model
    assert hasattr(tor_model, "load_from_hf_gpt2"), "HFGPT must implement load_from_hf_gpt2(hf_model)"
    tor_model.load_from_hf_gpt2(hf_model)

    # Prepare identical random inputs
    rng = np.random.default_rng(0)
    input_ids_np = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # HF forward
    with torch.no_grad():
        hf_logits = hf_model(torch.from_numpy(input_ids_np.astype(np.int64))).logits.detach().cpu().numpy()

    # Torcetti forward
    tor_logits = tor_model(Tensor(input_ids_np, requires_grad=False)).data

    # Compare
    np.testing.assert_allclose(tor_logits, hf_logits, rtol=1e-4, atol=1e-4)


def test_partial_parameter_mapping_shapes():
    """
    Additionally specify that key parameter groups map 1:1 in shape:
    - token embeddings (wte), position embeddings (wpe)
    - final ln_f and lm_head
    - per-block: ln1/ln2, q/k/v/out projections, and MLP fc/proj
    """
    vocab_size = 101
    embed_dim = 48
    num_heads = 6
    num_layers = 1
    max_seq_len = 16

    tor_model, hf_model = _build_matching_models(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    tor_model.load_from_hf_gpt2(hf_model)

    # Embeddings
    assert tor_model.transformer.wte.weight.shape == hf_model.transformer.wte.weight.detach().cpu().numpy().shape
    assert tor_model.transformer.wpe.weight.shape == hf_model.transformer.wpe.weight.detach().cpu().numpy().shape

    # Final norm and head
    assert tor_model.transformer.ln_f.weight.shape == hf_model.transformer.ln_f.weight.detach().cpu().numpy().shape
    assert tor_model.transformer.ln_f.bias.shape == hf_model.transformer.ln_f.bias.detach().cpu().numpy().shape

    # lm_head differs by transpose convention; ensure total params match
    tor_w = tor_model.lm_head.weight.shape  # (embed_dim, vocab)
    hf_w = tuple(hf_model.lm_head.weight.detach().cpu().numpy().shape)  # (vocab, embed_dim)
    assert tor_w == (hf_w[1], hf_w[0])

    # One block checks
    block = tor_model.transformer.h[0]
    hf_block = hf_model.transformer.h[0]

    # LN
    assert block.ln1.weight.shape == hf_block.ln_1.weight.detach().cpu().numpy().shape
    assert block.ln2.weight.shape == hf_block.ln_2.weight.detach().cpu().numpy().shape

    # Attention qkv/out
    assert block.attn.attn.q_proj.weight.shape == (embed_dim, embed_dim)
    assert block.attn.attn.k_proj.weight.shape == (embed_dim, embed_dim)
    assert block.attn.attn.v_proj.weight.shape == (embed_dim, embed_dim)
    assert block.attn.attn.out_proj.weight.shape == (embed_dim, embed_dim)

    # MLP
    assert block.ffn.linear1.weight.shape == (embed_dim, 4 * embed_dim)
    assert block.ffn.linear2.weight.shape == (4 * embed_dim, embed_dim)


