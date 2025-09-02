## Torcetti

An educational, NumPy-based re-implementation of a subset of PyTorch. Torcetti focuses on clarity and testability to help you learn how tensors, automatic differentiation, neural network modules, and training loops work under the hood.

### Project goals

- Provide a compact, readable reference implementation of core deep learning primitives.
- Emphasize correctness and pedagogy over raw performance or feature breadth.
- Encourage test-driven learning; the included tests define the expected behavior.
- Offer practical examples for building and training small models, including attention mechanisms.

### What is implemented

- Core tensor and autograd
  - `torcetti.core.tensor.Tensor` with automatic differentiation, lazy gradient allocation, and topological backward pass
  - Broadcasting-aware ops with gradient unbroadcasting
  - Type promotion rules for mixed dtypes
  - Views and shape ops: `reshape`/`view`, `permute`, `flatten`, `repeat`, `expand`, `unsqueeze`, `squeeze`, transpose `T`
  - Indexing and slicing with gradient scatter; `take` (flattened), advanced indexing tests
  - Elementwise and reduction ops: `sum`, `mean`, `max`, `min`, `prod`, `var(ddof=)`, `argmax`, `argmin`, `abs`, `clamp`, `exp`, `log`, `sqrt`, `sin`, `cos`, `floor`, `ceil`, `round`
  - Matrix ops: `@`/`matmul`/`dot`
  - Utilities: `where`, `topk`, `cat`, `stack`, `meshgrid`, `multinomial`
  - Gradient mode: `no_grad()`, `grad_enabled()`; tensor methods: `backward`, `detach`, `detach_`, `zero_grad`
  - Dtype utilities: `get_default_dtype`, `set_default_dtype`

- Factory functions (also re-exported at package top-level)
  - `tensor`, `as_tensor`, `zeros`, `zeros_like`, `ones`, `ones_like`, `full`, `full_like`, `empty`, `empty_like`
  - Random: `rand`, `rand_like`, `randn`, `randn_like`, `randint`, `normal`
  - Ranges/linear algebra helpers: `arange`, `linspace`, `eye`, `diag`

- Neural network building blocks (`torcetti.nn`)
  - Base: `Module`, parameter registration, `train/eval`, `zero_grad`, `state_dict`/`load_state_dict`, containers (`Sequential`, `ModuleList`, `ModuleDict`)
  - Layers: `Linear`, `Embedding`, `Conv2d`, `BatchNorm1d`, `LayerNorm`, `Dropout`, `MaxPool2d`, `AvgPool2d`
  - Activations: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`
  - Attention: `MultiheadAttention`; functional `scaled_dot_product_attention`, `multi_head_attention`
  - Transformer components:
    - QK normalization (`l2`, `rms`)
    - Prefill/decode disaggregation
    - Multi-Query Attention (MQA)
    - Grouped-Query Attention (GQA)
    - Rotary Positional Embeddings (RoPE) via `build_rope_cache`, `apply_rotary_pos_emb`
    - Key/Value cache (KV cache)
    - Windowed attention

- Losses
  - Mean Squared Error: `MSE`
  - Cross Entropy: `CrossEntropyLoss` (via `log_softmax` + NLL)

- Optimizers (`torcetti.optim`)
  - `SGD` (momentum, dampening, Nesterov, weight decay)
  - `Adam` (bias correction, AMSGrad)
  - `AdamW` (decoupled weight decay)
  - `RMSprop` (centered, momentum)

- Data utilities
  - `Dataset` protocol and `DataLoader` with batching, optional shuffling, `drop_last`, and a default collate that handles tensors, arrays, scalars, dicts, lists/tuples

- Examples
  - Minimal GPT components under `examples/GPT/` (attention, transformer blocks, trainer)
  - Simple training scripts (e.g., linear regression)
  - Hugging Face GPT‑2 compatibility via `examples/GPT/hf_compat.py` (load pretrained weights)

### GPT example

The `examples/GPT/` folder contains a compact, readable GPT implementation adapted from minGPT and aligned with Torcetti primitives.

- Components
  - Tokenization: `bpe.py` provides a GPT‑2 BPE tokenizer that downloads and caches the 50k merges/vocab
  - Model: `gpt_model.py` defines `GPT` using `Embedding`, `TransformerBlock`, pre‑norm `LayerNorm`, and a tied `lm_head`
  - Attention: `causal_self_attention.py` implements causal masking and a `KVCache` for efficient decode-time generation
  - Training: `trainer.py` builds `DataLoader`s from raw text, trains with `AdamW` and `CrossEntropyLoss`, reports loss/perplexity
  - Pretrained weights: `hf_compat.py` enables loading GPT‑2 weights from Hugging Face into Torcetti

- Train on a small dataset

```bash
python -m torcetti.examples.GPT.trainer --dataset tiny_shakespeare --block_size 128 --batch_size 16 --epochs 2
```

- Generate text with the included script

```bash
python -m torcetti.examples.GPT.generate
```

- Load Hugging Face GPT‑2 weights into Torcetti (optional)

```python
from torcetti.examples.GPT.hf_compat import HFGPT
model = HFGPT.from_pretrained("gpt2")
model.eval()
```

Notes
- Generation uses top‑k sampling and `multinomial`; switch to greedy by setting `do_sample=False` in `examples/GPT/generate.py`.
- Trainer reshapes logits/targets to `(B*L, V)` for `CrossEntropyLoss`.

### What you can do

- Build and train small MLPs and CNNs using `Sequential`, `Linear`, `Conv2d`, activations, pooling, normalization layers.
- Train models with `SGD`, `Adam`, `AdamW`, `RMSprop`; save and load weights via `state_dict`/`load_state_dict`.
- Experiment with attention: `MultiheadAttention` including grouped (GQA) and multi-query (MQA) variants; apply rotary embeddings.
- Prototype custom ops and layers while observing gradient flow and debugging with the provided tests.
- Use a simple `DataLoader` to batch numpy data and feed `Tensor`s into models.

### Installation

Requirements: Python 3.12+

Using pip:

```bash
pip install -e .
```

Using uv (optional):

```bash
uv sync
# for dev tooling (pytest), if desired
uv sync --group dev
```

### Quickstart

Train a small regression model end-to-end.

```python
import numpy as np
import torcetti as tc
from torcetti.loss.mse import MSE
from torcetti.optim.adam import Adam

# Synthetic data
X = np.random.randn(256, 1).astype(np.float32)
y = (2.0 * X + 0.5 + 0.1 * np.random.randn(256, 1)).astype(np.float32)

model = tc.nn.Sequential(
    tc.nn.Linear(1, 32),
    tc.nn.ReLU(),
    tc.nn.Linear(32, 1),
)

criterion = MSE()
optimizer = Adam(model.parameters(), lr=1e-2)

for epoch in range(200):
    inputs = tc.tensor(X)
    targets = tc.tensor(y)

    preds = model(inputs)
    loss = criterion(preds, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", float(loss.data))
```

Evaluate without tracking gradients:

```python
from torcetti.core.grad_mode import no_grad

model.eval()
with no_grad():
    preds = model(tc.tensor(X))
```

### Running tests

```bash
pip install -e . && pip install pytest
pytest -q
```

### Design principles

- Readable, explicit code paths over micro-optimizations.
- High-verbosity implementation to make data flow and gradients easy to follow.
- Tests drive behavior and serve as executable documentation.

### Limitations

- CPU-only, NumPy-backed; not optimized for speed or large-scale training.
- Not a drop-in replacement for PyTorch; API compatibility is partial and intentionally simplified.
- Numeric stability and edge-case behavior may differ from PyTorch.
- Public APIs may evolve as the project grows; consult tests for the authoritative specification.




