import numpy as np
import pytest

# Skip module if core package isn't importable in the running environment
try:
    from torcetti.core.tensor import Tensor  # type: ignore
    from torcetti.loss.crossentropy import CrossEntropyLoss  # type: ignore
    from torcetti.optim.adamw import AdamW  # type: ignore
except ModuleNotFoundError:
    pytest.skip("torcetti package not importable; skipping trainer tests", allow_module_level=True)

# GPT may import torch in its module; skip if torch is not present in the env
pytest.importorskip("torch", reason="torch is required for GPT module import in this repo")

from examples.GPT.gpt_model import GPT
from examples.GPT.trainer import Trainer


class ToyLanguageDataset:
    def __init__(self, vocab_size: int, seq_len: int, size: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.inputs = rng.randint(0, vocab_size, size=(size, seq_len)).astype(np.int32)
        self.targets = rng.randint(0, vocab_size, size=(size, seq_len)).astype(np.int32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return Tensor(self.inputs[idx], requires_grad=False), Tensor(self.targets[idx], requires_grad=False)


def _build_setup(vocab_size=40, embed_dim=16, num_heads=4, num_layers=2, seq_len=8):
    model = GPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, max_seq_len=seq_len)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    return model, optimizer, criterion


def test_trainer_class_interface():
    model, optimizer, criterion = _build_setup()
    trainer = Trainer(model, optimizer, criterion)
    # Ensure required methods exist
    assert hasattr(trainer, "fit")
    assert hasattr(trainer, "train_batch")
    assert hasattr(trainer, "evaluate")


def test_training_reduces_loss_smoke():
    vocab_size, seq_len = 50, 8
    train_ds = ToyLanguageDataset(vocab_size, seq_len, size=32, seed=42)
    valid_ds = ToyLanguageDataset(vocab_size, seq_len, size=16, seed=7)

    model, optimizer, criterion = _build_setup(vocab_size=vocab_size, seq_len=seq_len)
    trainer = Trainer(model, optimizer, criterion, log_interval=10)

    # Use a very small manual loader to keep this test simple and fast
    class Loader:
        def __init__(self, dataset, batch_size):
            self.dataset = dataset
            self.batch_size = batch_size

        def __iter__(self):
            for i in range(0, len(self.dataset), self.batch_size):
                batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
                xs = Tensor(np.stack([x.data for x, _ in batch], axis=0), requires_grad=False)
                ys = Tensor(np.stack([y.data for _, y in batch], axis=0), requires_grad=False)
                yield xs, ys

    train_loader = Loader(train_ds, batch_size=8)
    valid_loader = Loader(valid_ds, batch_size=8)

    history = trainer.fit(train_loader, valid_loader, epochs=2)
    assert len(history["train_loss"]) == 2
    # Expect the training loss to go down across epochs (smoke check)
    assert history["train_loss"][-1] <= history["train_loss"][0]


