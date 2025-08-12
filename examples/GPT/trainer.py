import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import torcetti
from torcetti import Tensor, no_grad
from torcetti.loss.crossentropy import CrossEntropyLoss
from torcetti.optim.adamw import AdamW
from torcetti.utils.data import Dataset, DataLoader

from examples.GPT.gpt_model import GPT
from examples.GPT.bpe import BPETokenizer


class Trainer:
    def __init__(self, model: GPT, optimizer: AdamW, criterion: CrossEntropyLoss, log_interval: int = 50):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_interval = log_interval

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (B, L, V), targets: (B, L) -> reshape to (B*L, V) and (B*L)
        B, L, V = logits.shape
        logits_2d = logits.reshape(B * L, V)
        targets_1d = targets.reshape(B * L)
        loss = self.criterion(logits_2d, targets_1d)
        return loss

    def train_batch(self, batch) -> float:
        self.model.train()
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self._compute_loss(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.data)

    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with no_grad():
            for batch in data_loader:
                inputs, targets = batch
                logits = self.model(inputs)
                loss = self._compute_loss(logits, targets)
                total_loss += float(loss.data)
                count += 1
        return total_loss / max(count, 1)

    def fit(self, train_loader: DataLoader, valid_loader: Optional[DataLoader] = None, epochs: int = 1) -> Dict[str, list]:
        history = {"train_loss": [], "valid_loss": []}
        step = 0
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            print(f"batches: {len(train_loader)}")
            epoch_loss = 0.0
            for batch in train_loader:
                batch_loss = self.train_batch(batch)
                epoch_loss += batch_loss
                step += 1
                if self.log_interval and step % self.log_interval == 0:
                    ppl = float(np.exp(batch_loss)) if batch_loss < 50 else float('inf')
                    print(f"step {step}: loss={batch_loss:.4f} | ppl={ppl:.2f}")
            epoch_loss /= max(len(train_loader), 1)
            history["train_loss"].append(epoch_loss)

            if valid_loader is not None:
                val_loss = self.evaluate(valid_loader)
                history["valid_loss"].append(val_loss)
                print(f"epoch {epoch+1}: train_loss={epoch_loss:.4f} | valid_loss={val_loss:.4f} | train_ppl={np.exp(epoch_loss):.2f}")
            else:
                print(f"epoch {epoch+1}: train_loss={epoch_loss:.4f} | train_ppl={np.exp(epoch_loss):.2f}")

        return history


class TextDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, block_size: int):
        assert token_ids.ndim == 1
        self.ids = token_ids.astype(np.int32)
        self.block_size = int(block_size)
        self.num_items = max(0, len(self.ids) - self.block_size)

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.block_size]
        y = self.ids[idx + 1 : idx + 1 + self.block_size]
        return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)


def build_dataloader_from_text(text: str, tokenizer: BPETokenizer, block_size: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    encoded = tokenizer(text)  # Tensor shape (1, T)
    token_ids = encoded.data.reshape(-1)  # 1D numpy array
    print(f"Loaded text → {len(token_ids)} BPE tokens")
    dataset = TextDataset(token_ids, block_size=block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    print(f"block_size={block_size}, batch_size={batch_size}, steps_per_epoch={len(loader)}")
    return loader


def load_online_dataset(name: str) -> str:
    """Download and cache a tiny text dataset by name; return text contents."""
    name = (name or "").strip().lower()
    cache_dir = Path(os.path.expanduser("~/.cache/torcetti_datasets"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if name in {"tiny_shakespeare", "tiny-shakespeare", "shakespeare"}:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        dst = cache_dir / "tiny_shakespeare.txt"
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: tiny_shakespeare")

    if not dst.exists() or dst.stat().st_size == 0:
        import requests
        print(f"Downloading {name} from {url} → {dst}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dst.write_bytes(resp.content)

    return dst.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train GPT on text using BPETokenizer")
    parser.add_argument("--text_file", type=str, default=None, help="Path to a UTF-8 text file")
    parser.add_argument("--text", type=str, default=None, help="Inline text to train on (overrides --text_file if set)")
    parser.add_argument("--dataset", type=str, default=None, help="Name of a small online dataset to download (e.g., 'tiny_shakespeare')")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--save", type=str, default=None, help="Path to save model state_dict (npz)")
    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.dataset is not None:
        text = load_online_dataset(args.dataset)
    else:
        text = (
            "In the beginning the Universe was created. This has made a lot of people very angry "
            "and been widely regarded as a bad move."
        )

    tokenizer = BPETokenizer()

    vocab_size = 50257  # GPT-2 BPE vocab size used by BPETokenizer
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )

    # Log model info
    total_params = sum(int(np.prod(p.data.shape)) for p in model.parameters())
    print("Model config:")
    print(f"- vocab_size={vocab_size} | embed_dim={args.embed_dim} | heads={args.num_heads} | layers={args.num_layers} | max_seq_len={args.block_size}")
    print(f"- parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, log_interval=50)

    train_loader = build_dataloader_from_text(text, tokenizer, block_size=args.block_size, batch_size=args.batch_size, shuffle=True)

    history = trainer.fit(train_loader, valid_loader=None, epochs=args.epochs)
    print({k: [float(v) for v in vals] for k, vals in history.items()})

    if args.save:
        state = model.state_dict()
        # Save as a simple npz of tensors' data arrays
        np.savez(args.save, **{k: v.data for k, v in state.items()})
        print(f"Saved model state_dict (numpy arrays) to {args.save}")


if __name__ == "__main__":
    main()


