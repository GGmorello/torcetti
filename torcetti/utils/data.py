import numpy as np
import math

from torcetti import Tensor
from torcetti import stack


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

class DataLoader:
    def __init__(self, dataset, batch_size=1,
                shuffle=False, drop_last=False, seed=None, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.collate_fn = collate_fn or self._default_collate
        
        self.indices=np.arange(len(dataset))
        self.rng = None
        if shuffle:
            self.rng = np.random.RandomState(self.seed) if self.seed is not None else np.random
        self.ptr = 0

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.ptr = 0
        return self
    
    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration

        start = self.ptr
        end   = self.ptr + self.batch_size
        batch_idx = self.indices[start:end]
        if len(batch_idx) < self.batch_size and self.drop_last:
            raise StopIteration

        samples = [self.dataset[i] for i in batch_idx]
        self.ptr = end

        if self.collate_fn:
            return self.collate_fn(samples)
        else:
            from torcetti.core.ops import stack
            return stack(samples, axis=0) 

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    @staticmethod
    def _default_collate(samples):
        if not samples:
            raise ValueError("Received empty batch in collate_fn")

        first = samples[0]

        if isinstance(first, Tensor):
            return stack(samples, axis=0)

        if isinstance(first, np.ndarray):
            data = np.stack(samples, axis=0)
            return Tensor(data)

        if np.isscalar(first):
            data = np.array(samples, dtype=np.float32)
            return Tensor(data)

        if isinstance(first, dict):
            return {k: DataLoader._default_collate([s[k] for s in samples]) for k in first}

        if isinstance(first, (list, tuple)):
            transposed = list(zip(*samples))
            collated = [DataLoader._default_collate(list(items)) for items in transposed]
            return type(first)(collated)

        raise TypeError(f"Unsupported sample type for collation: {type(first)}")