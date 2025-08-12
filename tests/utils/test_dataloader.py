import unittest
import numpy as np

from torcetti.utils.data import Dataset, DataLoader
from torcetti.core.tensor import Tensor


class ToyDataset(Dataset):
    def __init__(self, n):
        self.data = np.arange(n, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Tensor(self.data[idx])


class TestDatasetBasic(unittest.TestCase):
    def test_len_and_getitem(self):
        ds = ToyDataset(10)
        self.assertEqual(len(ds), 10)
        self.assertAlmostEqual(ds[3].data, 3.0)


class TestDataLoaderSequential(unittest.TestCase):
    def test_batches_and_remainder(self):
        ds = ToyDataset(10)
        loader = DataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
        seen = []
        for batch in loader:
            self.assertIsInstance(batch, Tensor)
            seen.extend(batch.data.tolist())
        self.assertListEqual(seen, list(range(10)))

    def test_drop_last(self):
        ds = ToyDataset(10)
        loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (4,))


class TestDataLoaderShuffle(unittest.TestCase):
    def test_shuffle_deterministic(self):
        ds = ToyDataset(20)
        loader1 = DataLoader(ds, batch_size=5, shuffle=True, seed=123)
        order1 = np.concatenate([b.data for b in loader1]).tolist()
        loader2 = DataLoader(ds, batch_size=5, shuffle=True, seed=123)
        order2 = np.concatenate([b.data for b in loader2]).tolist()
        loader3 = DataLoader(ds, batch_size=5, shuffle=True, seed=42)
        order3 = np.concatenate([b.data for b in loader3]).tolist()
        self.assertListEqual(order1, order2)
        self.assertNotEqual(order1, order3)


if __name__ == "__main__":
    unittest.main()


