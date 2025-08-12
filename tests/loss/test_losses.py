"""
Test loss functions against PyTorch reference implementations.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from torcetti.core.tensor import Tensor
from torcetti.loss.mse import MSE
from torcetti.loss.crossentropy import CrossEntropyLoss
from tests.test_helpers import compare_forward_backward


class TestLossFunctions(unittest.TestCase):
    def test_mse_loss(self):
        np.random.seed(42)
        predictions = np.random.randn(3, 4)
        targets = np.random.randn(3, 4)
        torcetti_loss = MSE(); torch_loss = nn.MSELoss(reduction='mean')
        compare_forward_backward(lambda pred, target: torcetti_loss(pred, target), lambda pred, target: torch_loss(pred, target), [predictions, targets], requires_grad=[True, False])

    def test_cross_entropy_loss(self):
        np.random.seed(42)
        logits = np.random.randn(2, 3)
        targets = np.array([0, 2])
        torcetti_loss = CrossEntropyLoss(); torch_loss = nn.CrossEntropyLoss()
        compare_forward_backward(lambda pred, target: torcetti_loss(pred, target), lambda pred, target: torch_loss(pred, target.to(dtype=torch.long)), [logits, targets], requires_grad=[True, False])

    def test_cross_entropy_numerical_stability(self):
        large_logits = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
        targets = np.array([0, 2])
        torcetti_loss = CrossEntropyLoss(); torch_loss = nn.CrossEntropyLoss()
        compare_forward_backward(lambda pred, target: torcetti_loss(pred, target), lambda pred, target: torch_loss(pred, target.to(dtype=torch.long)), [large_logits, targets], requires_grad=[True, False])

    def test_mse_properties(self):
        np.random.seed(42)
        predictions = np.random.randn(10)
        targets = np.random.randn(10)
        torcetti_loss = MSE()
        loss_value = torcetti_loss(Tensor(predictions), Tensor(targets))
        self.assertGreaterEqual(loss_value.data, 0)
        zero_loss = torcetti_loss(Tensor(targets), Tensor(targets))
        self.assertAlmostEqual(zero_loss.data, 0.0, places=6)

    def test_cross_entropy_properties(self):
        np.random.seed(42)
        logits = np.random.randn(5, 3)
        targets = np.array([0, 1, 2, 0, 1])
        torcetti_loss = CrossEntropyLoss()
        loss_value = torcetti_loss(Tensor(logits), Tensor(targets.astype(np.float32)))
        self.assertGreaterEqual(loss_value.data, 0)
        perfect_logits = np.zeros((5, 3))
        perfect_logits[np.arange(5), targets.astype(int)] = 10.0
        perfect_loss = torcetti_loss(Tensor(perfect_logits), Tensor(targets.astype(np.float32)))
        self.assertLess(perfect_loss.data, loss_value.data)

    def test_batch_consistency(self):
        np.random.seed(42)
        pred1, pred2 = np.random.randn(3), np.random.randn(3)
        target1, target2 = np.random.randn(3), np.random.randn(3)
        mse_loss = MSE()
        loss1 = mse_loss(Tensor(pred1), Tensor(target1))
        loss2 = mse_loss(Tensor(pred2), Tensor(target2))
        batch_pred = np.stack([pred1, pred2]); batch_target = np.stack([target1, target2])
        batch_loss = mse_loss(Tensor(batch_pred), Tensor(batch_target))
        expected_batch_loss = (loss1.data + loss2.data) / 2
        self.assertAlmostEqual(batch_loss.data, expected_batch_loss, places=6)


if __name__ == '__main__':
    unittest.main()


