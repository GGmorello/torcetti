"""
Test loss functions against PyTorch reference implementations.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torcetti.core.tensor import Tensor
from torcetti.loss.mse import MSE
from torcetti.loss.crossentropy import CrossEntropyLoss
from tests.test_helpers import assert_tensors_close, compare_forward_backward


class TestLossFunctions(unittest.TestCase):
    """Test loss functions against PyTorch."""

    def test_mse_loss(self):
        """Test MSE loss function."""
        np.random.seed(42)
        predictions = np.random.randn(3, 4)
        targets = np.random.randn(3, 4)
        
        # Test MSE loss (torcetti doesn't have reduction parameter)
        torcetti_loss = MSE()
        torch_loss = nn.MSELoss(reduction='mean')
        
        def torcetti_fn(pred, target):
            return torcetti_loss(pred, target)
        
        def torch_fn(pred, target):
            return torch_loss(pred, target)
        
        compare_forward_backward(torcetti_fn, torch_fn, [predictions, targets], requires_grad=[True, False])

    def test_cross_entropy_loss(self):
        """Test CrossEntropy loss function."""
        np.random.seed(42)
        logits = np.random.randn(2, 3)  # 2 samples, 3 classes
        targets = np.array([0, 2])  # Target classes
        
        torcetti_loss = CrossEntropyLoss()
        torch_loss = nn.CrossEntropyLoss()
        
        def torcetti_fn(pred, target):
            return torcetti_loss(pred, target)
        
        def torch_fn(pred, target):
            return torch_loss(pred, target.to(dtype=torch.long))
        
        compare_forward_backward(torcetti_fn, torch_fn, [logits, targets], requires_grad=[True, False])

    def test_cross_entropy_numerical_stability(self):
        """Test CrossEntropy with large logits for numerical stability."""
        # Large logits that could cause overflow
        large_logits = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
        targets = np.array([0, 2])
        
        torcetti_loss = CrossEntropyLoss()
        torch_loss = nn.CrossEntropyLoss()
        
        def torcetti_fn(pred, target):
            return torcetti_loss(pred, target)
        
        def torch_fn(pred, target):
            return torch_loss(pred, target.to(dtype=torch.long))
        
        compare_forward_backward(torcetti_fn, torch_fn, [large_logits, targets], requires_grad=[True, False])

    def test_mse_properties(self):
        """Test mathematical properties of MSE loss."""
        np.random.seed(42)
        predictions = np.random.randn(10)
        targets = np.random.randn(10)
        
        torcetti_loss = MSE()
        loss_value = torcetti_loss(Tensor(predictions), Tensor(targets))
        
        # MSE should be non-negative
        self.assertGreaterEqual(loss_value.data, 0)
        
        # MSE should be zero when predictions equal targets
        zero_loss = torcetti_loss(Tensor(targets), Tensor(targets))
        self.assertAlmostEqual(zero_loss.data, 0.0, places=6)

    def test_cross_entropy_properties(self):
        """Test mathematical properties of CrossEntropy loss."""
        np.random.seed(42)
        logits = np.random.randn(5, 3)  # 5 samples, 3 classes
        targets = np.array([0, 1, 2, 0, 1])
        
        torcetti_loss = CrossEntropyLoss()
        loss_value = torcetti_loss(Tensor(logits), Tensor(targets.astype(np.float32)))
        
        # CrossEntropy should be non-negative
        self.assertGreaterEqual(loss_value.data, 0)
        
        # Loss should be higher when predictions are wrong
        # Create "perfect" predictions (high confidence for correct class)
        perfect_logits = np.zeros((5, 3))
        perfect_logits[np.arange(5), targets.astype(int)] = 10.0  # High confidence
        
        perfect_loss = torcetti_loss(Tensor(perfect_logits), Tensor(targets.astype(np.float32)))
        self.assertLess(perfect_loss.data, loss_value.data)

    def test_batch_consistency(self):
        """Test that batch processing gives same results as individual samples."""
        np.random.seed(42)
        
        # Test MSE
        pred1, pred2 = np.random.randn(3), np.random.randn(3)
        target1, target2 = np.random.randn(3), np.random.randn(3)
        
        mse_loss = MSE()
        
        # Individual losses
        loss1 = mse_loss(Tensor(pred1), Tensor(target1))
        loss2 = mse_loss(Tensor(pred2), Tensor(target2))
        
        # Batch loss
        batch_pred = np.stack([pred1, pred2])
        batch_target = np.stack([target1, target2])
        batch_loss = mse_loss(Tensor(batch_pred), Tensor(batch_target))
        
        # Should be average of individual losses
        expected_batch_loss = (loss1.data + loss2.data) / 2
        self.assertAlmostEqual(batch_loss.data, expected_batch_loss, places=6)


if __name__ == '__main__':
    unittest.main()
