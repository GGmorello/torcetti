"""Test optimizer state persistence across all optimizers."""

import numpy as np
import unittest
from torcetti.core.tensor import Tensor
from torcetti.core.parameter import Parameter
from torcetti.optim.sgd import SGD
from torcetti.optim.adam import Adam
from torcetti.optim.adamw import AdamW
from torcetti.optim.rmsprop import RMSprop


class TestOptimizerStatePersistence(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.param1 = Parameter(np.random.randn(3, 3).astype(np.float32))
        self.param2 = Parameter(np.random.randn(2, 4).astype(np.float32))
        self.params = [self.param1, self.param2]
        self.param1.grad = Tensor(np.random.randn(*self.param1.shape).astype(np.float32))
        self.param2.grad = Tensor(np.random.randn(*self.param2.shape).astype(np.float32))

    def test_sgd_state_persistence(self):
        optimizer = SGD(self.params, lr=0.01, momentum=0.9)
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(self.param2, optimizer.state)
        momentum_buffer1_step1 = optimizer.state[self.param1]['momentum_buffer'].copy()
        momentum_buffer2_step1 = optimizer.state[self.param2]['momentum_buffer'].copy()
        self.param1.grad = Tensor(np.random.randn(*self.param1.shape).astype(np.float32))
        self.param2.grad = Tensor(np.random.randn(*self.param2.shape).astype(np.float32))
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(self.param2, optimizer.state)
        momentum_buffer1_step2 = optimizer.state[self.param1]['momentum_buffer']
        momentum_buffer2_step2 = optimizer.state[self.param2]['momentum_buffer']
        self.assertFalse(np.allclose(momentum_buffer1_step1, momentum_buffer1_step2))
        self.assertFalse(np.allclose(momentum_buffer2_step1, momentum_buffer2_step2))

    def test_adam_state_persistence(self):
        optimizer = Adam(self.params, lr=0.001)
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(self.param2, optimizer.state)
        self.assertIn('step', optimizer.state[self.param1])
        self.assertIn('exp_avg', optimizer.state[self.param1])
        self.assertIn('exp_avg_sq', optimizer.state[self.param1])
        step1_count = optimizer.state[self.param1]['step']
        exp_avg1_step1 = optimizer.state[self.param1]['exp_avg'].copy()
        self.param1.grad = Tensor(np.random.randn(*self.param1.shape).astype(np.float32))
        self.param2.grad = Tensor(np.random.randn(*self.param2.shape).astype(np.float32))
        optimizer.step()
        self.assertEqual(optimizer.state[self.param1]['step'], step1_count + 1)
        exp_avg1_step2 = optimizer.state[self.param1]['exp_avg']
        self.assertFalse(np.allclose(exp_avg1_step1, exp_avg1_step2))

    def test_adamw_state_persistence(self):
        optimizer = AdamW(self.params, lr=0.001, weight_decay=0.01)
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(self.param2, optimizer.state)
        self.assertIn('step', optimizer.state[self.param1])
        self.assertIn('exp_avg', optimizer.state[self.param1])
        self.assertIn('exp_avg_sq', optimizer.state[self.param1])
        step1_count = optimizer.state[self.param1]['step']
        exp_avg1_step1 = optimizer.state[self.param1]['exp_avg'].copy()
        self.param1.grad = Tensor(np.random.randn(*self.param1.shape).astype(np.float32))
        self.param2.grad = Tensor(np.random.randn(*self.param2.shape).astype(np.float32))
        optimizer.step()
        self.assertEqual(optimizer.state[self.param1]['step'], step1_count + 1)
        exp_avg1_step2 = optimizer.state[self.param1]['exp_avg']
        self.assertFalse(np.allclose(exp_avg1_step1, exp_avg1_step2))

    def test_rmsprop_state_persistence(self):
        optimizer = RMSprop(self.params, lr=0.001, momentum=0.9)
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(self.param2, optimizer.state)
        self.assertIn('buffer', optimizer.state[self.param1])
        self.assertIn('square_avg', optimizer.state[self.param1])
        buffer1_step1 = optimizer.state[self.param1]['buffer'].copy()
        square_avg1_step1 = optimizer.state[self.param1]['square_avg'].copy()
        self.param1.grad = Tensor(np.random.randn(*self.param1.shape).astype(np.float32))
        self.param2.grad = Tensor(np.random.randn(*self.param2.shape).astype(np.float32))
        optimizer.step()
        buffer1_step2 = optimizer.state[self.param1]['buffer']
        square_avg1_step2 = optimizer.state[self.param1]['square_avg']
        self.assertFalse(np.allclose(buffer1_step1, buffer1_step2))
        self.assertFalse(np.allclose(square_avg1_step1, square_avg1_step2))

    def test_all_optimizers_use_param_object_as_key(self):
        optimizers = [
            SGD(self.params, lr=0.01, momentum=0.9),
            Adam(self.params, lr=0.001),
            AdamW(self.params, lr=0.001),
            RMSprop(self.params, lr=0.001, momentum=0.9)
        ]
        for optimizer in optimizers:
            optimizer.step()
            self.assertIn(self.param1, optimizer.state)
            self.assertIn(self.param2, optimizer.state)
            self.assertNotIn(id(self.param1), optimizer.state)
            self.assertNotIn(id(self.param2), optimizer.state)

    def test_state_isolation_between_parameters(self):
        optimizer = Adam(self.params, lr=0.001)
        optimizer.step()
        self.assertIsNot(optimizer.state[self.param1], optimizer.state[self.param2])
        optimizer.state[self.param1]['step'] = 999
        self.assertNotEqual(optimizer.state[self.param2]['step'], 999)

    def test_parameter_replacement_creates_new_state(self):
        optimizer = Adam([self.param1], lr=0.001)
        optimizer.step()
        original_state = optimizer.state[self.param1].copy()
        new_param = Parameter(np.random.randn(*self.param1.shape).astype(np.float32))
        new_param.grad = Tensor(np.random.randn(*new_param.shape).astype(np.float32))
        optimizer.param_groups[0]['params'] = [new_param]
        optimizer.step()
        self.assertIn(self.param1, optimizer.state)
        self.assertIn(new_param, optimizer.state)
        self.assertIsNot(optimizer.state[self.param1], optimizer.state[new_param])


if __name__ == '__main__':
    unittest.main()


