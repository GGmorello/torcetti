import unittest
from torcetti.core.tensor import Tensor


class TestDetachSemantics(unittest.TestCase):
    def test_detach_stops_gradient(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = a * 2
        c = b.detach()
        d = c * 3
        loss = d.sum(); loss.backward()
        self.assertIsNone(a.grad.data)
        self.assertFalse(c.requires_grad)

    def test_detach_in_place(self):
        a = Tensor([1.0], requires_grad=True)
        b = a * 2
        b.detach_()
        self.assertFalse(b.requires_grad)
        d = (b * 3).sum(); d.backward()
        self.assertIsNone(a.grad.data)


if __name__ == '__main__':
    unittest.main()


