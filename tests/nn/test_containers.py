import unittest
import numpy as np

from torcetti.core.tensor import Tensor
from torcetti.nn import Module, Linear, ReLU, ModuleList, ModuleDict, Sequential


class TestModuleList(unittest.TestCase):
    def test_basic_list_semantics(self):
        ml = ModuleList([Linear(3, 4), ReLU(), Linear(4, 2)])

        self.assertEqual(len(ml), 3)
        self.assertIsInstance(ml[0], Linear)
        self.assertIsInstance(ml[1], ReLU)
        self.assertIsInstance(ml[2], Linear)

        types = [type(m).__name__ for m in ml]
        self.assertEqual(types, ["Linear", "ReLU", "Linear"])

    def test_parameters_traversal_in_parent_module(self):
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleList([Linear(3, 5), ReLU(), Linear(5, 2)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Wrapper()
        params = list(model.parameters())
        # Two Linear layers -> 4 parameters total (2 weights + 2 biases)
        self.assertEqual(len(params), 4)

        x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        y = model(x)
        self.assertEqual(y.shape, (2, 2))
        y.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_state_dict_keys_and_roundtrip(self):
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleList([Linear(3, 5), ReLU(), Linear(5, 2)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        np.random.seed(0)
        w1 = Wrapper(); w2 = Wrapper()

        state = w1.state_dict()
        expected_prefixes = {"layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias"}
        self.assertTrue(expected_prefixes.issubset(set(state.keys())))

        # Zero out target params then load and compare
        def resolve_prefix(root, key_prefix):
            obj = root
            for part in key_prefix:
                if part.isdigit():
                    obj = obj[int(part)]  # ModuleList supports indexing
                else:
                    obj = getattr(obj, part)
            return obj

        for k, t in state.items():
            parts = k.split(".")
            prefix_parts = parts[:-1]
            leaf = parts[-1]
            target_obj = resolve_prefix(w2, prefix_parts)
            if leaf == "weight":
                target_obj.weight.data[...] = 0
            elif leaf == "bias":
                target_obj.bias.data[...] = 0

        # Should accept ModuleList indexing with keys like 'layers.0.weight'
        w2.load_state_dict(state)

        s2 = w2.state_dict()
        for k in expected_prefixes:
            np.testing.assert_allclose(state[k].data, s2[k].data)

    def test_plain_python_list_is_not_traversed_like_pytorch(self):
        # Expect: a raw Python list attribute is NOT traversed by state_dict (PyTorch behavior)
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = [Linear(3, 5), ReLU(), Linear(5, 2)]  # raw list, not ModuleList/Sequential

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        w = Wrapper()
        st = w.state_dict()
        keys = set(st.keys())
        # No keys should be produced under the raw list name
        self.assertFalse(any(k.startswith("layers.") for k in keys))

    def test_dynamic_modification_append_extend(self):
        # Expect: append/extend change length and parameter traversal updates accordingly
        ml = ModuleList()
        self.assertEqual(len(ml), 0)
        ml.append(Linear(4, 4))
        self.assertEqual(len(ml), 1)
        ml.extend([ReLU(), Linear(4, 2)])
        self.assertEqual(len(ml), 3)

        class Wrapper(Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Wrapper(ml)
        # Two Linear layers -> 4 parameters
        self.assertEqual(len(list(model.parameters())), 4)


class TestSequentialContainers(unittest.TestCase):
    def test_nested_module_keys_and_loading(self):
        np.random.seed(1)
        net_src = Sequential(Linear(2, 2), Linear(2, 2))
        state = net_src.state_dict()
        expected_keys = {'0.weight', '0.bias', '1.weight', '1.bias'}
        self.assertEqual(set(state.keys()), expected_keys)
        net_tgt = Sequential(Linear(2, 2), Linear(2, 2))
        net_tgt.load_state_dict(state)
        for src_layer, tgt_layer in zip(net_src.layers, net_tgt.layers):
            np.testing.assert_allclose(src_layer.weight.data, tgt_layer.weight.data)
            np.testing.assert_allclose(src_layer.bias.data, tgt_layer.bias.data)

    def test_sequential_parameters_are_traversed(self):
        seq = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
        params = list(seq.parameters())
        # Two Linear layers -> 4 parameters total (2 weights + 2 biases)
        self.assertEqual(len(params), 4)


# -----------------------------------------------------------------------------
# ModuleDict Tests
# -----------------------------------------------------------------------------


class TestModuleDict(unittest.TestCase):
    def test_basic_dict_semantics(self):
        md = ModuleDict({"linear1": Linear(3, 4), "relu": ReLU(), "linear2": Linear(4, 2)})
        self.assertEqual(len(md), 3)
        self.assertIsInstance(md["linear1"], Linear)
        self.assertIn("relu", md)

        types = [type(m).__name__ for m in md]
        self.assertEqual(types, ["Linear", "ReLU", "Linear"])

    def test_parameters_traversal_in_parent_module(self):
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({"l1": Linear(3, 5), "act": ReLU(), "l2": Linear(5, 2)})

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        model = Wrapper()
        params = list(model.parameters())
        # Two Linear layers -> 4 parameters total (2 weights + 2 biases)
        self.assertEqual(len(params), 4)

        x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        y = model(x)
        self.assertEqual(y.shape, (2, 2))
        y.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_state_dict_keys_and_roundtrip(self):
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({"first": Linear(3, 5), "act": ReLU(), "second": Linear(5, 2)})

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        np.random.seed(0)
        w1 = Wrapper(); w2 = Wrapper()

        state = w1.state_dict()
        expected_prefixes = {"layers.first.weight", "layers.first.bias", "layers.second.weight", "layers.second.bias"}
        self.assertTrue(expected_prefixes.issubset(set(state.keys())))

        def resolve_prefix(root, key_parts):
            obj = root
            for part in key_parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = obj[part]  # Access inside ModuleDict
            return obj

        # Zero out target params then load and compare
        for k, t in state.items():
            parts = k.split(".")
            prefix_parts = parts[:-1]
            leaf = parts[-1]
            target_obj = resolve_prefix(w2, prefix_parts)
            if leaf == "weight":
                target_obj.weight.data[...] = 0
            elif leaf == "bias":
                target_obj.bias.data[...] = 0

        w2.load_state_dict(state)

        s2 = w2.state_dict()
        for k in expected_prefixes:
            np.testing.assert_allclose(state[k].data, s2[k].data)

    def test_plain_python_dict_is_not_traversed_like_pytorch(self):
        class Wrapper(Module):
            def __init__(self):
                super().__init__()
                self.layers = {"l1": Linear(3, 5), "act": ReLU(), "l2": Linear(5, 2)}  # raw dict

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        w = Wrapper()
        st = w.state_dict()
        keys = set(st.keys())
        self.assertFalse(any(k.startswith("layers.") for k in keys))

    def test_dynamic_modification_setitem_update(self):
        md = ModuleDict()
        self.assertEqual(len(md), 0)
        md["l0"] = Linear(4, 4)
        self.assertEqual(len(md), 1)
        md.update({"act": ReLU(), "l1": Linear(4, 2)})
        self.assertEqual(len(md), 3)

        class Wrapper(Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        model = Wrapper(md)
        # Two Linear layers -> 4 parameters
        self.assertEqual(len(list(model.parameters())), 4)


if __name__ == '__main__':
    unittest.main()


