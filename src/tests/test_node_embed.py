import unittest

import torch

from pyggnn.nn.node_embed import AtomicNum2Node, AtomicDict2Node


class TestAtomicNum2Node(unittest.TestCase):
    def setUp(self):
        self.node_dim = 10
        self.max_num = 100
        n_node = 5
        self.z = torch.randint(0, self.max_num, (n_node,))
        self.c = torch.rand(n_node)

    def test_charge(self):
        charge_choises = [True, False]
        for charge in charge_choises:
            with self.subTest(charge=charge):
                node_embed = AtomicNum2Node(self.node_dim, self.max_num, charge=charge)
                output = node_embed(self.z, self.c)
                if charge:
                    self.assertEqual(output.shape, (self.z.shape[0], self.node_dim + 1))
                else:
                    self.assertEqual(output.shape, (self.z.shape[0], self.node_dim))


class TestAtomicDict2Node(unittest.TestCase):
    def setUp(self):
        self.node_dim = 10
        self.max_num = 100
        n_node = 5
        self.z = torch.randint(0, self.max_num, (n_node,))
        self.c = torch.rand(n_node)

    def test_charge(self):
        charge_choises = [True, False]
        for charge in charge_choises:
            with self.subTest(charge=charge):
                node_embed = AtomicDict2Node(self.node_dim, self.max_num, charge=charge)
                output = node_embed(self.z, self.c)
                if charge:
                    self.assertEqual(output.shape, (self.z.shape[0], self.node_dim + 1))
                else:
                    self.assertEqual(output.shape, (self.z.shape[0], self.node_dim))

if __name__ == "__main__":
    unittest.main()
