import unittest
from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_sum_no_grad(self):
        inp = [1.0,2.0,3.0]
        t1 = Tensor(inp, requires_grad=True)
        t2 = t1.sum()
        t2.backward()
        self.assertEqual(t1.grad.data.tolist(), [1.0] * len(inp))
    
    def test_sum_with_grad(self):
        inp = [1.0,2.0,3.0]
        t1 = Tensor(inp, requires_grad=True)
        t2 = t1.sum()
        t2.backward(Tensor(4.))
        self.assertEqual(t1.grad.data.tolist(),[4.0] * len(inp))

    def test_sum_require_no_grad(self):
        inp = [1.0,2.0,3.0]
        t1 = Tensor(inp, requires_grad=False)
        t2 = t1.sum()
        self.assertEqual(t2.requires_grad,False)
        