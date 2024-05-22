import unittest
from autograd.tensor import Tensor
import numpy as np

class TestTensorPow(unittest.TestCase):
    def test_sum_no_grad(self):
        inp = np.array([1.0,2.0,3.0])
        t1 = Tensor(inp, requires_grad=True)
        t2 = t1 ** 2
        t2.backward(Tensor([2.,3.,4.]))
        np.testing.assert_array_equal(t2.data, inp ** 2)
        np.testing.assert_array_equal(t1.grad.data, 2*(inp**1) * np.array([2.,3.,4.]))
    
    def test_sum_with_grad(self):
        inp = np.array([1.0,2.0,3.0])
        t1 = Tensor(inp, requires_grad=True)
        t2 = t1 ** 3
        t2.backward(Tensor(4.))
        np.testing.assert_array_equal(t1.grad.data, 3*(inp**2) * np.array([4.]))

    def test_sum_require_no_grad(self):
        inp = np.array([1.0,2.0,3.0])
        t1 = Tensor(inp, requires_grad=False)
        t2 = t1 ** 4
        assert t2.requires_grad == False
        np.testing.assert_array_equal(t2.data, inp ** 4)
        