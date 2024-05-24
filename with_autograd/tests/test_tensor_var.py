import numpy as np
import unittest
from autograd import Tensor


class TestTensorVar(unittest.TestCase):
    def test_var_no_axis(self):
        data = np.random.randn(3, 4)
        x = Tensor(data, requires_grad=True)
        var_x = x.var(axis=None, ddof=0, keepdims=True)
        expected_var = np.var(data, ddof=0, keepdims=True)
        
        np.testing.assert_almost_equal(var_x.data, expected_var, decimal=5)

    def test_var_with_axis(self):
        data = np.random.randn(3, 4)
        x = Tensor(data, requires_grad=True)
        var_x = x.var(axis=1, ddof=0, keepdims=True)
        expected_var = np.var(data, axis=1, ddof=0, keepdims=True)
        
        np.testing.assert_almost_equal(var_x.data, expected_var, decimal=5)

    def test_var_with_multiple_axes(self):
        data = np.random.randn(3, 4, 5)
        x = Tensor(data, requires_grad=True)
        var_x = x.var(axis=(0, 2), ddof=0, keepdims=True)
        expected_var = np.var(data, axis=(0, 2), ddof=0, keepdims=True)
        
        np.testing.assert_almost_equal(var_x.data, expected_var, decimal=5)

    def test_var_grad(self):
        data = np.random.randn(3, 4)
        x = Tensor(data, requires_grad=True)
        var_x = x.var(axis=None, ddof=0, keepdims=True)
        # print(var_x.shape)
        # grad_output = np.ones_like(var_x.data)
        # var_x.backward(grad_output)
        var_x.backward()
        
        expected_grad = 2 * (x.data - np.mean(x.data)) / x.data.size
        
        np.testing.assert_almost_equal(x.grad.data, expected_grad, decimal=5)

    def test_var_grad_with_axis(self):
        data = np.random.randn(3, 4)
        x = Tensor(data, requires_grad=True)
        var_x = x.var(axis=1, ddof=0, keepdims=True)

        grad_output = np.ones_like(var_x.data)
        var_x.backward(grad_output)
        
        reduction_size = x.data.shape[1]
        expected_grad = 2 * (x.data - np.mean(x.data, axis=1, keepdims=True)) / reduction_size
        
        np.testing.assert_almost_equal(x.grad.data, expected_grad, decimal=5)