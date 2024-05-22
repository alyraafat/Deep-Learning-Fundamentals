import unittest
import numpy as np
from autograd import Tensor


class TestTensorLogExpSqrtAbs(unittest.TestCase):
    def test_log(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, name="a")
        result = a.log()
        expected_data = np.log(a.data)
        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        result_sum = result.sum()
        result_sum.backward()

        expected_grad = np.array([1.0, 0.5, 1/3])
        np.testing.assert_allclose(a.grad.data, expected_grad)

    def test_exp(self):
        a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True, name="a")
        result = a.exp()
        expected_data = np.exp(a.data)
        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        result_sum = result.sum()
        result_sum.backward()

        expected_grad = np.exp(a.data)
        np.testing.assert_allclose(a.grad.data, expected_grad)

    def test_sqrt(self):
        a = Tensor(np.array([1.0, 4.0, 9.0]), requires_grad=True, name="a")
        result = a.sqrt()
        expected_data = np.sqrt(a.data)
        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        result_sum = result.sum()
        result_sum.backward()

        expected_grad = np.array([0.5, 0.25, 1/6])
        np.testing.assert_allclose(a.grad.data, expected_grad)

    def test_abs(self):
        a = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True, name="a")
        result = a.abs()
        expected_data = np.abs(a.data)
        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        result_sum = result.sum()
        result_sum.backward()

        expected_grad = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(a.grad.data, expected_grad)