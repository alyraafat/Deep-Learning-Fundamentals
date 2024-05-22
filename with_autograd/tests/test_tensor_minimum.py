import numpy as np
import unittest
from autograd import Tensor

class TestTensorMinimum(unittest.TestCase):
    def test_minimum(self):
        a = Tensor(np.array([1, 3, 5]), requires_grad=True, name="a")
        b = Tensor(np.array([2, 3, 4]), requires_grad=True, name="b")

        result = Tensor.minimum(a, b)
        expected_data = np.array([1, 3, 4])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        # Perform backward pass
        result_sum = result.sum()
        result_sum.backward()

        expected_grad_a = np.array([1, 1, 0])  # gradient should flow where a <= b
        expected_grad_b = np.array([0, 0, 1])  # gradient should flow where b < a

        np.testing.assert_array_equal(a.grad.data, expected_grad_a)
        np.testing.assert_array_equal(b.grad.data, expected_grad_b)

    def test_minimum_broadcasting(self):
        a = Tensor(np.array([[1, 3, 5], [1, 3, 5]]), requires_grad=True, name="a")
        b = Tensor(np.array([2, 3, 4]), requires_grad=True, name="b")

        result = Tensor.minimum(a, b)
        expected_data = np.array([[1, 3, 4], [1, 3, 4]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        # Perform backward pass
        result_sum = result.sum()
        result_sum.backward()

        expected_grad_a = np.array([[1, 1, 0], [1, 1, 0]])  # gradient should flow where a <= b
        expected_grad_b = np.array([0, 0, 2])  # gradient should flow where b < a

        np.testing.assert_array_equal(a.grad.data, expected_grad_a)
        np.testing.assert_array_equal(b.grad.data, expected_grad_b)