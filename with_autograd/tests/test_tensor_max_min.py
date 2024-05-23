import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorMaxMin(unittest.TestCase):

    def test_max_no_grad(self):
        t = Tensor(np.array([[1, 3], [2, 4]]), requires_grad=False)
        max_t = t.max(axis=0)
        
        expected_output = np.array([2, 4])
        np.testing.assert_array_equal(max_t.data, expected_output)
        self.assertFalse(max_t.requires_grad)

    def test_max_with_grad(self):
        t = Tensor(np.array([[1, 3], [2, 4]]), requires_grad=True)
        max_t = t.max(axis=0)
        
        expected_output = np.array([2, 4])
        np.testing.assert_array_equal(max_t.data, expected_output)
        self.assertTrue(max_t.requires_grad)
        self.assertEqual(len(max_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.array([1.0, 1.0])
        max_t.backward(Tensor(grad))
        
        expected_grad = np.array([[0, 0], [1, 1]])
        np.testing.assert_array_equal(t.grad.data, expected_grad)

    def test_min_no_grad(self):
        t = Tensor(np.array([[1, 3], [2, 4]]), requires_grad=False)
        min_t = t.min(axis=0)
        
        expected_output = np.array([1, 3])
        np.testing.assert_array_equal(min_t.data, expected_output)
        self.assertFalse(min_t.requires_grad)

    def test_min_with_grad(self):
        t = Tensor(np.array([[1, 3], [2, 4]]), requires_grad=True)
        min_t = t.min(axis=0)
        
        expected_output = np.array([1, 3])
        np.testing.assert_array_equal(min_t.data, expected_output)
        self.assertTrue(min_t.requires_grad)
        self.assertEqual(len(min_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.array([1.0, 1.0])
        min_t.backward(Tensor(grad))
        
        expected_grad = np.array([[1, 1], [0, 0]])
        np.testing.assert_array_equal(t.grad.data, expected_grad)