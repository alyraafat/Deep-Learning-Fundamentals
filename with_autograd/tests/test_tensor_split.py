import unittest
import numpy as np
from autograd import Tensor


class TestTensorSplit(unittest.TestCase):

    def test_split_no_grad(self):
        t = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=False)
        split_tensors = t.split(3, axis=1)

        expected_output1 = np.array([[1], [4]])
        expected_output2 = np.array([[2], [5]])
        expected_output3 = np.array([[3], [6]])
        
        np.testing.assert_array_equal(split_tensors[0].data, expected_output1)
        np.testing.assert_array_equal(split_tensors[1].data, expected_output2)
        np.testing.assert_array_equal(split_tensors[2].data, expected_output3)
        self.assertFalse(split_tensors[0].requires_grad)
        self.assertFalse(split_tensors[1].requires_grad)
        self.assertFalse(split_tensors[2].requires_grad)

    def test_split_with_grad(self):
        t = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
        split_tensors = t.split(3, axis=1)

        expected_output1 = np.array([[1], [4]])
        expected_output2 = np.array([[2], [5]])
        expected_output3 = np.array([[3], [6]])
        
        np.testing.assert_array_equal(split_tensors[0].data, expected_output1)
        np.testing.assert_array_equal(split_tensors[1].data, expected_output2)
        np.testing.assert_array_equal(split_tensors[2].data, expected_output3)
        self.assertTrue(split_tensors[0].requires_grad)
        self.assertTrue(split_tensors[1].requires_grad)
        self.assertTrue(split_tensors[2].requires_grad)
        
        # Propagate a gradient back to the original tensor
        grad1 = np.array([[1], [1]])
        grad2 = np.array([[1], [1]])
        grad3 = np.array([[1], [1]])
        split_tensors[0].backward(Tensor(grad1))
        split_tensors[1].backward(Tensor(grad2))
        split_tensors[2].backward(Tensor(grad3))
        
        expected_grad = np.array([[1, 1, 1], [1, 1, 1]])
        np.testing.assert_array_equal(t.grad.data, expected_grad)

    def test_split_with_indices_no_grad(self):
        t = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), requires_grad=False)
        split_tensors = t.split([1, 3], axis=1)
        
        expected_output1 = np.array([[1], [5]])
        expected_output2 = np.array([[2, 3], [6, 7]])
        expected_output3 = np.array([[4], [8]])
        
        np.testing.assert_array_equal(split_tensors[0].data, expected_output1)
        np.testing.assert_array_equal(split_tensors[1].data, expected_output2)
        np.testing.assert_array_equal(split_tensors[2].data, expected_output3)
        self.assertFalse(split_tensors[0].requires_grad)
        self.assertFalse(split_tensors[1].requires_grad)
        self.assertFalse(split_tensors[2].requires_grad)

    def test_split_with_indices_with_grad(self):
        t = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), requires_grad=True)
        split_tensors = t.split([1, 3], axis=1)
        
        expected_output1 = np.array([[1], [5]])
        expected_output2 = np.array([[2, 3], [6, 7]])
        expected_output3 = np.array([[4], [8]])
        
        np.testing.assert_array_equal(split_tensors[0].data, expected_output1)
        np.testing.assert_array_equal(split_tensors[1].data, expected_output2)
        np.testing.assert_array_equal(split_tensors[2].data, expected_output3)
        self.assertTrue(split_tensors[0].requires_grad)
        self.assertTrue(split_tensors[1].requires_grad)
        self.assertTrue(split_tensors[2].requires_grad)
        
        # Propagate a gradient back to the original tensor
        grad1 = np.array([[1], [1]])
        grad2 = np.array([[1, 1], [1, 1]])
        grad3 = np.array([[1], [1]])
        split_tensors[0].backward(Tensor(grad1))
        split_tensors[1].backward(Tensor(grad2))
        split_tensors[2].backward(Tensor(grad3))
        
        expected_grad = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        np.testing.assert_array_equal(t.grad.data, expected_grad)