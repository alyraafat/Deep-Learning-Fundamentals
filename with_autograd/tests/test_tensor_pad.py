import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorPad(unittest.TestCase):
    
    def test_pad_no_grad(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=False)
        pad_width = ((1, 1), (1, 1))
        constant_values = 0
        padded_t = t.pad(pad_width, constant_values)
        
        expected_output = np.array([[0, 0, 0, 0],
                                    [0, 1, 2, 0],
                                    [0, 3, 4, 0],
                                    [0, 0, 0, 0]])

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertFalse(padded_t.requires_grad)
    
    def test_pad_with_grad(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        pad_width = ((1, 1), (1, 1))
        constant_values = 0
        padded_t = t.pad(pad_width, constant_values)
        
        expected_output = np.array([[0, 0, 0, 0],
                                    [0, 1, 2, 0],
                                    [0, 3, 4, 0],
                                    [0, 0, 0, 0]])

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertTrue(padded_t.requires_grad)
        self.assertEqual(len(padded_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.ones_like(padded_t.data)
        expected_grad = np.array([[1, 1], [1, 1]])
        padded_t.backward(Tensor(grad))

        np.testing.assert_array_equal(t.grad.data, expected_grad)
    
    def test_pad_different_constant_value(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=False)
        pad_width = ((1, 1), (1, 1))
        constant_values = 5
        padded_t = t.pad(pad_width, constant_values)
        
        expected_output = np.array([[5, 5, 5, 5],
                                    [5, 1, 2, 5],
                                    [5, 3, 4, 5],
                                    [5, 5, 5, 5]])

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertFalse(padded_t.requires_grad)
    
    def test_pad_3d_tensor(self):
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=False)
        pad_width = ((1, 1), (1, 1), (1, 1))
        constant_values = 0
        padded_t = t.pad(pad_width, constant_values)

        
        expected_output = np.array([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 5, 6, 0],
                [0, 7, 8, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        
        # print(f'expected_output: {expected_output.shape}, padded_t: {padded_t.shape}')

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertFalse(padded_t.requires_grad)
    
    def test_pad_3d_tensor_with_grad(self):
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
        pad_width = ((1, 1), (1, 1), (1, 1))
        constant_values = 0
        padded_t = t.pad(pad_width, constant_values)
        
        expected_output = np.array([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 5, 6, 0],
                [0, 7, 8, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertTrue(padded_t.requires_grad)
        self.assertEqual(len(padded_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.ones_like(padded_t.data)
        expected_grad = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        padded_t.backward(Tensor(grad))

        np.testing.assert_array_equal(t.grad.data, expected_grad)

    
    def test_pad_different_widths(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        pad_width = ((1, 2), (2, 1))
        constant_values = 0
        padded_t = t.pad(pad_width, constant_values)
        
        expected_output = np.array([[0, 0, 0, 0, 0],
                                    [0, 0, 1, 2, 0],
                                    [0, 0, 3, 4, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])

        np.testing.assert_array_equal(padded_t.data, expected_output)
        self.assertTrue(padded_t.requires_grad)
        self.assertEqual(len(padded_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.ones_like(padded_t.data)
        expected_grad = np.array([[1, 1], [1, 1]])
        padded_t.backward(Tensor(grad))

        np.testing.assert_array_equal(t.grad.data, expected_grad)



