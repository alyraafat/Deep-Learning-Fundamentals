import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorClamp(unittest.TestCase):

    def test_clamp_no_grad(self):
        t = Tensor(np.array([0.5, 2.0, -1.0, 4.5]), requires_grad=False)
        min_val, max_val = 0.0, 3.0
        clamped_t = t.clamp(min_val, max_val)
        
        expected_output = np.array([0.5, 2.0, 0.0, 3.0])
        np.testing.assert_array_equal(clamped_t.data, expected_output)
        self.assertFalse(clamped_t.requires_grad)

    def test_clamp_with_grad(self):
        t = Tensor(np.array([0.5, 2.0, -1.0, 4.5]), requires_grad=True)
        min_val, max_val = 0.0, 3.0
        clamped_t = t.clamp(min_val, max_val)
        
        expected_output = np.array([0.5, 2.0, 0.0, 3.0])
        np.testing.assert_array_equal(clamped_t.data, expected_output)
        self.assertTrue(clamped_t.requires_grad)
        self.assertEqual(len(clamped_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.array([1.0, 1.0, 1.0, 1.0])
        clamped_t.backward(Tensor(grad))
        
        expected_grad = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(t.grad.data, expected_grad)

    def test_clamp_3d_tensor(self):
        t = Tensor(np.array([[[0.5, 4.0], [-1.0, 2.5]], [[-3.0, 5.5], [0.0, 1.0]]]), requires_grad=False)
        min_val, max_val = 0.0, 3.0
        clamped_t = t.clamp(min_val, max_val)
        
        expected_output = np.array([[[0.5, 3.0], [0.0, 2.5]], [[0.0, 3.0], [0.0, 1.0]]])
        np.testing.assert_array_equal(clamped_t.data, expected_output)
        self.assertFalse(clamped_t.requires_grad)

    def test_clamp_3d_tensor_with_grad(self):
        t = Tensor(np.array([[[0.5, 4.0], [-1.0, 2.5]], [[-3.0, 5.5], [0.0, 1.0]]]), requires_grad=True)
        min_val, max_val = 0.0, 3.0
        clamped_t = t.clamp(min_val, max_val)
        
        expected_output = np.array([[[0.5, 3.0], [0.0, 2.5]], [[0.0, 3.0], [0.0, 1.0]]])
        np.testing.assert_array_equal(clamped_t.data, expected_output)
        self.assertTrue(clamped_t.requires_grad)
        self.assertEqual(len(clamped_t.depends_on), 1)
        
        # Propagate a gradient back to the original tensor
        grad = np.ones_like(clamped_t.data)
        clamped_t.backward(Tensor(grad))
        
        expected_grad = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
        np.testing.assert_array_equal(t.grad.data, expected_grad)