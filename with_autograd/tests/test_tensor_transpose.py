import unittest
import numpy as np
from autograd import Tensor  # Adjust the import according to your module setup

class TestTranspose(unittest.TestCase):
    def test_transpose_data_correctness(self):
        # Test to ensure the data is transposed correctly
        data = np.array([[1, 2], [3, 4]])
        tensor = Tensor(data, requires_grad=True)
        transposed_tensor = tensor.T

        # Check if data is correctly transposed
        np.testing.assert_array_equal(transposed_tensor.data, data.T, "The tensor data should be transposed.")

    def test_transpose_gradient_propagation(self):
        # Test to ensure gradients are propagated correctly through transpose operation
        data = np.array([[1, 2, 3], [4, 5, 6]])
        tensor = Tensor(data, requires_grad=True)
        transposed_tensor = tensor.T

        # Assume some gradient is passed to the transposed tensor during backpropagation
        grad = np.array([[1, 1], [1, 1], [1, 1]])  # This should match the shape of the transposed tensor
        transposed_tensor.backward(Tensor(grad))

        # Expected gradient on the original tensor should be the transpose of the gradient passed
        expected_grad = grad.T
        np.testing.assert_array_equal(tensor.grad.data, expected_grad, "Gradients should be transposed back to match the original tensor shape.")


