import unittest
import numpy as np
from autograd import Tensor  # Adjust this import based on your actual file structure

class TestTensorSetItem(unittest.TestCase):

    def test_set_item_updates_data_correctly(self):
        # Test to ensure data is updated correctly
        tensor = Tensor([[1, 2], [3, 4]], requires_grad=True)
        modified_tensor = tensor.set_item((0, 1), 10)
        # Check the modified tensor's data
        expected_data = np.array([[1, 10], [3, 4]], dtype=np.float32)
        np.testing.assert_array_almost_equal(modified_tensor.data, expected_data, err_msg="Tensor data should be updated correctly.")

    def test_set_item_gradient_propagation(self):
        # Test to ensure gradients are handled correctly
        tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        modified_tensor = tensor.set_item((0, 1), 10.0)
        modified_tensor.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
        # Expected gradient should be the same as the passed gradient but zeroed out at non-modified indices
        expected_grad = np.array([[0, 1], [0, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(tensor.grad.data, expected_grad, err_msg="Gradients should be propagated correctly through modified indices.")

    def test_set_item_does_not_change_original_tensor(self):
        # Test that the original tensor remains unchanged if a new tensor is returned
        tensor = Tensor([[1, 2], [3, 4]], requires_grad=True)
        modified_tensor = tensor.set_item((0, 1), 10)
        expected_original_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        np.testing.assert_array_almost_equal(tensor.data, expected_original_data, err_msg="Original tensor data should remain unchanged.")

