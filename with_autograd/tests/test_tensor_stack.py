import unittest
from autograd import Tensor
import numpy as np



class TestTensorStack(unittest.TestCase):
    def test_stack_data(self):
        a = Tensor(np.array([1, 2, 3]), name="a")
        b = Tensor(np.array([4, 5, 6]), name="b")
        c = Tensor(np.array([7, 8, 9]), name="c")

        result = Tensor.stack([a, b, c], axis=0)
        expected_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertFalse(result.requires_grad)
    
    def test_stack_requires_grad(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=True, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=0)
        expected_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)
    
    def test_stack_backward(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=True, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=0)

        np.testing.assert_array_equal(result.data, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        # split_grad = np.split(result.data, 3, axis=0)
        # print(f'split_grad: {split_grad}, split_grad[0]: {split_grad[0]}, split_grad[1]: {split_grad[1]}, split_grad[2]: {split_grad[2]}')
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        self.assertIsNone(c.grad)  # c does not require gradients

    def test_stack_axis(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=True, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=1)
        expected_data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

    def test_stack_data_axis_0(self):
        a = Tensor(np.array([1, 2, 3]), name="a")
        b = Tensor(np.array([4, 5, 6]), name="b")
        c = Tensor(np.array([7, 8, 9]), name="c")

        result = Tensor.stack([a, b, c], axis=0)
        expected_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertFalse(result.requires_grad)

    def test_stack_data_axis_1(self):
        a = Tensor(np.array([1, 2, 3]), name="a")
        b = Tensor(np.array([4, 5, 6]), name="b")
        c = Tensor(np.array([7, 8, 9]), name="c")

        result = Tensor.stack([a, b, c], axis=1)
        expected_data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertFalse(result.requires_grad)
    
    def test_stack_mixed_requires_grad(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=False, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=True, name="c")

        result = Tensor.stack([a, b, c], axis=0)
        expected_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)
    
    def test_stack_backward_axis_0(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=True, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=0)
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        self.assertIsNone(c.grad)  # c does not require gradients

    def test_stack_backward_axis_1(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=True, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=True, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=1)
        # split_grad = np.split(result.data, 3, axis=1)
        # print(f'split_grad: {split_grad}, split_grad[0]: {split_grad[0]}, split_grad[1]: {split_grad[1]}, split_grad[2]: {split_grad[2]}')
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        self.assertIsNone(c.grad)  # c does not require gradients

    def test_stack_no_gradients(self):
        a = Tensor(np.array([1, 2, 3]), requires_grad=False, name="a")
        b = Tensor(np.array([4, 5, 6]), requires_grad=False, name="b")
        c = Tensor(np.array([7, 8, 9]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=0)
        result_sum = result.sum()
        if result.requires_grad:
            result_sum.backward()

        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertIsNone(c.grad)

    def test_stack_axis_minus_1(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=False, name="c")

        result = Tensor.stack([a, b, c], axis=-1)
        expected_data = np.array([[[ 1,  5,  9],
                                [ 2,  6, 10]],
                                [[ 3,  7, 11],
                                [ 4,  8, 12]]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        # Perform backward pass
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        self.assertIsNone(c.grad)  # c does not require gradients
