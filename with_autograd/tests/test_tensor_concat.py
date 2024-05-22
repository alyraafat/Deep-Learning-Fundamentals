import unittest
from autograd import Tensor
import numpy as np

class TestTensorConcat(unittest.TestCase):
    def test_concat_data_axis_0(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), name="c")

        result = Tensor.concatenate([a, b, c], axis=0)
        expected_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertFalse(result.requires_grad)

    def test_concat_data_axis_1(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), name="c")

        result = Tensor.concatenate([a, b, c], axis=1)
        expected_data = np.array([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertFalse(result.requires_grad)
    
    def test_concat_requires_grad(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=True, name="c")

        result = Tensor.concatenate([a, b, c], axis=0)
        expected_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)
    
    def test_concat_mixed_requires_grad(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=False, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=True, name="c")

        result = Tensor.concatenate([a, b, c], axis=0)
        expected_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)
    
    def test_concat_backward_axis_0(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=True, name="c")

        result = Tensor.concatenate([a, b, c], axis=0)

        np.testing.assert_array_equal(result.data, np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]))
        result_sum = result.sum()
        # print(f"result_sum: {result_sum}")
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        np.testing.assert_array_equal(c.grad.data, np.ones_like(c.data))

    def test_concat_backward_axis_1(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=True, name="c")

        result = Tensor.concatenate([a, b, c], axis=1)
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        np.testing.assert_array_equal(c.grad.data, np.ones_like(c.data))

    def test_concat_no_gradients(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=False, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=False, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=False, name="c")

        result = Tensor.concatenate([a, b, c], axis=0)
        result_sum = result.sum()
        if result.requires_grad:
            result_sum.backward()

        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)
        self.assertIsNone(c.grad)

    def test_concat_axis_minus_1(self):
        a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
        b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
        c = Tensor(np.array([[9, 10], [11, 12]]), requires_grad=False, name="c")

        result = Tensor.concatenate([a, b, c], axis=-1)
        expected_data = np.array([[ 1,  2,  5,  6,  9, 10],
                                [ 3,  4,  7,  8, 11, 12]])

        np.testing.assert_array_equal(result.data, expected_data)
        self.assertTrue(result.requires_grad)

        # Perform backward pass
        result_sum = result.sum()
        result_sum.backward()

        np.testing.assert_array_equal(a.grad.data, np.ones_like(a.data))
        np.testing.assert_array_equal(b.grad.data, np.ones_like(b.data))
        self.assertIsNone(c.grad)  # c does not require gradients
