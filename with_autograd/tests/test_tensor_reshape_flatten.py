import unittest
from autograd import Tensor

class TestTensorReshapeFlatten(unittest.TestCase):
    def test_flatten(self):
        t = Tensor([[1, 2], [3, 4]], requires_grad=True)
        flattened_t = t.flatten()
        self.assertEqual(flattened_t.data.tolist(), [1, 2, 3, 4])
        flattened_t.backward(Tensor([1, 1, 1, 1]))
        self.assertEqual(t.grad.data.tolist(), [[1, 1], [1, 1]])

    def test_reshape(self):
        t = Tensor([1, 2, 3, 4], requires_grad=True)
        reshaped_t = t.reshape(-1, 2)
        self.assertEqual(reshaped_t.data.tolist(), [[1, 2], [3, 4]])
        reshaped_t.backward(Tensor([[1, 1], [1, 1]]))
        self.assertEqual(t.grad.data.tolist(), [1, 1, 1, 1])

    def test_multiple_minus_one_dimensions(self):
        t = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
        with self.assertRaises(ValueError) as context:
            t.reshape(-1, -1, 2)
        self.assertIn("Only one dimension can be set to -1", str(context.exception))

    def test_incorrect_total_elements(self):
        t = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
        with self.assertRaises(ValueError) as context:
            t.reshape(2, 5)  # 2 * 5 does not match 6 (total number of elements in t)
        self.assertIn("Total elements in new shape must be equal to the number of elements in tensor", str(context.exception))

    def test_negative_dimension_other_than_minus_one(self):
        t = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
        with self.assertRaises(ValueError) as context:
            t.reshape(-3, 2)
        self.assertIn("Shape dimensions must be non-negative or -1", str(context.exception))

