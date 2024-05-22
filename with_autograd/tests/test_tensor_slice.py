import unittest
from autograd.tensor import Tensor

class TestTensorSlice(unittest.TestCase):
    def test_simple_slice(self):
        t = Tensor([0, 1, 2, 3, 4, 5], requires_grad=True)
        sliced_t = t[1:4]  # Slicing from index 1 to 3
        
        assert sliced_t.data.tolist() == [1, 2, 3]

        # Propagate a gradient back to the original tensor
        sliced_t.backward(Tensor([1., 1., 1.]))

        assert t.grad.data.tolist() == [0, 1, 1, 1, 0, 0]

    def test_multidimensional_slice(self):
        t = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
        sliced_t = t[:, 1:]  # Slicing all rows, from the second column to the end
        
        assert sliced_t.data.tolist() == [[1, 2], [4, 5]]

        # Propagate a gradient back to the original tensor
        sliced_t.backward(Tensor([[1., 1.], [1., 1.]]))

        assert t.grad.data.tolist() == [[0, 1, 1], [0, 1, 1]]

    def test_non_contiguous_slice(self):
        t = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=True)
        sliced_t = t[::2]  # Slicing with step, taking every second element

        assert sliced_t.data.tolist() == [0, 2, 4, 6, 8]

        # Propagate a gradient back to the original tensor
        sliced_t.backward(Tensor([1., 1., 1., 1., 1.]))

        assert t.grad.data.tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    def test_edge_case_slice(self):
        t = Tensor([0, 1, 2, 3, 4], requires_grad=True)
        sliced_t = t[1:3]  # Slicing a middle section
        
        assert sliced_t.data.tolist() == [1, 2]

        # Propagate a gradient back to the original tensor
        sliced_t.backward(Tensor([1., 1.]))

        assert t.grad.data.tolist() == [0, 1, 1, 0, 0]
    
    def test_reverse_slice(self):
        t = Tensor([0, 1, 2, 3, 4, 5], requires_grad=True)
        reversed_t = t[::-1]  # Reversing the tensor

        assert reversed_t.data.tolist() == [5, 4, 3, 2, 1, 0]

        # Propagate a gradient back to the original tensor with a sequence
        reversed_t.backward(Tensor([1., 2., 3., 4., 5., 6.]))

        # Check that the gradient is also reversed properly
        assert t.grad.data.tolist() == [6., 5., 4., 3., 2., 1.]

    def test_reverse_slice_partial(self):
        t = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=True)
        reversed_t = t[2:8][::-1]  # Reversing a slice of the tensor

        assert reversed_t.data.tolist() == [7, 6, 5, 4, 3, 2]

        # Propagate a gradient back to the original tensor with a sequence
        reversed_t.backward(Tensor([10., 20., 30., 40., 50., 60.]))

        # Check that the gradient is properly placed and reversed within the sliced range
        assert t.grad.data.tolist() == [0, 0, 60., 50., 40., 30., 20., 10., 0, 0]


