import unittest
from autograd.tensor import Tensor

class TestTensorAdd(unittest.TestCase):
    # @unittest.skip("skip")
    def test_add_no_grad(self):
        t1 = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        t2 = Tensor([4.0, 5.0, 6.0], requires_grad=False)
        t3 = t1+t2
        assert t3.data.tolist() == [5.0, 7.0, 9.0]
        assert t3.requires_grad == False

    def test_add_with_grad(self):
        t1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        t2 = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        
        t3 = t1+t2
        assert t3.data.tolist() == [5.0, 7.0, 9.0]
        
        t3.backward(Tensor([2.,-2.,5.]))

        assert t1.grad.data.tolist() == [2.,-2.,5.]
        assert t2.grad.data.tolist() == [2.,-2.,5.]

        t1 += 0.1
        assert t1.grad is None
        assert t1.data.tolist() == [1.1, 2.1, 3.1]

    def test_broadcast_add(self):
        # What is broadcasting? A couple of things:
        # If I do t1 + t2 and t1.shape == t2.shape, it's obvious what to do.
        # but I'm also allowed to add 1s to the beginning of either shape.
        #
        # t1.shape == (10, 5), t2.shape == (5,) => t1 + t2, t2 viewed as (1, 5)
        # t2 = [1, 2, 3, 4, 5] => view t2 as [[1, 2, 3, 4, 5]]
        #
        # The second thing I can do, is that if one tensor has a 1 in some dimension,
        # I can expand it
        # t1 as (10, 5) t2 as (1, 5) is [[1, 2, 3, 4, 5]]

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

        t3 = t1+t2   # shape (2, 3)
        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]

        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        t3 = t1+t2
        grad = [[1, 2, 3], [4, 5, 5]]
        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]

        t3.backward(Tensor(grad))

        assert t1.grad.data.tolist() == grad
        assert t2.grad.data.tolist() == [[5, 7, 8]]