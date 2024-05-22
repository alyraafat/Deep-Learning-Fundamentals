import numpy as np

from autograd.tensor import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    data = np.tanh(x.data)
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on.append(Dependency(x, grad_fn))

    return Tensor(
        data=data,
        requires_grad=x.requires_grad, 
        depends_on=depends_on,
        name='tanh'
    )

def relu(x: Tensor) -> Tensor:
    data = np.maximum(x.data, 0)
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (x.data > 0)

        depends_on.append(Dependency(x, grad_fn))

    return Tensor(
        data=data,
        requires_grad=x.requires_grad, 
        depends_on=depends_on,
        name='relu'
    )

def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    data = np.where(x.data > 0, x.data, alpha * x.data)
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            grad_copy = np.ones_like(x.data)
            grad_copy[x.data <= 0] = alpha
            return grad * grad_copy

        depends_on.append(Dependency(x, grad_fn))

    return Tensor(
        data=data,
        requires_grad=x.requires_grad, 
        depends_on=depends_on,
        name='leaky_relu'
    )

def sigmoid_eq(x: np.ndarray) -> np.ndarray:
    return 1 / (1+np.exp(-x))

def sigmoid(x: Tensor) ->Tensor:
    data = sigmoid_eq(x.data)
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            grad_sigmoid = sigmoid_eq(grad)
            return grad_sigmoid * (1 - grad_sigmoid)
        depends_on.append(Dependency(x, grad_fn))
    
    return Tensor(
        data=data,
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name='sigmoid'
    )

def softmax_eq(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e_x/np.sum(e_x,axis=-1, keepdims=True)

def softmax(x: Tensor) -> Tensor:
    data = softmax_eq(x.data)
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            s = data.reshape(-1, data.shape[-1])
            grad_output = np.empty_like(grad)
            for i in range(len(s)):
                s_i = s[i].reshape(-1, 1)
                jacobian = np.diagflat(s_i) - np.dot(s_i, s_i.T)
                grad_output[i] = grad[i] @ jacobian
            return grad_output.reshape(grad.shape)
        depends_on.append(Dependency(x, grad_fn))
    
    return Tensor(
        data=data,
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name='softmax'
    )