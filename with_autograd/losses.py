import numpy as np
from autograd import Tensor

def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    loss = ((y_true-y_pred) ** 2).mean()
    loss.curr_layer = 'mse'
    return loss

# def mse_prime(y_true: Tensor, y_pred: Tensor) -> Tensor:
#     return 2*(y_pred-y_true)/y_true.size


def mae(y_true: Tensor,y_pred: Tensor) -> Tensor:
    return ((y_true-y_pred).abs()).mean()
    # return np.mean(np.abs(y_true-y_pred))

# def mae_prime(y_true,y_pred):
#     N = y_true.shape[0]
#     return -((y_true - y_pred) / (abs(y_true - y_pred) +10**-100))/N
#     ## 10**-100 added for stability to avoid div by zero


def binary_crossentropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    eps = 10**-100
    first_term = -y_true*(y_pred+eps).log()
    second_term = -(1-y_true)*(1-y_pred+eps).log()
    return (first_term+second_term).mean()
    # return np.mean(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred))

# def binary_crossentropy_prime(y_true, y_pred):
#     return ((y_pred-y_true)/(y_pred*(1-y_pred)))/np.size(y_true)


def categorical_crossentropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    eps = 10**-100
    return (-y_true*(y_pred+eps).log()).sum()
    # return np.sum(-y_true*np.log(y_pred+10**-100))

# def categorical_crossentropy_prime(y_true, y_pred):
#     return -y_true/(y_pred+10**-100)