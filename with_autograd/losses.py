import numpy as np
from autograd import Tensor

def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    loss = ((y_true-y_pred) ** 2).mean()
    loss.curr_layer = 'mse'
    return loss

def mae(y_true: Tensor,y_pred: Tensor) -> Tensor:
    return ((y_true-y_pred).abs()).mean()


def binary_crossentropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    eps = 10**-100
    first_term = -y_true*(y_pred+eps).log()
    second_term = -(1-y_true)*(1-y_pred+eps).log()
    return (first_term+second_term).mean()


def categorical_crossentropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    eps = 10**-100
    return (-y_true*(y_pred+eps).log()).sum()
