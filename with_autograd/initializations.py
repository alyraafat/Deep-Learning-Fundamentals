import numpy as np
from autograd.tensor import Tensor

def glorot_uniform(fan_out: int, fan_in: int) -> np.ndarray:
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))

def zeros(shape: tuple) -> np.ndarray:
    return np.zeros(shape)

def ones(shape: tuple) -> np.ndarray:
    return np.ones(shape)

def xavier_uniform_init(shape: tuple, hidden_size: int):
    k = 1.0 / hidden_size
    sqrt_k = np.sqrt(k)
    return np.random.uniform(-sqrt_k, sqrt_k, shape)