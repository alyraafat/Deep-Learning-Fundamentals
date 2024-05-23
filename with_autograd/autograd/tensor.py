import numpy as np
from typing import List, NamedTuple, Callable, Union, Optional
import inspect

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray, int]
Number = Union[float, int]
Tensorable = Union['Tensor', float, np.ndarray, int]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)
    
class Tensor:
    _global_id = 0  # class variable to keep track of global unique ids

    def __init__(self, data: Arrayable, requires_grad: bool=False, depends_on: List[Dependency]=None, curr_layer: str="", name: str="") -> None:
        self._data = self.ensure_np_array(data)
        self.shape = self._data.shape
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None
        self.depends_on = depends_on or []
        self._curr_layer = curr_layer
        self.name = name
        self.id = Tensor._global_id
        Tensor._global_id += 1
        if self.requires_grad:
            self.zero_grad()
    
    def ensure_np_array(self, arrayable: Arrayable) -> np.ndarray:
        return arrayable if isinstance(arrayable, np.ndarray) else np.array(arrayable)
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32))

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None

    @property
    def curr_layer(self) -> str:
        return self._curr_layer

    @curr_layer.setter
    def curr_layer(self, curr_layer: str) -> None:
        self._curr_layer = curr_layer

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, curr_layer={self.curr_layer})" 
    

    def __add__(self, other) -> 'Tensor':
        from .ops import _add
        """gets called if I do t + other"""
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        from .ops import _add
        """gets called if I do other + t"""
        return _add(ensure_tensor(other), self)
    
    def __iadd__(self, other) -> 'Tensor':
        """when we do t += other"""
        self.data = self.data + ensure_tensor(other).data
        # Invalidate the gradient
        self.grad = None

        return self

    def __mul__(self, other) -> 'Tensor':
        from .ops import _mul
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        from .ops import _mul
        return _mul(ensure_tensor(other), self)
    
    def __imul__(self, other) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        # Invalidate the gradient
        self.grad = None

        return self
    
    def __neg__(self) -> 'Tensor':
        from .ops import _neg
        return _neg(self)
    
    def __sub__(self, other) -> 'Tensor':
        from .ops import _sub
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        from .ops import _sub
        return _sub(ensure_tensor(other), self)
    
    def __isub__(self, other) -> 'Tensor':
        """when we do t -= other"""
        self.data = self.data - ensure_tensor(other).data
        # Invalidate the gradient
        self.grad = None
        return self
    
    def __pow__(self, exponent):
        from .ops import _power
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an integer or float")
        return _power(self, exponent)
    
    def __matmul__(self, other) -> 'Tensor':
        from .ops import _matmul
        return _matmul(self, ensure_tensor(other))
    
    def __getitem__(self, idxs) -> 'Tensor':
        from .ops import _slice
        return _slice(self, idxs)
    
    def __truediv__(self, other):
        # Implement division of tensors
        from .ops import _div
        return _div(self, ensure_tensor(other))

    def __rtruediv__(self, other):
        # Implement right division of tensors
        from .ops import _div
        return _div(ensure_tensor(other), self)

    def __itruediv__(self, other):
        # In-place division
        self.data /= ensure_tensor(other).data
        # Invalidate the gradient if necessary or recalculate if implemented
        self.grad = None
        return self
    
    def backward(self, grad: 'Tensor'=None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None and self.data.size == 1:
            grad = Tensor(1.)
        elif grad is None:
            raise RuntimeError("grad must be specified for non-0-tensor")
        
        # print(f'grad: {grad.data}')
        # print(f'self.grad: {self.grad.data}')
        # var_name = get_var_name(self)
        # var_name = var_name[0] if var_name else "Unknown"
        # print(f"Backward pass for {self.id}:")
        # print(f'self.grad.data shape: {self.grad.data.shape}, grad.data shape: {grad.data.shape}, self.name: {self.name}, grad.name: {grad.name}')
        # print(f'curr layer: {self._curr_layer}')
        self.grad.data += grad.data
        
        # print(f'self.depends_on: {self.depends_on}')
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        from .ops import _tensor_sum
        return _tensor_sum(self)
    
    @staticmethod
    def concatenate(tensors: List['Tensor'], axis: int) -> 'Tensor':
        from .ops import _concat
        return _concat(tensors=tensors, axis=axis)
    
    def flatten(self) -> 'Tensor':
        from .ops import _flatten
        return _flatten(self)

    def reshape(self, *shape) -> 'Tensor':
        from .ops import _reshape
        return _reshape(self, *shape)

    def mean(self, axis: Union[int,tuple]=None) -> 'Tensor':
        from .ops import _tensor_mean
        return _tensor_mean(self, axis=axis)
    
    @property
    def T(self) -> 'Tensor':
        from .ops import _transpose
        return _transpose(self)
    
    def set_item(self, idxs, value):
        from .ops import _set_item
        return _set_item(self, ensure_tensor(value), idxs)
    
    def log(self) -> 'Tensor':
        from .ops import _log
        return _log(self)
    
    def exp(self) -> 'Tensor':
        from .ops import _exp
        return _exp(self)
    
    def sqrt(self) -> 'Tensor':
        from .ops import _sqrt
        return _sqrt(self)
    
    def abs(self) -> 'Tensor':
        from .ops import _abs
        return _abs(self)

    @staticmethod
    def stack(tensors: List['Tensor'], axis: int=0) -> 'Tensor':
        from .ops import _stack
        return _stack(tensors, axis)
    
    @staticmethod
    def minimum(x: 'Tensor', y: 'Tensor') -> 'Tensor':
        from .ops import _minimum
        return _minimum(x, y)
    
    @staticmethod
    def maximum(x: 'Tensor', y: 'Tensor') -> 'Tensor':
        from .ops import _maximum
        return _maximum(x, y)
    
    def max(self, axis: Union[int, tuple]=None) -> 'Tensor':
        from .ops import _max
        return _max(self, axis)
    
    def min(self, axis: Union[int, tuple]=None) -> 'Tensor':
        from .ops import _min
        return _min(self, axis)
    
    def clamp(self, min_val: Number, max_val: Number) -> 'Tensor':
        from .ops import _clamp
        return _clamp(self, min_val, max_val)
    
    def split(self, indices_or_sections: Union[int, list[int]], axis: int) -> List['Tensor']:
        from .ops import _split
        return _split(self, indices_or_sections, axis)

    def pad(self, pad_width: tuple[tuple[int, int]], constant_values: Number=0) -> 'Tensor':
        from .ops import _pad
        return _pad(self, pad_width, constant_values)