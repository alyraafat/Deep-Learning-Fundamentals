import numpy as np
from . import Tensor, Dependency, ensure_tensor
from typing import Callable, Union

Number = Union[float, int]

def _tensor_sum(x: Tensor) -> Tensor:
    """Sum of tensor"""

    if not x.requires_grad:
        return Tensor(data=x.data.sum(), requires_grad=False, depends_on=[])
    
    return Tensor(
        data=x.data.sum(),
        requires_grad=x.requires_grad,
        depends_on=[Dependency(tensor=x, grad_fn=lambda grad: grad * np.ones_like(x.data))],
        name=f'{x.name}_sum'
    )

def _adjust_grad_for_broadcasting(grad: np.ndarray, z: Tensor) -> np.ndarray:

    added_dims = grad.ndim - z.data.ndim
    for _ in range(added_dims):
        grad = grad.sum(axis=0)
    
    for i, dim in enumerate(z.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

def _add(x: Tensor, y: Tensor) -> Tensor:
    """Add two tensors"""

    def grad_fn_wrapper(z: Tensor) -> Callable[[np.ndarray], np.ndarray]:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # print(f' x: {x.shape}, y: {y.shape}, z: {z.shape}, grad: {grad.shape}')
            # print(f' x: {x.requires_grad}, y: {y.requires_grad}, z: {z.requires_grad}')
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            # print(f'after assert')
            grad = _adjust_grad_for_broadcasting(grad=grad, z=z)
            return grad
        return grad_fn
    
    depends_on = []
    if x.requires_grad:
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn_wrapper(x)))

    if y.requires_grad:
        depends_on.append(Dependency(tensor=y, grad_fn=grad_fn_wrapper(y)))
    
    return Tensor(
        data = x.data + y.data,
        requires_grad = x.requires_grad or y.requires_grad,
        depends_on = depends_on,
        name=f'{x.name}_add_{y.name}'
    )

def _mul(x: Tensor, y: Tensor) -> Tensor:
    """Multiply two tensors"""

    def grad_fn_wrapper(curr: Tensor, other: Tensor) -> Callable[[np.ndarray], np.ndarray]:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            grad = grad * other.data
            # print(f'grad: {grad}')
            grad = _adjust_grad_for_broadcasting(grad=grad, z=curr)
            # print(f'g rad after broadcast func: {grad}')
            return grad
        return grad_fn
    
    depends_on = []
    if x.requires_grad:
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn_wrapper(curr=x, other=y)))

    if y.requires_grad:
        depends_on.append(Dependency(tensor=y, grad_fn=grad_fn_wrapper(curr=y, other=x)))
    
    return Tensor(
        data = x.data * y.data,
        requires_grad = x.requires_grad or y.requires_grad,
        depends_on = depends_on,
        name=f'{x.name}_mul_{y.name}'
    )

def _neg(x: Tensor) -> Tensor:
    """Negate a tensor"""
    depends_on = []
    if x.requires_grad:
        depends_on.append(Dependency(tensor=x, grad_fn=lambda grad: -grad))
    return Tensor(
        data = -x.data,
        requires_grad = x.requires_grad,
        depends_on = depends_on,
        name=f'neg_{x.name}'
    )

def _sub(x: Tensor, y: Tensor) -> Tensor:
    """Subtract two tensors"""
    return x + -y

def _matmul(x: Tensor, y: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors

    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3

    """
    
    def grad_fn_wrapper(other: Tensor, is_curr_first_term: bool) -> Callable[[np.ndarray], np.ndarray]:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if is_curr_first_term:
                return grad @ other.data.T
            else:
                return other.data.T @ grad
        return grad_fn
    
    depends_on = []
    if x.requires_grad:
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn_wrapper(other=y, is_curr_first_term=True)))
    if y.requires_grad:
        depends_on.append(Dependency(tensor=y, grad_fn=grad_fn_wrapper(other=x, is_curr_first_term=False)))
    return Tensor(
        data=x.data @ y.data,
        requires_grad=x.requires_grad or y.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_matmul_{y.name}'
    )

def _slice(x: Tensor, idxs) -> Tensor:
    """Slice a tensor"""

    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            zeros_grad = np.zeros_like(x.data)
            zeros_grad[idxs] = grad
            return zeros_grad
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=x.data[idxs],
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_slice_{idxs}'
    )

def _power(x: Tensor, pow: Number) -> Tensor:
    """raise tensor to power of pow"""

    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * pow * x.data**(pow-1)
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    
    return Tensor(
        data=x.data**pow,
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_power_{pow}'
    )

# def _div(x: Tensor, y: Tensor) -> Tensor:
#     """ divide x by y = x/y"""

#     depends_on = []
#     if x.requires_grad:
#         def grad_fn(grad: np.ndarray) -> np.ndarray:
#             # TODO: handle broadcasting
#             return grad / y.data
#         depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
#     if y.requires_grad:
#         def grad_fn(grad: np.ndarray) -> Tensor:
#             # TODO: handle broadcasting
#             return (-grad * x.data) / y.data**2
#         depends_on.append(Dependency(tensor=y,grad_fn=grad_fn))
#     return Tensor(
#         data=x.data / y.data,
#         requires_grad=x.requires_grad or y.requires_grad,
#         depends_on=depends_on
#     )

def _div(x: Tensor, y: Tensor) -> Tensor:
    """ divide x by y = x/y"""
    return x * (y**-1)
    
def _concat(tensors: list[Tensor], axis: int) -> Tensor:
    """concatenate 2 tensors along axis"""
    depends_on = []
    requires_grad = False
    data_to_concat = [tensor.data for tensor in tensors]
    start = 0
    for tensor in tensors:
        if not (0 <= axis < tensor.data.ndim or axis == -1):
            raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.data.ndim} dimensions")
        requires_grad = requires_grad or tensor.requires_grad
        if tensor.requires_grad:
            def grad_fn(grad: np.ndarray, start: int=start, tensor: Tensor=tensor) -> np.ndarray:
                condition = lambda i: i != axis if axis != -1 else i != grad.ndim - 1
                slices = tuple(slice(None) if condition(i=i) else slice(start, start+tensor.shape[axis]) for i in range(grad.ndim))
                return grad[slices]
            depends_on.append(Dependency(tensor=tensor, grad_fn=grad_fn))
        start += tensor.shape[axis]

    return Tensor(
        data=np.concatenate(data_to_concat, axis=axis),
        requires_grad=requires_grad,
        depends_on=depends_on,
        name=f'concat_{axis}'
    )

def _flatten(x: Tensor) -> Tensor:
    """flatten x"""

    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(x.shape)
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    
    return Tensor(
        data=x.data.flatten(),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_flatten'
    )

def _reshape(x: Tensor, *shape) -> Tensor:
    """reshape x"""
    # Calculate the product of the specified dimensions and handle -1 if present
    special_dim = -1  # This will store the index of the dimension set to -1
    product = 1
    for i, dim in enumerate(shape):
        if dim == -1:
            if special_dim != -1:  # More than one -1 in the shape
                raise ValueError("Only one dimension can be set to -1")
            special_dim = i
        elif dim < 0:
            raise ValueError("Shape dimensions must be non-negative or -1")
        else:
            product *= dim

    if special_dim != -1:  # If there is a dimension set to -1
        if x.data.size % product != 0:
            raise ValueError("The total size of the new array must be unchanged")
        shape = list(shape)
        shape[special_dim] = x.data.size // product  # Calculate the missing dimension
        shape = tuple(shape)
    else:
        if product != x.data.size:
            raise ValueError("Total elements in new shape must be equal to the number of elements in tensor")
    
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(x.shape)
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    
    return Tensor(
        data=x.data.reshape(*shape),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_reshape_{shape}'
    )

def _tensor_mean(x: Tensor, axis: Union[int, tuple]) -> Tensor:
    """ calculate mean of tensor """
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if axis is None:
                reduction_size = x.data.size
                expanded_grad = grad
            else:
                reduction_size = np.prod([x.shape[ax] for ax in axis])
                expanded_grad = np.expand_dims(grad, axis=axis)
                for ax in sorted(axis):
                    expanded_grad = np.repeat(expanded_grad, x.shape[ax], axis=ax)
            return expanded_grad / reduction_size
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.mean(x.data, axis=axis),
        requires_grad=x.requires_grad,
        depends_on=depends_on
    )

def _transpose(x: Tensor) -> Tensor:
    """Transpose x"""

    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.T
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=x.data.T,
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_transpose'
    ) 

def _set_item(x: Tensor, y: Tensor, idxs) -> Tensor:
    """set item"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            new_grad = np.zeros_like(x.data)
            new_grad[idxs] = grad[idxs]  # Only the modified index carries the gradient
            return new_grad
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    copied_data = np.copy(x.data)
    copied_data[idxs] = y.data
    return Tensor(
        data= copied_data,
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'{x.name}_set_item_{idxs}'
    )

def _log(x: Tensor) -> Tensor:
    """log of x"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / x.data
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.log(x.data),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'log_{x.name}'
    )

def _exp(x: Tensor) -> Tensor:
    """exponential of x"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.exp(x.data)
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.exp(x.data),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'exp_{x.name}'
    )

def _sqrt(x: Tensor) -> Tensor:
    """square root of x"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / (2 * np.sqrt(x.data))
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.sqrt(x.data),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'sqrt_{x.name}'
    )

def _abs(x: Tensor) -> Tensor:
    """abs of x"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.sign(x.data)
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.abs(x.data),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'abs_{x.name}'
    )

def _stack(tensors: list[Tensor], axis: int) -> Tensor:
    """Stack tensors along axis"""
    depends_on = []
    requires_grad = False
    data_to_stack = [tensor.data for tensor in tensors]
    for i, tensor in enumerate(tensors):
        requires_grad = requires_grad or tensor.requires_grad
        if tensor.requires_grad:
            def grad_fn(grad: np.ndarray, i: int=i) -> np.ndarray:
                split_grad = np.split(grad, len(tensors), axis=axis)
                # print(f'split_grad: {split_grad.shape}, split_grad[i]: {split_grad[i].shape}')
                return split_grad[i].reshape(tensor.data.shape)
            depends_on.append(Dependency(tensor=tensor, grad_fn=grad_fn))
    return Tensor(
        data=np.stack(data_to_stack, axis=axis),
        requires_grad=requires_grad,
        depends_on=depends_on,
        name=f'stack_{axis}'
    )

def _minimum(x: Tensor, y: Tensor) -> Tensor:
    """element-wise minimum of x and y"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            new_grad = grad * (x.data <= y.data)
            new_grad = _adjust_grad_for_broadcasting(grad=new_grad, z=x)
            return new_grad
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    if y.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            new_grad = grad * (y.data < x.data)
            new_grad = _adjust_grad_for_broadcasting(grad=new_grad, z=y)
            return new_grad
        depends_on.append(Dependency(tensor=y, grad_fn=grad_fn))
    return Tensor(
        data=np.minimum(x.data, y.data),
        requires_grad=x.requires_grad or y.requires_grad,
        depends_on=depends_on,
        name=f'min_{x.name}_{y.name}'
    )

def _maximum(x: Tensor, y: Tensor) -> Tensor:
    """element-wise maximum of x and y"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            new_grad = grad * (x.data >= y.data)
            new_grad = _adjust_grad_for_broadcasting(grad=new_grad, z=x)
            return new_grad
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    if y.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            assert grad.shape == x.shape or grad.shape == y.shape, "shape mismatch"
            new_grad = grad * (y.data > x.data)
            new_grad = _adjust_grad_for_broadcasting(grad=new_grad, z=y)
            return new_grad
        depends_on.append(Dependency(tensor=y, grad_fn=grad_fn))
    return Tensor(
        data=np.maximum(x.data, y.data),
        requires_grad=x.requires_grad or y.requires_grad,
        depends_on=depends_on,
        name=f'max_{x.name}_{y.name}'
    )

def _max(x: Tensor, axis: Union[int, tuple]) -> Tensor:
    """element-wise maximum along axis"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # if axis == None:
            #     new_grad = grad * (x.data == np.max(x.data))
            # else:
            #     #TODO: handle axis
            #     expanded_grad = np.expand_dims(grad, axis=axis)
            #     for ax in sorted(axis):
            #         expanded_grad = np.repeat(expanded_grad, x.shape[ax], axis=ax)
            #     expanded_grad = expanded_grad * (x.data == np.max(x.data, axis=axis))
            #     new_grad = expanded_grad
            
            expanded_grad = grad if axis is None else np.expand_dims(grad, axis)
            # Repeat along the specified axis to match the original shape
            expanded_grad = np.broadcast_to(expanded_grad, x.data.shape)
            # Create a mask where the maximum values are
            max_mask = x.data == np.max(x.data, axis=axis, keepdims=True)
            # Grad is only passed to the positions that had the maximum value
            new_grad = expanded_grad * max_mask
            return new_grad
        
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.max(x.data, axis=axis),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'max_{x.name}'
    )

def _min(x: Tensor, axis: Union[int, tuple]) -> Tensor:
    """element-wise minimum along axis"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            expanded_grad = grad if axis is None else np.expand_dims(grad, axis)
            # Repeat along the specified axis to match the original shape
            expanded_grad = np.broadcast_to(expanded_grad, x.data.shape)
            # Create a mask where the minimum values are
            min_mask = x.data == np.min(x.data, axis=axis, keepdims=True)
            # Grad is only passed to the positions that had the minimum value
            new_grad = expanded_grad * min_mask
            return new_grad
        
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.min(x.data, axis=axis),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'min_{x.name}'
    )

def _clamp(x: Tensor, min: Union[int, float], max: Union[int, float]) -> Tensor:
    """clamp x between min and max"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * ((x.data >= min) & (x.data <= max))
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.clip(x.data, min, max),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'clamp_{x.name}'
    )

def _split(x: Tensor, indices_or_sections: Union[int, list[int]], axis: int) -> list[Tensor]:
    """split x into multiple tensors"""
    if isinstance(indices_or_sections, int):
        split_indices = np.array_split(range(x.shape[axis]), indices_or_sections)
        indices_or_sections = [len(split_indices[i]) for i in range(len(split_indices))]
        indices_or_sections = list(np.cumsum(indices_or_sections[:-1]))
        indices_or_sections.append(x.shape[axis])
    else:
        # indices_or_sections = [np.array(idx) for idx in indices_or_sections]
        indices_or_sections = indices_or_sections + [x.shape[axis]]
        
    data = np.split(x.data, indices_or_sections, axis=axis)
    tensors = []
    for idx, d in enumerate(data):
        depends_on = []
        if x.requires_grad:
            def grad_fn(grad: np.ndarray, idx: int=idx) -> np.ndarray:
                grad_split = np.zeros_like(x.data)
                if idx>0:
                    start_idx = indices_or_sections[idx-1]
                    end_idx = indices_or_sections[idx]
                else:
                    start_idx = 0
                    end_idx = indices_or_sections[idx]

                slices = [slice(None)] * len(x.shape)
                slices[axis] = slice(start_idx, end_idx)
                grad_split[tuple(slices)] = grad
                return grad_split
            depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
        tensors.append(Tensor(
            data=d,
            requires_grad=x.requires_grad,
            depends_on=depends_on,
            name=f'split_{x.name}_{idx}'
        ))
    return tensors

def _pad(x: Tensor, pad_width: tuple[tuple[int, int]], constant_values: Union[int, float]) -> Tensor:
    """pad x"""
    depends_on = []
    if x.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            slices = tuple(slice(pad_width[i][0], x.shape[i] + pad_width[i][0]) for i in range(x.data.ndim))
            return grad[slices]
        depends_on.append(Dependency(tensor=x, grad_fn=grad_fn))
    return Tensor(
        data=np.pad(x.data, pad_width=pad_width, mode='constant', constant_values=constant_values),
        requires_grad=x.requires_grad,
        depends_on=depends_on,
        name=f'pad_{x.name}'
    )