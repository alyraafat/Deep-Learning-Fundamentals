from autograd.tensor import Dependency
from .tensor import Tensor
import numpy as np

class Parameter(Tensor):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Parameter either by specifying a shape or by passing a numpy array directly.
        
        Usage:
            Parameter(shape) - Initializes a Parameter with random values of the given shape.
            Parameter(data) - Initializes a Parameter with the given numpy array.
        """
        # print(f'args: {args}, kwargs: {kwargs}')

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            data = args[0]  # Initialize directly with numpy array
        elif all(isinstance(arg, int) for arg in args):  # Check if all arguments are integers
            shape = args
            data = np.random.randn(*shape) * np.sqrt(2. / sum(shape))  # Xavier initialization
        else:
            raise TypeError("Parameter initialization must be with a shape or a numpy array.")
        
        super().__init__(data=data, requires_grad=True, **kwargs)
