from .layer import Layer
import numpy as np
from autograd import Tensor

class ActivationLayer(Layer):
    def __init__(self,activation):
        """
        Initialize the ActivationLayer with a specific activation function and its derivative.
        
        Parameters:
        - activation: Callable that computes the activation function.
        """
        super(ActivationLayer,self).__init__()
        self.activation = activation
        
    def build(self, input_shape):
        pass

    def forward_propagation(self, inp: Tensor, training=True) -> Tensor:
        """
        Perform the forward propagation through the activation function.
        
        Parameters:
        - inp: Input data (Tensor).
        
        Returns:
        - Output of the activation function.
        """
        self.input=inp
        self.output = self.activation(inp)
        return self.output

    def trainable_params(self):
        return None











