import numpy as np
from autograd import Tensor, Parameter
from .layer import Layer
from initializations import zeros, ones

class LayerNorm(Layer):
    def __init__(self, D: int, eps: float = 1e-5, name: str = None, bias: bool=True, elementwise_affine: bool=True):
        super(LayerNorm, self).__init__()
        self.D = D
        self.eps = eps
        self.name = name
        self.bias = bias
        self.elementwise_affine = elementwise_affine
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        self.normalized_shape = input_shape[-self.D:]
        if self.elementwise_affine:
            self.gamma = Parameter(ones(shape=self.normalized_shape))
            if self.bias:
                self.beta = Parameter(zeros(shape=self.normalized_shape))
        
    def build(self, input_shape: tuple):
        self.input_shape = input_shape
        self.initialize_parameters(input_shape)
    
    def forward_propagation(self, inp: Tensor, training: bool = True) -> Tensor:
        """
        Parameters:
        inp: Tensor
            The input tensor to the layer (batch size, height, width, channels)
        training: bool
        """
        self.input = inp
        axis = tuple(range(-self.D, 0))
        layer_mean = inp.mean(axis=axis, keepdims=True)
        layer_var_biased = inp.var(axis=axis, ddof=0, keepdims=True)
        
        normed_inp = (inp - layer_mean) / (layer_var_biased + self.eps).sqrt()

        if self.elementwise_affine:
            self.output = self.gamma * normed_inp
            if self.bias:
                self.output = self.output + self.beta
        else:
            self.output = normed_inp

        return self.output