import numpy as np
from autograd import Tensor, Parameter
from .layer import Layer
from initializations import zeros, ones

class BatchNorm2D(Layer):
    def __init__(self, eps: float = 1e-5, momentum: float = 0.1, name: str = None, track_running_stats: bool=True, affine: bool=True):
        super(BatchNorm2D, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.name = name
        self.track_running_stats = track_running_stats
        self.affine = affine
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        channels = input_shape[-1]
        self.num_features = channels
        self.num_params = 0
        if self.affine:
            self.gamma = Parameter(ones(shape=(self.num_features,)))
            self.beta = Parameter(zeros(shape=(self.num_features,)))
            self.num_params = 2 * self.num_features
        if self.track_running_stats:
            self.running_mean = zeros(shape=(self.num_features,))
            self.running_var = ones(shape=(self.num_features,))
        self.output_shape = input_shape
    
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

        if training:
            batch_mean = inp.mean(axis=(0,1,2), keepdims=True)
            batch_var_biased = inp.var(axis=(0,1,2), ddof=0, keepdims=True)
            batch_var_unbiased = inp.var(axis=(0,1,2), ddof=1, keepdims=True)

            if self.track_running_stats:
                self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * batch_var_unbiased + (1 - self.momentum) * self.running_var
            
            normed_inp = (inp - batch_mean) / (batch_var_biased + self.eps).sqrt()
        else:
            if self.track_running_stats:
                normed_inp = (inp - self.running_mean) / (self.running_var + self.eps).sqrt()
            else:
                batch_mean = inp.mean(axis=(0, 1, 2), keepdims=True)
                batch_var_biased = inp.var(axis=(0, 1, 2), ddof=0, keepdims=True)
                normed_inp = (inp - batch_mean) / (batch_var_biased + self.eps).sqrt()
        
        if self.affine:
            self.output = self.gamma * normed_inp + self.beta
        else:
            self.output = normed_inp

        return self.output