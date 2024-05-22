from .layer import Layer
import numpy as np
from autograd import Tensor,Parameter

class FCLayer(Layer):
    def __init__(self,output_dim: int, use_bias: bool=True):
        super(FCLayer,self).__init__()
        self.output_dim=output_dim
        self.use_bias=use_bias
    
    def initialize_parameters(self,input_shape: tuple):
        self.is_initialized=True
        batch_size = input_shape[0]
        input_dim = input_shape[-1]
        self.input_dim=input_dim
        self.weights = Parameter(np.random.randn(input_dim,self.output_dim) / np.sqrt(input_dim + self.output_dim), name='weights')
        self.bias = Parameter(np.random.randn(1,self.output_dim) / np.sqrt(input_dim + self.output_dim), name='bias')
        self.num_params = np.prod(self.weights.shape) + np.prod(self.bias.shape)
    
    def build(self, input_shape: tuple):
        self.initialize_parameters(input_shape)
        # print(f'FC Layer setting curr layer')
        super().set_parameter_curr_layer('dense')

    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input=inp
        self.output = inp @ self.weights
        if self.use_bias:
            self.output = self.output + self.bias
        # print(f'inp: {inp.shape}, weights: {self.weights.shape}, bias: {self.bias.shape}, self.output: {self.output.shape}')
        return self.output

    def trainable_params(self):
        return self.weights, self.bias
