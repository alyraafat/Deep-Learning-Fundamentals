from .layer import Layer
import numpy as np
from autograd import Tensor, Parameter
class GlobalAveragePoolingLayer(Layer):
    def __init__(self):
        super(GlobalAveragePoolingLayer,self).__init__()

    def build(self, input_shape):
        pass
    
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input=inp
        self.output = self.input.mean(axis=(1,2))
        self.output = self.output.reshape(self.output.shape[0],1,self.output.shape[1])
        return self.output

    def trainable_params(self):
        return None