from .layer import Layer
import numpy as np
from autograd import Tensor

class DropoutLayer(Layer):
    def __init__(self,drop_rate=0.2):
        self.drop_rate = drop_rate
        self.p = 1-self.drop_rate
    
    def build(self, input_shape):
        self.is_initialized = True
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.num_params = 0
    
    def forward_propagation(self, inp, training: bool=True):
        self.input = inp
        if not training:
            self.output = inp
            return self.output
        self.dropout = np.random.rand(*inp.shape)>self.drop_rate
        self.dropout /= self.p
        self.dropout = Tensor(self.dropout, requires_grad=False)
        self.output = self.input * self.dropout
        return self.output

    def trainable_params(self):
        return None