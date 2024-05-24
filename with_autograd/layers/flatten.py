from .layer import Layer
from autograd import Tensor
import numpy as np
class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def build(self, input_shape: tuple):
        # print(f'Flatten Layer setting curr layer')
        # super().set_parameter_curr_layer('flatten')
        self.input_shape = input_shape
        self.output_shape = (input_shape[0], np.prod(input_shape[1:]))
        self.num_params = 0
        self.is_initialized = True
        
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input=inp
        self.output = inp.flatten().reshape(inp.shape[0],-1)
        return self.output

    def trainable_params(self):
        return None