from .layer import Layer
from autograd import Tensor

class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def build(self, input_shape):
        # print(f'Flatten Layer setting curr layer')
        # super().set_parameter_curr_layer('flatten')
        pass
        
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input=inp
        self.output = inp.flatten().reshape(inp.shape[0],-1)
        return self.output

    def trainable_params(self):
        return None