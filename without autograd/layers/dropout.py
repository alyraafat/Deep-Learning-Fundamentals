from .layer import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self,drop_rate=0.2):
        self.drop_rate = drop_rate
    
    def build(self, input_shape):
        pass
    
    def forward_propagation(self, inp):
        self.input = inp
        self.dropout = np.random.randn(*inp.shape)>self.drop_rate
        self.output = np.copy(self.input)
        self.output *= self.dropout
        self.output /= (1-self.drop_rate)
        return self.output
    
    def backward_propagation(self, output_error):
        dX = output_error*self.dropout
        dX /= (1-self.drop_rate)
        return dX

    def trainable_params(self):
        return None