from .layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def build(self, input_shape):
        pass
        
    def forward_propagation(self, inp):
        self.input=inp
        self.output = inp.flatten().reshape(1,-1)
        return self.output
    
    def backward_propagation(self, output_error):
        return output_error.reshape(self.input.shape)

    def trainable_params(self):
        return None