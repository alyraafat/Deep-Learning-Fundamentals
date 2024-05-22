# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.is_initialized = False
        self.num_params = 0
        
    # initializes the weights and biases of the layer
    def build(self, input_shape):
        raise NotImplementedError
    
    # computes the output Y of a layer for a given input X
    def forward_propagation(self, inp):
        raise NotImplementedError
    
    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error):
        raise NotImplementedError

    # return trainable params for grad calculation
    def trainable_params(self):
        raise NotImplementedError