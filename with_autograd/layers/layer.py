from autograd import Tensor, Module
# Base class
class Layer(Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.input = None
        self.output = None
        self.is_initialized = False
        self.num_params = 0
        
    # initializes the weights and biases of the layer
    def build(self, input_shape: tuple):
        raise NotImplementedError
    
    # computes the output Y of a layer for a given input X
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        raise NotImplementedError

    def __call__(self, inp: Tensor, training: bool=True) -> Tensor:
        if not self.is_initialized:
            self.build(inp.shape)
        return self.forward_propagation(inp, training=training)
    
    # return trainable params for grad calculation
    def trainable_params(self):
        raise NotImplementedError