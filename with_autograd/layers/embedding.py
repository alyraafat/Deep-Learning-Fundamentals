import numpy as np
from autograd import Tensor, Parameter
from .layer import Layer
from initializations import gaussian, zeros

class Embedding(Layer):
    def __init__(self, embedding_dim: int, vocab_size: int, name: str = None):
        super(Embedding,self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.name = name
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        self.num_params = self.vocab_size * self.embedding_dim
        self.weights = Parameter(gaussian(shape=(self.vocab_size, self.embedding_dim)))
        self.output_shape = input_shape + (self.embedding_dim,)
    
    def build(self, input_shape: tuple):
        self.initialize_parameters(input_shape)

    def forward_propagation(self, inp: Tensor, training: bool = True) -> Tensor:
        """
        Parameters:
        inp: Tensor
            The input tensor to the layer (batch size, sequence length)
        training: bool
        """
        self.input = inp
        self.output = self.weights[inp.data]
        return self.output
        