from .layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self,output_dim):
        super(FCLayer,self).__init__()
        self.output_dim=output_dim
    
    def initialize_parameters(self,input_dim):
        self.is_initialized=True
        self.input_dim=input_dim
        self.weights = np.random.randn(input_dim,self.output_dim) / np.sqrt(input_dim + self.output_dim)
        self.bias = np.random.randn(1,self.output_dim) / np.sqrt(input_dim + self.output_dim)
        self.num_params = np.prod(self.weights.shape) + np.prod(self.bias.shape)
    
    def build(self, input_shape):
        self.initialize_parameters(input_shape[-1])

    def forward_propagation(self, inp):
        self.input=inp
        # if not self.is_initialized:
        #   self.initialize_parameters(inp.shape[-1])
        self.output = inp @ self.weights + self.bias
        return self.output
    
    def backward_propagation(self, output_error):
        dW = self.input.T @ output_error
        db = output_error
        dX = output_error @ self.weights.T
        # update parameters
        # self.weights -= learning_rate*dW
        # self.bias -= learning_rate*db
        # print(f'FC layer output_error shape: {output_error.shape}, weights shape: {self.weights.shape}, bias shape: {self.bias.shape}')
        # print(f'FC layer dX shape: {dX.shape}, dW shape: {dW.shape}, db shape: {db.shape}')
        return dX, dW, db

    def trainable_params(self):
        return self.weights, self.bias
