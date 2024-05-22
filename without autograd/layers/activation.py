from .layer import Layer
import numpy as np
from activations import softmax_prime

class ActivationLayer(Layer):
    def __init__(self,activation,activation_prime):
        """
        Initialize the ActivationLayer with a specific activation function and its derivative.
        
        Parameters:
        - activation: Callable that computes the activation function.
        - activation_prime: Callable that computes the derivative of the activation function.
        """
        super(ActivationLayer,self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def build(self, input_shape):
        pass

    def forward_propagation(self, inp):
        """
        Perform the forward propagation through the activation function.
        
        Parameters:
        - inp: Input data (numpy array).
        
        Returns:
        - Output of the activation function.
        """
        self.input=inp
        self.output = self.activation(inp)
        return self.output
    
    def backward_propagation(self, output_error):
        """
        Perform the backward propagation through the derivative of the activation function.
        
        Parameters:
        - output_error: Gradient of the loss with respect to the output of this layer.
        
        Returns:
        - Gradient of the loss with respect to the input of this layer.
        """
        jacobian_matrix = self.activation_prime(self.input)
        if self.activation_prime != softmax_prime:
            return output_error * jacobian_matrix
        else:
            return output_error @ jacobian_matrix

    def trainable_params(self):
        return None




class SoftmaxLayer(Layer):

    def forward_propagation(self,inp):
        self.input=inp
        e_x = np.exp(inp-np.max(inp,axis=-1,keepdims=True))
        self.output = e_x/np.sum(e_x,axis=-1,keepdims=True)
        return self.output
        
    def backward_propagation(self, output_error):
        # n = self.output.shape[-1]
        # dX = np.empty((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         if i==j:
        #             dX[i,j] = self.output[i]*(1-self.output[j])
        #         else:
        #             dX[i,j] = -self.output[i]*self.output[j]
        # Vectorized version
        s = self.output.reshape(-1, 1)
        dX = np.diagflat(s) - np.dot(s, s.T)
        dX = output_error @ dX
        return dX

    def trainable_params(self):
        return None






