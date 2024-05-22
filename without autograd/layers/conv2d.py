from .layer import Layer
import numpy as np
from scipy import signal

def cross_correlation(inp,kernel,padding='valid'):
    x=inp
    kernel_size = kernel.shape[0]
    new_height = x.shape[0]-kernel_size+1
    new_width = x.shape[1]-kernel_size+1
    if padding=='full':
        x = np.zeros(shape=(inp.shape[0]+2*(kernel_size-1),inp.shape[1]+2*(kernel_size-1)))
        x = fill_array(x,inp)
        # print(x)
        new_height = x.shape[0]-kernel_size+1
        new_width = x.shape[1]-kernel_size+1
    elif padding=='same':
        p = kernel_size-1
        x = np.zeros(shape=(inp.shape[0]+p,inp.shape[1]+p))
        start = p//2
        end = -p//2 if p>0 else x.shape[0]
        x[start:end,start:end] = inp
        new_height,new_width=inp.shape
    
    output = np.zeros(shape=(new_height,new_width))
    for i in range(new_height):
        for j in range(new_width):
            mat = x[i:i+kernel_size,j:j+kernel_size]
            output[i,j] = np.sum(mat*kernel)
    return output

def fill_array(x,y):
    start_x = int(np.ceil((x.shape[0]-y.shape[0])/2))
    start_y = int(np.ceil((x.shape[1]-y.shape[1])/2))
    x[start_x:start_x+y.shape[0],start_y:start_y+y.shape[1]]=y
    return x

class Conv2D(Layer):
    def __init__(self,kernel_size,filters):
        super(Conv2D,self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
    
    def initialize_parameters(self,input_shape):
        self.is_initialized = True
        self.input_shape=input_shape
        input_depth,input_height,input_width = input_shape
        self.input_depth = input_depth
        self.output_shape = (self.filters,input_height-self.kernel_size+1,input_width-self.kernel_size+1)
        self.kernels_shape = (self.filters,self.input_depth,self.kernel_size,self.kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.bias = np.random.randn(*self.output_shape)
        self.kernels_shape_for_summary = (self.filters,self.kernel_size,self.kernel_size,self.input_depth)
        # self.input_shape_for_summary = (input_height,input_width,input_depth)
        # self.output_shape_for_summary = (self.output_shape[1],self.output_shape[2],self.output_shape[0])
        self.num_params = np.prod(self.kernels.shape) + np.prod(self.bias.shape)
    
    def build(self, input_shape):
        """
        Build the layer with the given input shape
        """
        self.initialize_parameters(input_shape)

    def forward_propagation(self, inp):
        self.input=inp
        self.input_conv = np.reshape(inp,newshape=(inp.shape[-1],inp.shape[0],inp.shape[1]))
        # if not self.is_initialized:
        #   self.initialize_parameters(self.input_conv.shape)
        self.output_conv = np.copy(self.bias)
        for i in range(self.filters):
          for j in range(self.input_depth):
            # you can use cross_correlation method implemented above instead of signal.correlate2d, however it is a little bit slower
            self.output_conv[i] += signal.correlate2d(self.input_conv[j],self.kernels[i,j],'valid')
        self.output = np.reshape(self.output_conv,newshape=(self.output_conv.shape[1],self.output_conv.shape[2],self.output_conv.shape[0]))
        return self.output
    
    def backward_propagation(self, output_error):
        output_error = np.reshape(output_error,newshape=self.output_shape)
        db = output_error
        dK = np.zeros(shape=self.kernels_shape)
        dX = np.zeros(shape=self.input_shape)
        for i in range(self.filters):
          for j in range(self.input_depth):
            # you can use cross_correlation method implemented above instead of signal.correlate2d, however it is a little bit slower
            dK[i,j] += signal.correlate2d(self.input_conv[j],output_error[i],'valid')
            dX[j] += signal.convolve2d(output_error[i],self.kernels[i,j],'full')
        # self.kernels -= learning_rate*dK
        # self.bias -= learning_rate*db
        dX = np.reshape(dX,newshape=self.input.shape)
        return dX, dK, db

    def trainable_params(self):
        return self.kernels, self.bias