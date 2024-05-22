from .layer import Layer
import numpy as np
from scipy import signal
from autograd import Tensor, Parameter
from typing import Union

def cross_correlation(inp: Tensor, kernel: Tensor, padding: str='valid'):
    x=inp
    kernel_size = kernel.shape[0]
    new_height = x.shape[0]-kernel_size+1
    new_width = x.shape[1]-kernel_size+1
    if padding=='full':
        x = Parameter(np.zeros(shape=(inp.shape[0]+2*(kernel_size-1), inp.shape[1]+2*(kernel_size-1))))
        x = fill_array(x,inp)
        # print(x)
        new_height = x.shape[0]-kernel_size+1
        new_width = x.shape[1]-kernel_size+1
    elif padding=='same':
        p = kernel_size-1
        x = Parameter(np.zeros(shape=(inp.shape[0]+p, inp.shape[1]+p)))
        start = p//2
        end = -p//2 if p>0 else x.shape[0]
        # x[start:end,start:end] = inp
        x = x.set_item((slice(start,end),slice(start,end)),inp)
        new_height,new_width = inp.shape
    
    output = Parameter(np.zeros(shape=(new_height,new_width)))
    for i in range(new_height):
        for j in range(new_width):
            mat = x[i:i+kernel_size,j:j+kernel_size]
            # output[i,j] = np.sum(mat*kernel)
            convolve = (mat*kernel).sum()
            output = output.set_item((i,j),convolve)
    return output

def fill_array(x: Tensor,y: Tensor) -> Tensor:
    start_x = int(np.ceil((x.shape[0]-y.shape[0])/2))
    start_y = int(np.ceil((x.shape[1]-y.shape[1])/2))
    # x[start_x:start_x+y.shape[0],start_y:start_y+y.shape[1]]=y
    x_1 = x.set_item((slice(start_x,start_x+y.shape[0]),slice(start_y,start_y+y.shape[1])),y)
    return x_1

class Conv2D(Layer):
    def __init__(self, kernel_size: Union[int, tuple], filters: int, padding: str='valid'):
        super(Conv2D,self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.filters = filters
        self.padding = padding
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape=input_shape
        batch_size,input_height,input_width, input_depth = input_shape
        self.input_depth = input_depth
        if self.padding == 'same':
            output_height = input_height
            output_width = input_width
        elif self.padding == 'full':
            output_height = input_height + self.kernel_size[0] - 1
            output_width = input_width + self.kernel_size[1] - 1
        else:  # 'valid'
            output_height = input_height - self.kernel_size[0] + 1
            output_width = input_width - self.kernel_size[1] + 1

        self.output_shape = (self.filters,output_height,output_width)
        self.kernels_shape = (self.filters,self.input_depth,self.kernel_size[0],self.kernel_size[1])
        self.kernels = Parameter(np.random.randn(*self.kernels_shape))
        self.bias = Parameter(np.random.randn(*self.output_shape))
        self.kernels_shape_for_summary = (self.filters,self.kernel_size[0],self.kernel_size[1],self.input_depth)
        # self.input_shape_for_summary = (input_height,input_width,input_depth)
        # self.output_shape_for_summary = (self.output_shape[1],self.output_shape[2],self.output_shape[0])
        self.num_params = np.prod(self.kernels.shape) + np.prod(self.bias.shape)
    
    def build(self, input_shape: tuple):
        """
        Build the layer with the given input shape
        """
        self.initialize_parameters(input_shape)

    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input=inp
        batch_size, input_height, input_width, input_channels = inp.shape
        self.input_conv = inp.reshape(batch_size, input_channels,input_height,input_width)
        self.output_conv = Parameter(np.zeros(shape=(batch_size, *self.output_shape)))
        for f in range(self.filters):
            for b in range(batch_size):
                for c in range(input_channels):
                    cross_corr_out = cross_correlation(self.input_conv[b,c], self.kernels[f,c], self.padding)
                    intermediate_res = self.output_conv[b,f] + cross_corr_out
                    self.output_conv = self.output_conv.set_item((b,f),intermediate_res)
                    # self.output_conv[b,f] += cross_corr_out
        self.output_conv = self.output_conv + self.bias
        self.output = self.output_conv.reshape(batch_size, self.output_conv.shape[2], self.output_conv.shape[3], self.output_conv.shape[1])
        return self.output

    def trainable_params(self):
        return self.kernels, self.bias