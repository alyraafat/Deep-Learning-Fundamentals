from .layer import Layer
import numpy as np
from scipy import signal
from autograd import Tensor, Parameter
from typing import Union
import math


class Conv2D(Layer):
    def __init__(self, kernel_size: Union[int, tuple], filters: int, padding: str='valid', strides: Union[int, tuple]=(1,1)):
        super(Conv2D,self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.filters = filters
        self.padding = padding
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape=input_shape
        batch_size,input_height,input_width, input_depth = input_shape
        self.input_depth = input_depth
        # if self.padding == 'same':
        #     output_height = math.ceil(input_height / self.strides[0])
        #     output_width = math.ceil(input_width / self.strides[1])
        # elif self.padding == 'full':
        #     output_height = input_height + self.kernel_size[0] - 1
        #     output_width = input_width + self.kernel_size[1] - 1
        # else:  # 'valid'
        #     output_height = math.floor((input_height - self.kernel_size[0]) / self.strides[0] + 1)
        #     output_width = math.floor((input_width - self.kernel_size[1]) / self.strides[1] + 1)
        if self.padding == 'same':
            output_height = math.ceil(input_height / self.strides[0])
            output_width = math.ceil(input_width / self.strides[1])
            pad_h = max((output_height - 1) * self.strides[0] + self.kernel_size[0] - input_height, 0)
            pad_w = max((output_width - 1) * self.strides[1] + self.kernel_size[1] - input_width, 0)
            self.pad_top = pad_h // 2
            self.pad_bottom = pad_h - self.pad_top
            self.pad_left = pad_w // 2
            self.pad_right = pad_w - self.pad_left
        elif self.padding == 'full':
            output_height = input_height + self.kernel_size[0] - 1
            output_width = input_width + self.kernel_size[1] - 1
        else:  # 'valid'
            output_height = math.floor((input_height - self.kernel_size[0]) / self.strides[0] + 1) 
            output_width = math.floor((input_width - self.kernel_size[1]) / self.strides[1] + 1)

        self.output_shape = (self.filters,output_height,output_width)
        self.kernels_shape = (self.filters,self.kernel_size[0],self.kernel_size[1])
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
                    cross_corr_out = self.cross_correlation(self.input_conv[b,c], self.kernels[f])
                    intermediate_res = self.output_conv[b,f] + cross_corr_out
                    self.output_conv = self.output_conv.set_item((b,f),intermediate_res)
                    # self.output_conv[b,f] += cross_corr_out
        self.output_conv = self.output_conv + self.bias
        self.output = self.output_conv.reshape(batch_size, self.output_conv.shape[2], self.output_conv.shape[3], self.output_conv.shape[1])
        return self.output

    def trainable_params(self):
        return self.kernels, self.bias
    
    def cross_correlation(self, inp: Tensor, kernel: Tensor):
        kernel_size_h, kernel_size_w = kernel.shape
        new_height, new_width = self.output_shape[1], self.output_shape[2]
        # print(inp.shape)
        if self.padding == 'same':
            inp_padded = inp.pad(pad_width=( 
                                 (self.pad_top, self.pad_bottom), 
                                 (self.pad_left, self.pad_right), 
                                ),constant_values=0)
        elif self.padding == 'full':
            inp_padded = inp.pad(pad_width=( 
                                 (self.kernel_size[0] - 1, self.kernel_size[0] - 1), 
                                 (self.kernel_size[1] - 1, self.kernel_size[1] - 1), 
                                ),constant_values=0)
        else:
            inp_padded = inp
        # print(f'inp_padded: {inp_padded.shape}, new_height: {new_height}, new_width: {new_width}')
        output = Parameter(np.zeros(shape=(new_height,new_width)))
        x = np.arange(start=0, stop=inp_padded.shape[0]-kernel_size_h+1,step=self.strides[0])
        y = np.arange(start=0, stop=inp_padded.shape[1]-kernel_size_w+1,step=self.strides[1])
        h = 0
        for i in x:
            w = 0
            for j in y:
                mat = inp_padded[i:i+kernel_size_h,j:j+kernel_size_w]
                # print(f'mat : {mat.shape}, kernel: {kernel.shape}')
                # if mat.shape==kernel.shape:
                convolve = (mat*kernel).sum()
                output = output.set_item((h,w),convolve)
                w+=1
            h+=1
        return output

    def fill_array(self, x: Tensor,y: Tensor) -> Tensor:
        start_x = int(np.ceil((x.shape[0]-y.shape[0])/2))
        start_y = int(np.ceil((x.shape[1]-y.shape[1])/2))
        # x[start_x:start_x+y.shape[0],start_y:start_y+y.shape[1]]=y
        x_1 = x.set_item((slice(start_x,start_x+y.shape[0]),slice(start_y,start_y+y.shape[1])),y)
        return x_1
