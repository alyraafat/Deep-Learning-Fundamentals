import numpy as np
from typing import List, Tuple, Union
from autograd import Tensor,Parameter
from .layer import Layer
import math 
class MaxPool2D(Layer):
    def __init__(self, pool_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]], padding: str = 'valid'):
        super(MaxPool2D, self).__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
    
    def build(self, input_shape: Tuple[int, int, int, int]):
        pass

    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input = inp
        batch_size, height, width, channels = inp.shape
        self.input_pool = inp.reshape(batch_size, channels, height, width)
        self.out_height, self.out_width = self.calculate_output_shape(inp.shape)
        output = Parameter(np.zeros(shape=(batch_size, channels, self.out_height, self.out_width)))
        for b in range(batch_size):
            for c in range(channels):
                output_pool = self.apply_pools(self.input_pool[b,c])
                output = output.set_item((b, c), output_pool)
        return output.reshape(batch_size, self.out_height, self.out_width, channels)

    def calculate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int]:
        _ , height, width, _ = input_shape
        if self.padding == 'same':
            out_height = ((height - 1) // self.strides[0]) + 1
            out_width = ((width - 1) // self.strides[1]) + 1
            pad_h = max((out_height - 1) * self.strides[0] + self.pool_size[0] - height, 0)
            pad_w = max((out_width - 1) * self.strides[1] + self.pool_size[1] - width, 0)
            self.pad_top = pad_h // 2
            self.pad_bottom = pad_h - self.pad_top
            self.pad_left = pad_w // 2
            self.pad_right = pad_w - self.pad_left
        else: # valid
            out_height = ((height-self.pool_size[0])//self.strides[0]) + 1
            out_width = ((width-self.pool_size[1])//self.strides[1]) + 1
        return out_height, out_width
    
    def apply_pools(self, inp: Tensor) -> Tensor:
        if self.padding == 'same':
            inp_padded = inp.pad(pad_width=( 
                                 (self.pad_top, self.pad_bottom), 
                                 (self.pad_left, self.pad_right), 
                                ),constant_values=0)
        else: # valid
            inp_padded = inp
        x = np.arange(start=0, stop=inp_padded.shape[0]-self.pool_size[0]+1,step=self.strides[0])
        y = np.arange(start=0, stop=inp_padded.shape[1]-self.pool_size[1]+1,step=self.strides[1])
        # mask = []
        # print(f'inp_padded: {inp_padded.shape}, out_height: {self.out_height}, out_width: {self.out_width}')
        output = Parameter(np.zeros(shape=(self.out_height,self.out_width)))
        z = 0
        for i in x:
            k = 0
            for j in y:
                end_h = i+self.pool_size[0]
                end_w = j+self.pool_size[1]
                # print(i,j,end_h,end_w)
                # print(inp)
                mat = inp[i:end_h, j:end_w]
                if mat.shape==self.pool_size:
                    val = mat.max()
                    output = output.set_item((z,k), val)
                k+=1
            z+=1
        return output