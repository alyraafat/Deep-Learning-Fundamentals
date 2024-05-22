from .layer import Layer
import numpy as np
class GlobalAveragePoolingLayer(Layer):
    def __init__(self):
        super(GlobalAveragePoolingLayer,self).__init__()

    def build(self, input_shape):
        pass
    
    def forward_propagation(self, inp):
        self.input=inp
        self.input_pool = np.reshape(inp,newshape=(inp.shape[-1],inp.shape[0],inp.shape[1]))
        output = np.zeros(shape=(inp.shape[-1],1))
        for i in range(inp.shape[-1]):
            output[i] = np.average(self.input_pool[i])
        self.output_pool = output
        self.output = np.reshape(output,newshape=(1,inp.shape[-1]))
        return self.output

    def backward_propagation(self, output_error):
        output_error = np.reshape(output_error,newshape=self.output_pool.shape)
        dX = np.zeros(shape=(self.input_pool.shape))
        denominator = self.input_pool.shape[1]*self.input_pool.shape[2]
        for i in range(self.input_pool.shape[0]):
            dX[i] += output_error[i]/denominator
        dX = np.reshape(dX,newshape=self.input.shape)
        return dX

    def trainable_params(self):
        return None