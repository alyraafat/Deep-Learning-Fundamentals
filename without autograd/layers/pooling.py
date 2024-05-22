from .layer import Layer
import numpy as np

def apply_pools(inp,pool_size,strides,mode):
    x = np.arange(start=0, stop=inp.shape[0],step=strides)
    y = np.arange(start=0, stop=inp.shape[1],step=strides)
    mask = []
    height = ((inp.shape[0]-pool_size)//strides) + 1
    width = ((inp.shape[1]-pool_size)//strides) + 1
    output = np.zeros(shape=(height,width))
    z = 0
    for i in x:
        k = 0
        for j in y:
            mat = inp[i:i+pool_size,j:j+pool_size]
            if mat.shape==(pool_size,pool_size):
                val = 0
                if mode=='max':
                    val = np.max(mat)
                    index = np.array(mat).argmax()
                    a,b = np.unravel_index(index,mat.shape)
                    mask.append((i+a,j+b))
                else:
                    val = np.average(mat)
                    mask.append((i,j))
                output[z,k] = val
            k+=1
        z+=1
    return output,mask


class PoolingLayer(Layer):
    def __init__(self,pool_size,strides,mode):
        super(PoolingLayer,self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.mode = mode
    
    def build(self, input_shape):
        pass
    
    def forward_propagation(self, inp):
        self.input = inp
        self.input_pool = np.reshape(inp,newshape=(inp.shape[-1],inp.shape[0],inp.shape[1]))
        self.masks = dict()
        height = ((inp.shape[0]-self.pool_size)//self.strides) + 1
        width = ((inp.shape[1]-self.pool_size)//self.strides) + 1
        output = np.zeros(shape=(inp.shape[-1],height,width))
        for i in range(self.input_pool.shape[0]):
            output_pool,mask=apply_pools(self.input_pool[i],self.pool_size,self.strides,self.mode)
            self.masks[i] = mask
            output[i] = output_pool
        self.output_pool = output
        self.output = np.reshape(output,newshape=(height,width,output.shape[0]))
        return self.output
    
    def backward_propagation(self, output_error):
        output_error = np.reshape(output_error,newshape=self.output_pool.shape)
        dX = np.zeros(shape=self.input_pool.shape)
        for i in range(self.input_pool.shape[0]):
            z=0
            k=0
            arr = self.masks[i]
            for x,y in arr:
                if k==output_error.shape[-1]:
                    k=0
                    z+=1
                if self.mode=='max':
                    dX[i,x,y] += output_error[i,z,k]
                else:
                    denominator = self.pool_size**2
                    dX[i,x:x+self.pool_size,y:y+self.pool_size] += output_error[i,z,k]/denominator
                k+=1
        dX = np.reshape(dX,newshape=self.input.shape)
        return dX

    def trainable_params(self):
        return None