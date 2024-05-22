import pandas as pd
import numpy as np
from layers import FCLayer, Conv2D


class Network:
    def __init__(self):
        self.layers = []
    
    def add(self,layer):
        self.layers.append(layer)
    
    def use(self,loss,loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    def build(self,shape):
        curr_shape = shape
        for layer in self.layers:
            aux_input = np.zeros(shape=curr_shape)
            layer.build(curr_shape)
            output = layer.forward_propagation(aux_input)
            curr_shape = output.shape

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_learning_rate_scheduler(self, scheduler):
        self.learning_rate_scheduler = scheduler
    
    def get_trainable_params(self):
        params = []
        for layer in self.layers:
            curr_params = layer.trainable_params()
            if curr_params is not None:
                params += curr_params
        return params
            
    def summary(self):
        """
        Returns a pandas dataframe with the summary of the network
        """
        summary_list = []
        total_params = 0
        for layer in self.layers:
            layer_dict = {}
            layer_dict['type'] = type(layer).__name__
            layer_dict['input_shape'] = layer.input.shape  # (height,width,depth if image)
            layer_dict['output_shape'] = layer.output.shape # (height,width,depth if image)
            layer_dict['fc_layer_shape'] = (layer.input_dim,layer.output_dim) if type(layer)==FCLayer else None
            layer_dict['kernels_shape'] =  layer.kernels_shape_for_summary if type(layer)==Conv2D else None #(filters,kernel_size,kernel_size,depth)
            # if type(layer)==FCLayer:
            #     layer_dict['number_of_params'] = np.prod(layer_dict['fc_layer_shape'])+np.prod(layer.bias.shape)
            # elif type(layer)==Conv2D:
            #     layer_dict['number_of_params'] = np.prod(layer_dict['kernels_shape'])+np.prod(layer.bias.shape)
            # else:
            #     layer_dict['number_of_params'] = 0
            layer_dict['number_of_params'] = layer.num_params
            total_params+=layer_dict['number_of_params']
            summary_list.append(layer_dict)

        total_row = {
            'type': 'Total number of params',
            'input_shape': '',
            'output_shape': '',
            'fc_layer_shape': '',
            'kernels_shape': '',
            'number_of_params': total_params
        }
        summary_list.append(total_row)
        df = pd.DataFrame(summary_list)
        # print(df)
        return df

    def predict(self,input_data):
        result = []
        for x in input_data:
            # print(np.array(x).shape)
            output=x
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result


    def fit(self,x_train,y_train,epochs):
        """
        
        Parameters:
        -x_train: list of input data
        -y_train: list of output data
        -epochs: number of epochs

        """

        self.build(x_train[0].shape)

        for epoch in range(epochs):
            err = 0
            if hasattr(self,'learning_rate_scheduler'):
                self.learning_rate_scheduler(epoch=epoch, optimizer=self.optimizer)
            for i,x in enumerate(x_train):
                y_pred = x
                for layer in self.layers:
                    y_pred = layer.forward_propagation(y_pred)
                loss = self.loss(y_train[i],y_pred)
                err+=loss
                dY = self.loss_prime(y_train[i],y_pred) # dE/dY
                output_error=dY
                ## new code
                grads = []
                all_params = []
                for layer in reversed(self.layers):
                    curr_grads = layer.backward_propagation(output_error)
                    curr_params = layer.trainable_params()
                    if curr_params is not None:
                        all_params.extend(curr_params)
                    if isinstance(curr_grads, tuple) and len(curr_grads)>1:
                        grads.extend(curr_grads[1:])
                        output_error = curr_grads[0]
                    else:
                        output_error = curr_grads
                self.optimizer.update(all_params,grads)
                
            err/=len(x_train)
            print(f'epoch {epoch+1}: loss={err}')


        