import numpy as np
from layers import Layer
from autograd import tanh, relu, Tensor, Parameter
from initializations import glorot_uniform

class RNN(Layer):
    def __init__(
            self, 
            hidden_size, 
            output_size, 
            return_sequences=False, 
            bidirectional=False, 
            init='glorot_uniform',
            activation='tanh'
        ):
        """
        Initializes the RNN layer with the given input, hidden, and output sizes.

        Parameters:
        - input_size (int): The size of the input vector.
        - hidden_size (int): The size of the hidden state vector.
        - output_size (int): The size of the output vector.
        - return_sequences (bool, optional): Flag indicating whether to return the hidden states for all time steps.
                                             Defaults to False.
        - bidirectional (bool, optional): Flag indicating whether to use a bidirectional RNN. Defaults to False.
        - init (str, optional): The weight initialization strategy. Defaults to 'glorot_uniform'.
        - activation (str, optional): The activation function to use. Defaults to 'tanh'.
        """
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.is_initialized = False
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.init = init
        self.activation = activation
        if self.activation == 'tanh':
            self.activation_fn = tanh
        elif self.activation == 'relu':
            self.activation_fn = relu
        

    def initialize_parameters(self,input_size):
        """
        Initializes the weights and biases of the RNN layer.

        Parameters:
        - input_size (int): The size of the input vector.
        """
        self.is_initialized = True
        if self.init == 'glorot_uniform':
            self.w_hh = Parameter(glorot_uniform(self.hidden_size, self.hidden_size), name='w_hh')
            self.w_hx = Parameter(glorot_uniform(self.hidden_size,input_size), name='w_hx')
            if self.bidirectional:
                self.w_hh_back = Parameter(glorot_uniform(self.hidden_size, self.hidden_size), name='w_hh_back')
                self.w_hx_back = Parameter(glorot_uniform(self.hidden_size,input_size), name='w_hx_back')
        else:
            self.w_hh = Parameter(np.random.uniform(-1./self.hidden_size, 1./self.hidden_size, (self.hidden_size, self.hidden_size)), name='w_hh')
            self.w_hx = Parameter(np.random.uniform(-1./input_size,1./input_size, (self.hidden_size, input_size)), name='w_hx')
            if self.bidirectional:
                self.w_hh_back = Parameter(np.random.uniform(-1./self.hidden_size, 1./self.hidden_size, (self.hidden_size, self.hidden_size)), name='w_hh_back')
                self.w_hx_back = Parameter(np.random.uniform(-1./input_size,1./input_size, (self.hidden_size, input_size)), name='w_hx_back')
                
        self.b_h = Parameter(np.zeros((self.hidden_size,)), name='b_h')
        if self.bidirectional:
            self.b_h_back = Parameter(np.zeros((self.hidden_size,)), name='b_h_back')
        
        self.num_params = self.hidden_size * self.hidden_size + self.hidden_size * input_size + self.hidden_size
        if self.bidirectional:
            self.num_params *= 2

    def build(self, input_shape):
        """
        Builds the RNN layer with the given input shape.

        Parameters:
        - input_shape (tuple): The shape of the input data.
        """
        self.initialize_parameters(input_shape[-1])
        # print(f'RNN setting curr layer')
        super().set_parameter_curr_layer(curr_layer='rnn')
        
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        """
        Forward pass for a basic RNN layer over a batch of sequences of inputs.

        Parameters:
        - inp (Tensor): Input data with shape (batch_size, sequence_length, input_size),
                               where each column represents an input at a time step for each batch.

        Returns:
        - hidden_states (Tensor): Hidden states for each time step for each batch.
        """

        # if not self.is_initialized:
        #     self.initialize_parameters(inp.shape[-1])
        # print(f'rnn input: {inp.shape}')
        
        self.input = inp
        self.forward_hidden_states = self.process_time_steps(self.input, False)
        
        if self.bidirectional:
            rev_input = self.input[:, ::-1, :]
            self.backward_hidden_states = self.process_time_steps(rev_input, True)
        self.hidden_states = self.forward_hidden_states
        if not self.return_sequences:
            self.last_forward_hidden_state = self.hidden_states[:, -1, :]
            if self.bidirectional:
                # self.hidden_states = np.concatenate((self.hidden_states, self.backward_hidden_states[0, :]), axis=-1)
                self.last_backward_hidden_state = self.backward_hidden_states[:, 0, :]
                self.hidden_states = Tensor.concatenate(tensors=[self.last_forward_hidden_state,self.last_backward_hidden_state],axis=-1)
        else:
            # self.hidden_states = np.concatenate((self.forward_hidden_states, self.backward_hidden_states), axis=-1)
            if self.bidirectional:
                self.hidden_states = Tensor.concatenate(tensors=[self.forward_hidden_states,self.backward_hidden_states],axis=-1)
        
        return self.hidden_states

    def process_time_steps(self, inp: Tensor, reverse=False) -> Tensor:
            """
            Process the input sequence over time steps using the RNN layer.
            
            Args:
                inp (Tensor): Input sequence of shape (batch_size, sequence_length, input_size).
                reverse (bool, optional): Flag indicating whether to process the sequence in reverse order. 
                                          Defaults to False.
                                          
            Returns:
                ndarray: Hidden states of the RNN layer after processing the input sequence. 
                         If `return_sequences` is False, returns the hidden state at the last time step 
                         of shape (batch_size, hidden_size). 
                         If `return_sequences` is True, returns the hidden states for all time steps 
                         of shape (batch_size, sequence_length, hidden_size).
            """
            # print(f'inp shape rnn: {inp.shape}')
            batch_size, sequence_length, _ = inp.shape
            hidden_states = Parameter(np.zeros((batch_size, sequence_length + 1, self.hidden_size)), name='hidden_states_forward' if not reverse else 'hidden_states_back')
            if reverse:
                w_hx = self.w_hx_back
                w_hh = self.w_hh_back
                b_h = self.b_h_back
            else:
                w_hx = self.w_hx
                w_hh = self.w_hh
                b_h = self.b_h
            for i in range(sequence_length):
                curr_input = inp[: ,i, :]
                j = i
                if reverse:
                    j = sequence_length - i
                curr_hidden_state = hidden_states[: ,j, :] # (batch_size, hidden_size)
                # h = self.activation_fn(np.dot(curr_input, w_hx.T) + np.dot(curr_hidden_state,w_hh) + b_h)
                # print(f'curr_input shape: {curr_input.shape}, w_hx shape: {w_hx.shape}')
                # print(f'curr_input: {curr_input}')
                # a = curr_input @ w_hx.T
                # print(a.shape)
   
                # b = curr_hidden_state @ w_hh
                # c = a + b + b_h
                h = self.activation_fn(curr_input @ w_hx.T + curr_hidden_state @ w_hh + b_h)
                if reverse:
                    # hidden_states[: ,j-1, :] = h
                    hidden_states = hidden_states.set_item((slice(None),j-1,slice(None)),h)
                else:
                    # hidden_states[: ,j+1, :] = h
                    hidden_states = hidden_states.set_item((slice(None),j+1,slice(None)),h)
                # y = np.dot(self.w_yh, h) + self.b_y
                # self.outputs[:, i, :] = y
            hidden_states_modified = hidden_states[:, 1:sequence_length+1, :] if not reverse else hidden_states[: ,0:sequence_length, :]
            
            return hidden_states_modified
    
    def trainable_params(self):
        """
        Returns the trainable parameters of the RNN layer.
        """
        params = (self.w_hx, self.w_hh, self.b_h)
        if self.bidirectional:
            params += (self.w_hx_back, self.w_hh_back, self.b_h_back)
        return params