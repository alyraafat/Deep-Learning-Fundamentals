import numpy as np
from layers import Layer
from activations import tanh, tanh_prime, relu, relu_prime
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
            self.activation_prime = tanh_prime
        elif self.activation == 'relu':
            self.activation_fn = relu
            self.activation_prime = relu_prime
        

    def initialize_parameters(self,input_size):
        """
        Initializes the weights and biases of the RNN layer.

        Parameters:
        - input_size (int): The size of the input vector.
        """
        self.is_initialized = True
        if self.init == 'glorot_uniform':
            self.w_hh = glorot_uniform(self.hidden_size, self.hidden_size)
            self.w_hx = glorot_uniform(self.hidden_size,input_size)
            if self.bidirectional:
                self.w_hh_back = glorot_uniform(self.hidden_size, self.hidden_size)
                self.w_hx_back = glorot_uniform(self.hidden_size,input_size)
        else:
            self.w_hh = np.random.uniform(-1./self.hidden_size, 1./self.hidden_size, (self.hidden_size, self.hidden_size))
            self.w_hx = np.random.uniform(-1./input_size,1./input_size, (self.hidden_size, input_size))
            if self.bidirectional:
                self.w_hh_back = np.random.uniform(-1./self.hidden_size, 1./self.hidden_size, (self.hidden_size, self.hidden_size))
                self.w_hx_back = np.random.uniform(-1./input_size,1./input_size, (self.hidden_size, input_size))
                
        self.b_h = np.zeros((self.hidden_size,))
        if self.bidirectional:
            self.b_h_back = np.zeros((self.hidden_size,))
        
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
        
    def forward_propagation(self, inp):
        """
        Forward pass for a basic RNN layer over a batch of sequences of inputs.

        Parameters:
        - inp (numpy.ndarray): Input data with shape (batch_size, sequence_length, input_size),
                               where each column represents an input at a time step for each batch.

        Returns:
        - hidden_states: Hidden states for each time step for each batch.
        """

        # if not self.is_initialized:
        #     self.initialize_parameters(inp.shape[-1])
        # print(f'rnn input: {inp.shape}')
        
        self.input = inp
        self.forward_hidden_states = self.process_time_steps(self.input, False)
        
        if self.bidirectional:
            self.backward_hidden_states = self.process_time_steps(self.input[::-1, :], True)
        self.hidden_states = self.forward_hidden_states
        if not self.return_sequences:
            self.hidden_states = self.hidden_states[-1, :]
            if self.bidirectional:
                self.hidden_states = np.concatenate((self.hidden_states, self.backward_hidden_states[0, :]), axis=-1)
        else:
            self.hidden_states = np.concatenate((self.forward_hidden_states, self.backward_hidden_states), axis=-1)
        
        self.output = self.hidden_states
        return self.hidden_states

    def process_time_steps(self, inp, reverse=False):
            """
            Process the input sequence over time steps using the RNN layer.
            
            Args:
                inp (ndarray): Input sequence of shape (batch_size, sequence_length, input_size).
                reverse (bool, optional): Flag indicating whether to process the sequence in reverse order. 
                                          Defaults to False.
                                          
            Returns:
                ndarray: Hidden states of the RNN layer after processing the input sequence. 
                         If `return_sequences` is False, returns the hidden state at the last time step 
                         of shape (batch_size, hidden_size). 
                         If `return_sequences` is True, returns the hidden states for all time steps 
                         of shape (batch_size, sequence_length, hidden_size).
            """
            
            sequence_length, _ = inp.shape
            hidden_states = np.zeros((sequence_length + 1, self.hidden_size))
            if reverse:
                w_hx = self.w_hx_back
                w_hh = self.w_hh_back
                b_h = self.b_h_back
            else:
                w_hx = self.w_hx
                w_hh = self.w_hh
                b_h = self.b_h
            for i in range(sequence_length):
                curr_input = inp[i, :]
                j = i
                if reverse:
                    j = sequence_length - i
                curr_hidden_state = hidden_states[j, :] # (batch_size, hidden_size)
                h = self.activation_fn(np.dot(curr_input, w_hx.T) + np.dot(curr_hidden_state,w_hh) + b_h)
                if reverse:
                    hidden_states[j-1, :] = h
                else:
                    hidden_states[j+1, :] = h
                # y = np.dot(self.w_yh, h) + self.b_y
                # self.outputs[:, i, :] = y
            hidden_states = hidden_states[1:sequence_length+1, :] if not reverse else hidden_states[0:sequence_length, :]
            
            return hidden_states
    
    def backward_propagation(self, output_error):
        """
        Backward pass for a basic RNN layer over a batch of sequences of inputs.

        Parameters:
        - output_error (numpy.ndarray): Gradient of the loss with respect to the output of the RNN layer.

        Returns:
        - dX: Gradient of the loss with respect to the input of the RNN layer.
        - w_hx_grad: Gradient of the loss with respect to the input-to-hidden weights.
        - w_hh_grad: Gradient of the loss with respect to the hidden-to-hidden weights.
        - b_h_grad: Gradient of the loss with respect to the hidden bias.
        """
        output_error_forward = output_error[:, :self.hidden_size] if self.return_sequences else output_error[:self.hidden_size]
        output_error_backward = output_error[:, self.hidden_size:] if self.return_sequences else output_error[self.hidden_size:]
        dX, w_hx_grad, w_hh_grad, b_h_grad = self.backward_propagation_helper(
            output_error_forward, 
            self.input, 
            self.forward_hidden_states, 
            False, 
            self.w_hx, 
            self.w_hh, 
            self.b_h
        )
        grads = (dX, w_hx_grad, w_hh_grad, b_h_grad)
        if self.bidirectional:
            dX_back, w_hx_grad_back, w_hh_grad_back, b_h_grad_back = self.backward_propagation_helper(
                output_error_backward, 
                self.input[::-1, :], 
                self.backward_hidden_states, 
                True,
                self.w_hx_back, 
                self.w_hh_back, 
                self.b_h_back
            )
            dX += dX_back[::-1 , :]
            grads += (w_hx_grad_back, w_hh_grad_back, b_h_grad_back)

        return grads


    def backward_propagation_helper(self, output_error, inp, hidden_states, reverse, w_hx, w_hh, b_h):
        """
        Backward pass for a basic RNN layer over a batch of sequences of inputs.

        Parameters:
        - output_error (numpy.ndarray): Gradient of the loss with respect to the output of the RNN layer.
        - inp (numpy.ndarray): Input data with shape (sequence_length, input_size).
        - hidden_states (numpy.ndarray): Hidden states for each time step for each batch.
        - reverse (bool): Flag indicating whether to process the sequence in reverse order.
        - w_hx (numpy.ndarray): Input-to-hidden weights.
        - w_hh (numpy.ndarray): Hidden-to-hidden weights.
        - b_h (numpy.ndarray): Hidden bias.

        Returns:
        - dX: Gradient of the loss with respect to the input of the RNN layer.
        - w_hx_grad: Gradient of the loss with respect to the input-to-hidden weights.
        - w_hh_grad: Gradient of the loss with respect to the hidden-to-hidden weights.
        - b_h_grad: Gradient of the loss with respect to the hidden bias.
        """

        sequence_length, _ = inp.shape
        dX = np.zeros_like(inp)
        w_hx_grad = np.zeros_like(w_hx)
        w_hh_grad = np.zeros_like(w_hh)
        b_h_grad = np.zeros_like(b_h)
        
        # print(f'w_hx shape: {w_hx.shape}, w_hh shape: {w_hh.shape}, b_h shape: {b_h.shape}')
        # print(f'before::: w_hx_grad shape: {w_hx_grad.shape}, w_hh_grad shape: {w_hh_grad.shape}, b_h_grad shape: {b_h_grad.shape}, output_error shape: {output_error.shape}, dX shape: {dX.shape}')
        # equations
        # a_t = W_hx​ @ x_t ​+ W_hh @ ​h_t−1​ + b_h
        # h_t = activation(a_t)
        
        # If return_sequences is False, extend output_error across all timesteps
        if not self.return_sequences:
            # Initialize a full sequence length error array with zeros
            temp_output_error = np.zeros((sequence_length, self.hidden_size))
            # Assign output_error to the last timestep
            if reverse:
                temp_output_error[0, :] = output_error
            else:
                temp_output_error[-1, :] = output_error
            output_error = temp_output_error
        for i in range(sequence_length-1, -1, -1):
            dl_dht = output_error[i, :] # (hidden_size)
            dht_dat = self.activation_prime(hidden_states[i, :]) * output_error[i, :] # (hidden_size)
            dl_dat = dl_dht * dht_dat  # (hidden_size)

            # calculate w_hx_grad
            dat_dwhx = inp[i, :] # (input_size)
            # dht_dwhx = dl_dat.T @ dat_dwhx # (hidden_size, input_size)
            dht_dwhx = np.outer(dl_dat, dat_dwhx)  # (hidden_size, input_size)
            w_hx_grad += dht_dwhx # (hidden_size, input_size)

            # calculate b_h_grad
            dat_dbh = 1 # (1,)
            dht_dbh = dl_dat * dat_dbh # (hidden_size)
            # b_h_grad += np.sum(dht_dbh, axis=0)
            b_h_grad += dht_dbh

            # calculate dX
            dat_dxt = self.w_hx # (hidden_size, input_size)
            # print(f'dat_dxt shape: {dat_dxt.shape}, dl_dat shape: {dl_dat.shape}')
            
            dht_dxt = np.dot(dl_dat, dat_dxt) # (input_size)
            # print(dht_dxt)
            # print("Is NaN present in dht_dxt:", np.isnan(dht_dxt).any())
            # print("Is Inf present in dht_dxt:", np.isinf(dht_dxt).any())
            dX[i, :] = dht_dxt

            # calculate w_hh_grad
            if i > 0 and not reverse:
                prev_h = hidden_states[i-1, :]  # (hidden_size)
            elif i < sequence_length - 1 and reverse:
                prev_h = hidden_states[i+1, :]
            elif (i==0 and not reverse) or (i==sequence_length-1 and reverse):
                prev_h = np.zeros_like(hidden_states[0, :])  # Use zero for the initial state
            

            # Calculate the gradient for W_hh at this timestep using the outer product
            # dht_dwhh = dl_dat.T @ prev_h  # (hidden_size, hidden_size)
            dht_dwhh = np.outer(dl_dat, prev_h)  # (hidden_size, hidden_size)
            w_hh_grad += dht_dwhh  # (hidden_size, hidden_size)

        # print(f'after::: w_hx_grad shape: {w_hx_grad.shape}, w_hh_grad shape: {w_hh_grad.shape}, b_h_grad shape: {b_h_grad.shape}, dX shape: {dX.shape}')
        
        return dX, w_hx_grad, w_hh_grad, b_h_grad
    def trainable_params(self):
        """
        Returns the trainable parameters of the RNN layer.
        """
        params = (self.w_hx, self.w_hh, self.b_h)
        if self.bidirectional:
            params += (self.w_hx_back, self.w_hh_back, self.b_h_back)
        return params