import numpy as np
from autograd import Tensor, Parameter, sigmoid, tanh
from .layer import Layer
from initializations import xavier_uniform_init, zeros

class GRU(Layer):
    def __init__(self, hidden_size: int, return_sequences: bool=False, bidirectional: bool=False):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        batch_size, _, input_dim = input_shape
        self.gru_unit = GRUUnit(self.hidden_size)
        self.gru_unit.initialize_parameters(input_shape=(batch_size, input_dim))
        if self.bidirectional:
            self.gru_unit_rev = GRUUnit(self.hidden_size)
            self.gru_unit_rev.initialize_parameters(input_shape=(batch_size, input_dim))

    def build(self, input_shape: tuple):
        self.initialize_parameters(input_shape)
    
    def forward_propagation(self, inp: Tensor, training: bool=True):
        self.input = inp
        batch_size, seq_len, _ = inp.shape
        self.h = Parameter(zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
        if self.bidirectional:
            self.h_rev = Parameter(np.zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
        for t in range(seq_len):
            ht = self.gru_unit.forward_propagation(inp[:, t], self.h[:, t], training=training)
            self.h = self.h.set_item((slice(None), t+1), ht)
            if self.bidirectional:
                curr_t = seq_len - t - 1
                ht_rev = self.gru_unit_rev.forward_propagation(inp[:, curr_t], self.h_rev[:, curr_t+1], training=training)
                self.h_rev = self.h_rev.set_item((slice(None), t), ht_rev)

        if self.return_sequences:
            if self.bidirectional:
                forward_hidden_states = self.h[:, 1:]
                backward_hidden_states = self.h_rev[:, :-1]
                self.hidden_states = Tensor.concatenate([forward_hidden_states, backward_hidden_states], axis=-1)
            else:
                self.hidden_states = self.h[:, 1:]
        else:
            if self.bidirectional:
                self.hidden_states = Tensor.concatenate([self.h[:,-1], self.h_rev[:,0]], axis=-1)
            else:
                self.hidden_states = self.h[:,-1]

        return self.hidden_states


class GRUUnit:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
    
    def initialize_parameters(self, input_shape: tuple) -> None:
        batch_size, input_dim = input_shape
        self.w_ir = Parameter(xavier_uniform_init((self.hidden_size, input_dim)), name='w_ir')
        self.w_hr = Parameter(xavier_uniform_init((self.hidden_size, self.hidden_size)), name='w_hr')
        self.b_ir = Parameter(zeros((self.hidden_size,)), name='b_ir')
        self.b_hr = Parameter(zeros((self.hidden_size,)), name='b_hr')

        self.w_iz = Parameter(xavier_uniform_init((self.hidden_size, input_dim)), name='w_iz')
        self.w_hz = Parameter(xavier_uniform_init((self.hidden_size, self.hidden_size)), name='w_hz')
        self.b_iz = Parameter(zeros((self.hidden_size,)), name='b_iz')
        self.b_hz = Parameter(zeros((self.hidden_size,)), name='b_hz')

        self.w_in = Parameter(xavier_uniform_init((self.hidden_size, input_dim)), name='w_in')
        self.w_hn = Parameter(xavier_uniform_init((self.hidden_size, self.hidden_size)), name='w_hn')
        self.b_in = Parameter(zeros((self.hidden_size,)), name='b_in')
        self.b_hn = Parameter(zeros((self.hidden_size,)), name='b_hn')

    def forward_propagation(self, inp: Tensor, h_prev: Tensor, training: bool=True) -> Tensor:
        r = sigmoid(inp @ self.w_ir + h_prev @ self.w_hr + self.b_ir + self.b_hr)
        z = sigmoid(inp @ self.w_iz + h_prev @ self.w_hz + self.b_iz + self.b_hz)
        n = tanh(inp @ self.w_in + self.b_in + r * (h_prev @ self.w_hn + self.b_hn))
        h = (1 - z) * n + z * h_prev
        return h
        