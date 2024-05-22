from autograd import Tensor, Parameter, sigmoid, tanh
from .layer import Layer
import numpy as np
from initializations import xavier_uniform_init, zeros

class LSTM(Layer):
    def __init__(self, hidden_size: int, return_sequences: bool=False, bidirectional: bool=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        batch_size, _, input_dim = input_shape
        self.lstm_unit = LSTMUnit(self.hidden_size)
        self.lstm_unit.initialize_parameters(input_shape=(batch_size, input_dim))
        if self.bidirectional:
            self.lstm_unit_rev = LSTMUnit(self.hidden_size)
            self.lstm_unit_rev.initialize_parameters(input_shape=(batch_size, input_dim))
    
    def forward_propagation(self, inp: Tensor, training: bool=True) -> Tensor:
        self.input = inp
        batch_size, seq_len, _ = inp.shape
        self.h = Parameter(zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
        self.c = Parameter(zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
        if self.bidirectional:
            self.h_rev = Parameter(np.zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
            self.c_rev = Parameter(np.zeros(shape=(batch_size, seq_len+1, self.hidden_size)))
        for t in range(seq_len):
            ht, ct = self.lstm_unit.forward_propagation(inp[:, t], self.h[:, t], self.c[:, t], training=training)
            self.h = self.h.set_item((slice(None), t+1), ht)
            self.c = self.c.set_item((slice(None), t+1), ct)
            if self.bidirectional:
                curr_t = seq_len - t - 1
                ht_rev, ct_rev = self.lstm_unit_rev.forward_propagation(inp[:, curr_t], self.h_rev[:, curr_t+1], self.c_rev[:, curr_t+1], training=training)
                self.h_rev = self.h_rev.set_item((slice(None), t), ht_rev)
                self.c_rev = self.c_rev.set_item((slice(None), t), ct_rev)

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



class LSTMUnit:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
    
    def initialize_parameters(self, input_shape: tuple):
        self.is_initialized = True
        self.input_shape = input_shape
        batch_size, input_dim = input_shape
        self.Wii = Parameter(xavier_uniform_init(shape=(input_dim, self.hidden_size), hidden_size=self.hidden_size))
        self.Whi = Parameter(xavier_uniform_init(shape=(self.hidden_size, self.hidden_size), hidden_size=self.hidden_size))
        self.Wif = Parameter(xavier_uniform_init(shape=(input_dim, self.hidden_size), hidden_size=self.hidden_size))
        self.Whf = Parameter(xavier_uniform_init(shape=(self.hidden_size, self.hidden_size), hidden_size=self.hidden_size))
        self.Wig = Parameter(xavier_uniform_init(shape=(input_dim, self.hidden_size), hidden_size=self.hidden_size))
        self.Whg = Parameter(xavier_uniform_init(shape=(self.hidden_size, self.hidden_size), hidden_size=self.hidden_size))
        self.Wio = Parameter(xavier_uniform_init(shape=(input_dim, self.hidden_size), hidden_size=self.hidden_size))
        self.Who = Parameter(xavier_uniform_init(shape=(self.hidden_size, self.hidden_size), hidden_size=self.hidden_size))
        self.bii = Parameter(zeros(shape=(self.hidden_size,)))
        self.bhi = Parameter(zeros(shape=(self.hidden_size,)))
        self.bif = Parameter(zeros(shape=(self.hidden_size,)))
        self.bhf = Parameter(zeros(shape=(self.hidden_size,)))
        self.big = Parameter(zeros(shape=(self.hidden_size,)))
        self.bhg = Parameter(zeros(shape=(self.hidden_size,)))
        self.bio = Parameter(zeros(shape=(self.hidden_size,)))
        self.bho = Parameter(zeros(shape=(self.hidden_size,)))
        
    
    def forward_propagation(self, inp: Tensor, ht_1: Tensor, ct_1: Tensor ,training: bool=True) -> Tensor:
        self.input = inp
        self.batch_size = inp.shape[0]
        self.i = sigmoid(inp @ self.Wii + ht_1 @ self.Whi + self.bii + self.bhi)
        self.f = sigmoid(inp @ self.Wif + ht_1 @ self.Whf + self.bif + self.bhf)
        self.g = tanh(inp @ self.Wig + ht_1 @ self.Whg + self.big + self.bhg)
        self.o = sigmoid(inp @ self.Wio + ht_1 @ self.Who + self.bio + self.bho)
        self.ct = self.f * ct_1 + self.i * self.g
        self.ht = self.o * tanh(self.ct)
        return self.ht, self.ct
        



        