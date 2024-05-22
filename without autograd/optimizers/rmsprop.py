from .optimizer import Optimizer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.999, eps=10e-08):
        super().__init__(learning_rate)
        self.beta = beta
        self.S = None
        self.eps = eps
        
    def update(self, params, grads):
        if self.S is None:
            self.S = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params,grads)):
            self.S[i] = self.beta * self.S[i]+ (1-self.beta) * (grad**2)
            param -= self.learning_rate * grad/(np.sqrt(self.S[i])+self.eps)
        