from .optimizer import Optimizer
import numpy as np
from autograd.module import Module, Tensor

class RMSprop(Optimizer):
    def __init__(self, learning_rate: float=0.01, beta: float=0.999, eps: float=10e-08) -> None:
        super().__init__(learning_rate)
        self.beta = beta
        self.S = None
        self.eps = eps
        
    def step(self, module: Module):
        if self.S is None:
            self.S = [Tensor(np.zeros_like(param.data)) for param in module.parameters()]

        for i, param in enumerate(module.parameters()):
            self.S[i] = self.beta * self.S[i]+ (1-self.beta) * (param.grad**2)
            param -= self.learning_rate * param.grad/(np.sqrt(self.S[i])+self.eps)
        