from .optimizer import Optimizer
import numpy as np
from autograd.module import Module, Tensor

class Momentum(Optimizer):
    def __init__(self, learning_rate: float=0.01, momentum: float=0.9) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def step(self, module: Module):
        if self.velocity is None:
            self.velocity = [Tensor(np.zeros_like(param.data)) for param in module.parameters()]
            
        for i, parameter in enumerate(module.parameters()):
            self.velocity[i] = self.momentum * self.velocity[i]+ (1-self.momentum) * parameter.grad
            parameter -= self.learning_rate * self.velocity[i]
            
            
        