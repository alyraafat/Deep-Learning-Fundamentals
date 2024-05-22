from .optimizer import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params,grads)):
            self.velocity[i] = self.momentum * self.velocity[i]+ (1-self.momentum) * grad
            param -= self.learning_rate * self.velocity[i]
        