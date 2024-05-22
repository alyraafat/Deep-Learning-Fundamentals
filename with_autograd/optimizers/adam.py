from .optimizer import Optimizer
import numpy as np
from autograd.module import Module, Tensor

class Adam(Optimizer):
    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=10e-08, t: int=0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.velocity = None
        self.S = None
        self.beta2 = beta2
        self.eps = eps
        self.t = t
        
    def step(self, module: Module):
        if self.velocity is None:
            self.velocity = []
            self.S = []
            for param in module.parameters(): 
                p = Tensor(np.zeros_like(param.data))
                self.velocity.append(p)
                self.S.append(p)
        self.t += 1
        for i, param in enumerate(module.parameters()):
            self.velocity[i] = self.beta1 * self.velocity[i]+ (1-self.beta1) * param.grad
            self.S[i] = self.beta2 * self.S[i]+ (1-self.beta2) * (param.grad**2)
            velocity_corrected = self.velocity[i]/(1-self.beta1**self.t)
            S_corrected = self.S[i]/(1-self.beta2**self.t)
            # print(f'param shape: {param.shape}, velocity shape: {velocity_corrected.shape}, S shape: {S_corrected.shape}, grad shape: {grad.shape}')
            param -= self.learning_rate * velocity_corrected/(S_corrected**0.5+self.eps)