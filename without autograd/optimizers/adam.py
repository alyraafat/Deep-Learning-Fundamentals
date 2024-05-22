from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=10e-08, t=0):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.velocity = None
        self.S = None
        self.beta2 = beta2
        self.eps = eps
        self.t = t
        
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = []
            self.S = []
            for param in params: 
                p = np.zeros_like(param)
                self.velocity.append(p)
                self.S.append(p)
        self.t += 1
        for i, (param, grad) in enumerate(zip(params,grads)):
            self.velocity[i] = self.beta1 * self.velocity[i]+ (1-self.beta1) * grad
            self.S[i] = self.beta2 * self.S[i]+ (1-self.beta2) * (grad**2)
            velocity_corrected = self.velocity[i]/(1-self.beta1**self.t)
            S_corrected = self.S[i]/(1-self.beta2**self.t)
            # print(f'param shape: {param.shape}, velocity shape: {velocity_corrected.shape}, S shape: {S_corrected.shape}, grad shape: {grad.shape}')
            param -= self.learning_rate * velocity_corrected/(np.sqrt(S_corrected)+self.eps)