
from autograd.module import Module

class Optimizer:
    def __init__(self, learning_rate: float=0.01) -> None:
        self.learning_rate = learning_rate
    
    def step(self, module: Module) -> None:
        raise NotImplementedError

    def set_hyperparameters(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, param, value)