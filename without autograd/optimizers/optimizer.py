
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        raise NotImplementedError

    def set_hyperparameters(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, param, value)