from .tensor import Tensor
from .parameter import Parameter
import inspect
from typing import Iterator


class Module:

    def __init__(self):
        self._modules = []
        
    def add_module(self, name: str, module: 'Module'):
        setattr(self, name, module)
        self._modules.append((name, module))
    
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                # value.curr_layer = name
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def set_parameter_curr_layer(self, curr_layer: str) -> None:
        for name, value in inspect.getmembers(self):
            # print(f'curr layer: {curr_layer}')
            if isinstance(value, Parameter):
                # print(f"Setting curr_layer for Parameter: {name}")
                value.curr_layer = curr_layer  # Use the property setter
            elif isinstance(value, Module):
                # print(f"Recursively setting curr_layer in Module: {name}")
                value.set_parameter_curr_layer(curr_layer)  # Recursively set


    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
    def forward(self, input: Tensor, training=True) -> Tensor:
        raise NotImplementedError
    
    def predict(self, input: Tensor) -> Tensor:
        return self.forward(input, training=False)
    
    def summary(self) -> None:
        # print(self)
        summary = []
        # print(inspect.getmembers(self))
        # for name, value in inspect.getmembers(self):
        for name, module in self._modules:
            if isinstance(module, Module):
                summary.append({
                    'name': module.__class__.__name__,
                    'input shape': module.input_shape,
                    'output shape': module.output_shape,
                    'num params': module.num_params
                })
                nested_summary = module.summary()
                if nested_summary:
                    summary.extend(nested_summary)
                
        return summary