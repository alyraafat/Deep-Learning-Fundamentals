from autograd import Tensor
from graphviz import Digraph

def draw_computation_graph(tensor: Tensor, filename='computation_graph'):
    dot = Digraph()

    def add_nodes(tensor: Tensor):
        if str(tensor.id) not in dot.node_attr:
            dot.node(str(tensor.id), label=f"{tensor.id}, {tensor.requires_grad}") #  {tensor.name}
            for dependency in tensor.depends_on:
                dot.edge(str(dependency.tensor.id), str(tensor.id))
                add_nodes(dependency.tensor)

    add_nodes(tensor)
    dot.render(filename, format='png')
