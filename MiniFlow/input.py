from node import Node


class Input(Node):
    def __init__(self):
        Node.__init__(self)
                
    def forward_propagation(self, value=None):
        if value is not None:
            self.value = value

        """ Or you can set input value explicitly
        after outside of this class"""
        
    def backward_propagation(self):
        self.gradients = {self: 0}

        for node in self.outbound_nodes:
            error = node.gradients[self]
            self.gradients[self] += error
