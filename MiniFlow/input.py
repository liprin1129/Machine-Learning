from node import Node

class Input(Node):
    def __init__(self):
        Node.__init__(self)
                
    def forward_propagation(self, value = None):
        """ Or you can set input value explicitly 
        after outside of this class"""
        
        if value is not None:
            self.value = value
