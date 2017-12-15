from abc import ABCMeta, abstractmethod

class Node(object):
    __metaclass__ = ABCMeta
    def __init__(self, inbound_nodes = []):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        if not len(self.inbound_nodes) == 0:
            for node in self.inbound_nodes:
                node.outbound_nodes.append(self)

        self.value = None
        self.gradients = {}


    @abstractmethod
    def forward_propagation(self):
        pass
    
    """ # if abs module isn't allowed for your python version
    def forward(self):
        raise NotImplementedError("Forward propagation is neccessary!")
    """


    @abstractmethod
    def backward_propagation(self):
        pass
    
if __name__ == "__main__":
    node = Node()
