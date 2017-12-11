from abc import ABCMeta, abstractmethod

class Node(object):
    __metaclass__ = ABCMeta
    def __init__(self, inbound_nodes = []):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        if not self.inbound_nodes == 0:
            for n in self.inbound_nodes:
                n.outbound_nodes.append(self)

        self.value = None


    @abstractmethod
    def depolarization(self):
        pass
    
    """ # if abs module isn't allowed for your python version
    def forward(self):
        raise NotImplementedError("Forward propagation is neccessary!")
    """
    
if __name__ == "__main__":
    node = Node()
