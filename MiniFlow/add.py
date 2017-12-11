from node import Node

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])
        self.value = 0
        
    def depolarization(self):
        for inbound_node in self.inbound_nodes:
            self.value += inbound_node.value
