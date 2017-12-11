from node import Node

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def depolarization(self):
        self.value = 0
        for inbound_node in self.inbound_nodes:
            self.value += inbound_node.value
