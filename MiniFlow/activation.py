from node import Node
import numpy as np

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def __sigmoid(self, x):
        return 1. / ( 1. + np.exp(-x) )

    def forward_propagation(self):
        print len(self.inbound_nodes)
        self.value = self.__sigmoid(self.inbound_nodes[0].value)

if __name__ == "__main__":
    from input import Input
    from network import Network
    from linear import Linear
    
    X, W, b = Input(), Input(), Input()

    f = Linear(X, W, b)
    g = Sigmoid(f)

    X_ = np.array([[-1., -2.], [-1, -2]])
    W_ = np.array([[2., -3], [2., -3]])
    b_ = np.array([-3., -5])

    feed_dict = {X: X_.T, W: W_.T, b: b_}
    
    graph = Network.topological_sort(feed_dict)
    output = Network.forward_propagation(g, graph)
    
    print(output.T)

