from node import Node
import numpy as np

class MSE(Node):
    def __init__(self, true_y, pred_y):
        Node.__init__(self, [true_y, pred_y])


    def forward_propagation(self):
        true_y = np.reshape(self.inbound_nodes[0].value, (-1, 1))
        pred_y = np.reshape(self.inbound_nodes[1].value, (-1, 1))
        
        self.value = np.sum( (true_y - pred_y)**2 ) / 2.

if __name__ == "__main__":
    from input import Input
    from network import Network
    from linear import Linear
    from activation import Sigmoid

    y, a = Input(), Input()
    cost = MSE(y, a)

    y_ = np.array([[1],
                   [2],
                   [3]])
    a_ = np.array([[4.5],
                   [5],
                   [10]])

    feed_dict = {y: y_, a: a_}
    
    graph = Network.topological_sort(feed_dict)
    Network.forward_propagation(graph)
    
    print(cost.value)
