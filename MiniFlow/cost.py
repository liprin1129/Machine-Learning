from node import Node
import numpy as np

class L2(Node):
    def __init__(self, true_y, pred_y):
        Node.__init__(self, [true_y, pred_y])
        self.diff = 0.
        self.pred_y = 0.
        
    def forward_propagation(self):
        # true_y = np.reshape(self.inbound_nodes[0].value, (-1, 1))
        # pred_y = np.reshape(self.inbound_nodes[1].value, (-1, 1))

        true_y = self.inbound_nodes[0].value
        self.pred_y = self.inbound_nodes[1].value
        
        self.diff = true_y - self.pred_y
        #print "TRUE_Y: ", np.shape(true_y), "\n", true_y
        #print "PRED_Y: ", np.shape(pred_y), "\n", pred_y

        #self.m = self.inbound_nodes[0].value.shape[0]
        self.value = np.sum( self.diff**2. ) / 2.
        #self.value = np.mean(self.diff**2)
        #print "L2: ", self.value, ", PRED_Y: ", pred_y, "\n"
        
    def backward_propagation(self):
        # gradient of true_y
        self.gradients[self.inbound_nodes[0]] = self.diff
        # gradient of pred_y
        self.gradients[self.inbound_nodes[1]] = - self.diff

        # print "BACK L2: ", np.shape(self.gradients[self.inbound_nodes[1]]), "\n"

if __name__ == "__main__":
    from input import Input
    from network import Network
    from linear import Linear
    from activation import Sigmoid

    y, a = Input(), Input()
    cost = L2(y, a)

    y_ = np.array([[1],
                   [2],
                   [3]])
    a_ = np.array([[4.5],
                   [5],
                   [10]])

    feed_dict = {y: y_, a: a_}
    
    graph = Network.topological_sort(feed_dict)
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
    print(cost.value)
