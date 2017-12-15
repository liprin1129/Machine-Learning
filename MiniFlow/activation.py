from node import Node
import numpy as np

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
        
    def _sigmoid(self, x):
        #print "SIGMOID X: ", np.min(x), np.max(x)
        return 1. / ( 1. + np.exp(-x) )

    def forward_propagation(self):
        #print "IN!: ", self.inbound_nodes
        self.value = self._sigmoid(self.inbound_nodes[0].value)
        # print "SIGMOID: ", np.shape(self.value), "\n", self.value, "\n"
        
    def backward_propagation(self):
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}
        
        for outbound_node in self.outbound_nodes:
            error = outbound_node.gradients[self]

            # print "BACK ACTIVATION ERROR: ", np.shape(error), "\n", error
            sigmoid = self.value

            #print "KKK: ", np.shape(self.gradients[self.inbound_nodes[0]])
            #print "SIGMOID: ", np.shape(self.value), "\n"
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1. - sigmoid) * error
            # print "GRAD SIGMOID: ", np.shape(self.gradients[self.inbound_nodes[0]]), "\n"

        # print self.gradients

if __name__ == "__main__":
    from input import Input
    from network import Network
    from linear import Linear
    from cost import L2

    """
    #One input one layer
    X, y = Input(), Input()
    W, b = Input(), Input()
        
    X_ = np.reshape(np.array([-1., -2.]), (1, 2))
    W_ = np.reshape(np.array([2., -3.]), (1, 2))
    b_ = np.reshape(np.array([-3.]), (-1, 1))
    y_ = np.reshape(np.array([1.]), (-1, 1))

    l1 = Linear(X, W, b)
    s1 = Sigmoid(l1)
    cost = L2(y, s1)

    feed_dict = {X: X_, y: y_,
                 W: W_, b: b_}
    """

    # Two input one layer
    X, y = Input(), Input()
    W, b = Input(), Input()
        
    X_ = np.array([[1., 2.], [-1, -2]])
    W_ = np.reshape(np.array([1., -3.]), (1, 2))
    b_ = np.reshape(np.array([-3.]), (-1, 1))
    y_ = np.reshape(np.array([1., 2.]), (-1, 1))

    l1 = Linear(X, W, b)
    s1 = Sigmoid(l1)
    cost = L2(y, s1)

    feed_dict = {X: X_, y: y_,
                 W: W_, b: b_}

    """ # Two input two layer
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    X_ = np.reshape(np.array([[-1., -2.], [1., 2.]]), (2, 2))
    W1_ = np.reshape(np.array([2., 3.]), (1, 2))
    b1_ = np.reshape(np.array([-3.]), (-1, 1))
    W2_ = np.reshape(np.array([2., 3.]), (1, 2))
    b2_ = np.reshape(np.array([-3.]), (-1, 1))
    y_ = np.reshape(np.array([1.]), (-1, 1))

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = L2(y, l2)

    feed_dict = {X: X_, y: y_,
                 W1: W1_, b1: b1_,
                 W2: W2_, b2: b2_}
    """

    graph = Network.topological_sort(feed_dict)
    Network.forward_propagation(graph)
    Network.backward_propagation(graph)
