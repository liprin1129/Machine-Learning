from node import Node
import numpy as np

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
        
    def forward_propagation(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value

        """
        print "INPUT: ", np.shape(inputs), "\n", inputs
        print "WEIGHTS: ", np.shape(weights), "\n", weights
        print "BIAS: ", np.shape(bias), "\n", bias
        """
        
        self.value = np.dot(inputs, weights) + bias
        #print "LINEAR: ", np.shape(self.value), "\n", self.value, "\n"

        # calculation on each element (it's slower than numpy)
        #self.value = bias
        #for x, w in zip(inputs, weights):
        #    self.value += x * w
    def backward_propagation(self):
        self.gradients = {node: np.zeros_like(node.value) for node in self.inbound_nodes}

        for outbound_node in self.outbound_nodes:
            error = outbound_node.gradients[self]
            """
            print "BACK LINEAR ERROR: ", np.shape(error), "\n", error
            print "BACK LINEAR INBOUND: ", np.shape(self.inbound_nodes[0].value), "\n", self.inbound_nodes[0].value
            """
            #print "INPUT: ", np.shape(np.dot(error, self.inbound_nodes[1].value.T))

            """
            print "INPUT: ", np.shape(self.inbound_nodes[0].value)
            print "WEIGHTS: ", np.shape(self.inbound_nodes[1].value)
            print "BIAS: ", np.shape(self.inbound_nodes[2].value)
            print "GRAD: ", "\n", self.gradients[self.inbound_nodes[1]]
            print "DOT: ", "\n", np.dot(self.inbound_nodes[0].value.T, error)
            """
            
            # Partial derivative of the loss with respect to input value
            self.gradients[self.inbound_nodes[0]] += np.dot(error, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, error)

            #print "BBB: ", np.sum(error, axis = 0, keepdims = False)

            ######################################################
            ### <CAUSTION> NEED TO CHECK!!!!!!!!!!!!!!!!!!!!!! ###
            ######################################################
            """
            grad_bias = np.sum(error, axis = 0, keepdims = True)
            grad_bias = np.reshape(grad_bias, (-1, 1))
            # print "BACK BIAS: ", np.shape(grad_bias), "\n", grad_bias
            self.gradients[self.inbound_nodes[2]] += grad_bias
            """
            self.gradients[self.inbound_nodes[2]] += error
            
            # print "BACK LINEAR GRAD 0: ", np.shape(self.gradients[self.inbound_nodes[0]]), "\n", self.gradients[self.inbound_nodes[0]]
            # print "BACK LINEAR GRAD 1: ", np.shape(self.gradients[self.inbound_nodes[1]]), "\n", self.gradients[self.inbound_nodes[1]]
            # print "BACK LINEAR GRAD 2: ", np.shape(self.gradients[self.inbound_nodes[2]]), "\n"

        # print self.gradients

if __name__ == "__main__":
    from input import Input
    from network import Network
    
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)

    feed_dict = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: [2]
    }
    
    graph = Network.topological_sort(feed_dict)
    output = Network.forward_propagation(graph)
    # should be 12.7 with this example
