from node import Node
import numpy as np


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward_propagation(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        #bias = np.sum(self.inbound_nodes[2].value, axis=0)
        bias = self.inbound_nodes[2].value
        #print "BIAS: ", np.shape(bias)
        
        self.value = np.dot(inputs, weights) + bias

        # calculation on each element (it's slower than numpy) """
        # self.value = bias
        # for x, w in zip(inputs, weights):
        #    self.value += x * w """
        
    def backward_propagation(self):
        self.gradients = {node: np.zeros_like(node.value)
                          for node in self.inbound_nodes}

        for outbound_node in self.outbound_nodes:
            error = outbound_node.gradients[self]
            #print "ERROR: ", np.shape(np.sum(error, axis=0))
            """
            Partial derivatives of the error with respect to
            inputs, weights, or biases
            """
            
            # Error for Inputs
            self.gradients[self.inbound_nodes[0]] += np.dot(
                error, self.inbound_nodes[1].value.T)

            # Error for Weights
            self.gradients[self.inbound_nodes[1]] += np.dot(
                self.inbound_nodes[0].value.T, error)

            # Error for Biases
            self.gradients[self.inbound_nodes[2]] += np.sum(error, axis=0)
            #self.gradients[self.inbound_nodes[2]] += error
