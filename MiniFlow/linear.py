from node import Node
import numpy as np

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
        
    def forward_propagation(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        
        self.value = np.dot(weights, inputs) + bias
        #self.value = bias
        #for x, w in zip(inputs, weights):
        #    self.value += x * w

if __name__ == "__main__":
    from input import Input
    from network import Network
    
    inputs, weights, bias = Input(), Input(), Input()

    f = Linear(inputs, weights, bias)

    feed_dict = {
        inputs: [6, 14, 3],
        weights: [0.5, 0.25, 1.4],
        bias: 2
    }
    
    graph = Network.topological_sort(feed_dict)
    output = Network.forward_propagation(f, graph)
    
    print(output) # should be 12.7 with this example
