from node import Node
import numpy as np


class L2(Node):
    def __init__(self, true_y, pred_y):
        Node.__init__(self, [true_y, pred_y])
        self.diff = 0.
        self.pred_y = 0.
        
    def forward_propagation(self):
        true_y = self.inbound_nodes[0].value
        self.pred_y = self.inbound_nodes[1].value
        
        self.diff = true_y - self.pred_y
        self.value = np.sum(self.diff**2.) / 2.
        
    def backward_propagation(self):
        # gradient of true_y
        self.gradients[self.inbound_nodes[0]] = self.diff
        # gradient of pred_y
        self.gradients[self.inbound_nodes[1]] = - self.diff
