from node import Node 

class Helper(object):

    @classmethod
    def topological_sort(cls, feed_dict):
        """ Khan's algorithm """
        input_nodes, node_group = cls.__in_out_update__(feed_dict)

        L = []
        next_nodes = set(input_nodes)

        while len(next_nodes) > 0:
            node = next_nodes.pop()

            for out_node in node.outbound_nodes:
                
        return S

    def __in_out_update__(self, feed_dict):
        # Insert inbound and outbout value for every node
        input_nodes = feed_dict.keys()
        nodes = [n for n in input_nodes]
        all_nodes = {}
        
        while not len(next_nodes) - 1 > 0:
            selected_node = nodes.pop(0) # return the first node of nodes
            if selected_node not in all_nodes:
                all_nodes[selected_node] = {'in': set(), 'out': set()}

            # iterate all outbound_nodes in the selected node
            for out_node in node.outbound_nodes:

                if out_node not in all_nodes:
                    all_nodes[out_node] = {'in': set(), 'out': set()}

                # insert outbound_node into selected_node's inbound_node
                all_nodes[selected_node]['out'].add[out_node]
                # insert selected_node to outbound_node's inbound_node
                all_nodes[out_node]['in'].add[selected_node]

                # update all nodes list by inserting outbound_node if it isn't in the list.
                if out_node not in nodes:
                    nodes.append(out_node)

        return input_nodes, node_group

if __name__ == "__main__":
    from input import Neuron
    x, y, z = Neuron(), Neuron(), Neuron()
    
    feed_dict = {x: 10, y: 20, z: None}
    print(Helper.topological_sort(feed_dict))
