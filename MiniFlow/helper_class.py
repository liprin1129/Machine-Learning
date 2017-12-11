from input import Input

class Network(object):
    
    @classmethod
    def topological_sort(cls, feed_dict):
        """ Khan's algorithm """
        input_nodes, node_group =cls.__in_out_update__(Network(), feed_dict)
        sorted_nodes = []
        next_nodes = set(input_nodes)

        while len(next_nodes) > 0:
            node = next_nodes.pop()

            if isinstance(node, Input):
                node.value = feed_dict[node]
                
            sorted_nodes.append(node)
            
            for out_node in node.outbound_nodes:
                node_group[node]['out'].remove(out_node)
                node_group[out_node]['in'].remove(node)

                if len(node_group[out_node]['in']) == 0:
                    next_nodes.add(out_node)

        return sorted_nodes
    
    @staticmethod
    def __in_out_update__(self, feed_dict):
        """
        Args:

        Returns:
        input_nodes: keys of feed_dict which represent all nodes
        """
        # Insert inbound and outbout value for every node
        input_nodes = feed_dict.keys()
        nodes = [n for n in input_nodes]
        node_group = {}
        
        while len(nodes) > 0:
            selected_node = nodes.pop(0) # return the first node of nodes

            if selected_node not in node_group:
                node_group[selected_node] = {'in': set(), 'out': set()}

            # iterate all outbound_nodes in the selected node
            for idx, out_node in enumerate(selected_node.outbound_nodes):
                if out_node not in node_group:
                    node_group[out_node] = {'in': set(), 'out': set()}

                # insert outbound_node into selected_node's inbound_node
                node_group[selected_node]['out'].add(out_node)
                # insert selected_node to outbound_node's inbound_node
                node_group[out_node]['in'].add(selected_node)

                # update all nodes list by inserting outbound_node if it isn't in the list.
                if out_node not in nodes:
                    nodes.append(out_node)

        return input_nodes, node_group

    @staticmethod
    def forward_propagation(output_node, sorted_nodes):
        for node in sorted_nodes:
            node.depolarization()

        return output_node.value
    
if __name__ == "__main__":

    from add import Add
    
    x, y, z = Input(), Input(), Input()
    f1 = Add(x, y)
    f2 = Add(f1, z)
    feed_dict = {x: 10, y: 20, z: 30}
    
    sorted_nodes = Network.topological_sort(feed_dict)
    print(Network.forward_propagation(f2, sorted_nodes))
