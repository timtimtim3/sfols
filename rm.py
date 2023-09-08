
import networkx as nx

class FiniteStateAutomaton:

    def __init__(self) -> None:
        self.graph =  nx.DiGraph()

    def add_state(self, node_name):
        self.graph.add_node(node_name) 

    def add_transition(self, src_state, dst_state, label):
        self.graph.add_edge(src_state, dst_state, predicate= label)

    def in_transitions(self, node):
        return list(self.graph.in_edges(node))
    
    def get_predicate(self, edge):
        predicates = nx.get_edge_attributes(self.graph, 'predicate')
        return predicates[edge]



