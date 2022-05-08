from typing import Set, Dict, List
from graph.node import Node


class CoappearanceGraph:
    
    def __init__(self, window_size: int):
        self.window_size: int = window_size
        self.node_set: Set[Node] = set()
        self.node_id_to_node: Dict[int, Node] = {}
        self.adjacency_list: Dict[Node, Set[Node]] = {}
        
    def add_or_get_node(self, node: Node) -> Node:
        if node in self.node_set:
            return self.node_id_to_node[node.node_id]
        self.node_set.add(node)
        self.node_id_to_node[node.node_id] = node
        self.adjacency_list[node] = set()
        return node
    
    def add_edge(self, node1: Node, node2: Node) -> None:
        if node1 not in self.node_set:
            self.add_or_get_node(node1)
        if node2 not in self.node_set:
            self.add_or_get_node(node2)
        self.adjacency_list[node1].add(node2)
    
    def add_edge_both_ways(self, node1: Node, node2: Node) -> None:
        self.add_edge(node1, node2)
        self.add_edge(node2, node1)
        
    def build_graph_from_preprocessed_texts(self, texts: List[List[int]]) -> None:
        for text in texts:
            for index, word_id in enumerate(text):
                current_node = self.add_or_get_node(Node(word_id))
                for neighbour_index in range(index + 1, min(len(text), index + self.window_size//2 + 1)):
                    forward_neighbour_node = self.add_or_get_node(Node(text[neighbour_index]))
                    self.add_edge_both_ways(current_node, forward_neighbour_node)

    def find_a_domination_set(self) -> Set[Node]:
        import numpy as np
        np.random.seed(42)
        domination_set: Set[Node] = set()
        node_dict = dict.fromkeys(self.node_set, None)
        while True:
            if not node_dict:
                break
            dominant_node = np.random.choice(list(node_dict.keys()))
            domination_set.add(dominant_node)
            del node_dict[dominant_node]
            for neighbour in self.adjacency_list[dominant_node]:
                if neighbour in node_dict:
                    del node_dict[neighbour]
        return domination_set
        