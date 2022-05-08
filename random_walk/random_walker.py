from graph.graph import CoappearanceGraph
import numpy as np
from typing import List, Set
from graph.node import Node
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)


class RandomWalker:
    
    def __init__(self, graph: CoappearanceGraph, number_of_walks_per_node: int, walk_length: int):
        self.graph: CoappearanceGraph = graph
        self.number_of_walks_per_node: int = number_of_walks_per_node
        self.walk_length: int = walk_length
        np.random.seed(42)
    
    
    def get_random_walks(self) -> List[List[Node]]:
        random_walks: List[List[Node]] = []
        
        logging.info("Finding a domination set...")
        domination_set: Set[Node] = self.graph.find_a_domination_set()
        logging.info("Found a domination set of size {}".format(len(domination_set)))
        
        nodes_neighbours = {node: list(self.graph.adjacency_list[node]) for node in self.graph.node_set}
        for start_node in tqdm(domination_set):
            random_walk: List[Node] = []
            for _ in range(self.number_of_walks_per_node):
                used_nodes = set([start_node])
                current_node = start_node
                random_walk.append(current_node)
                random_walk_len = 1
                while True:
                    if random_walk_len == self.walk_length:
                        break
                    for _ in range(1000): # 1000 tries to find a valid neighbour
                        neighbour = np.random.choice(nodes_neighbours[current_node])
                        if neighbour not in used_nodes:
                            used_nodes.add(neighbour)
                            random_walk.append(neighbour)
                            random_walk_len += 1
                            current_node = neighbour
                            break
                    else:
                        break
                if random_walk_len == self.walk_length:
                    random_walks.append(random_walk)
                    
        return random_walks
                    
                