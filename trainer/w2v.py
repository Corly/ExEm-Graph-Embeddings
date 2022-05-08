from gensim.models import Word2Vec
from typing import List
from graph.node import Node

class Word2VecTrainer:
    
    def __init__(self, embedding_dimension: int, window_size: int):
        self.embedding_dimension = embedding_dimension
        self.window_size = window_size
        
    def convert_ints_to_strs(self, random_walks: List[List[Node]]) -> List[List[str]]:
        return [[str(node.node_id) for node in rw] for rw in random_walks]
        
    def train(self, random_walks: List[List[Node]], save_path: str) -> None:
        str_converted_text = self.convert_ints_to_strs(random_walks)
        model = Word2Vec(sentences=str_converted_text, vector_size=self.embedding_dimension, window=self.window_size,
                            min_count=1, workers=4, seed=42)
        model.save(save_path)