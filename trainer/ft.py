from gensim.models import FastText
from typing import List
from graph.node import Node

class FastTextTrainer:
    
    def __init__(self, embedding_dimension: int, window_size: int, epochs: int):
        self.embedding_dimension = embedding_dimension
        self.window_size = window_size
        self.epochs_to_train = epochs
        
    def convert_ints_to_strs(self, random_walks: List[List[Node]]) -> List[List[str]]:
        return [[str(node.node_id) for node in rw] for rw in random_walks]
        
    def train(self, random_walks: List[List[Node]], save_path: str) -> None:
        str_converted_text = self.convert_ints_to_strs(random_walks)
        model = FastText(vector_size=self.embedding_dimension, window=self.window_size,
                            min_count=1, workers=4, seed=42)
        model.build_vocab(corpus_iterable=str_converted_text)
        model.train(corpus_iterable=str_converted_text, total_examples=len(str_converted_text),
                        epochs=self.epochs_to_train)
        model.save(save_path)