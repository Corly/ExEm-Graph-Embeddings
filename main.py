from datasets.dataset_loader import DatasetLoader
from preprocess.preprocess import Preprocesser
from graph.graph import CoappearanceGraph
from random_walk.random_walker import RandomWalker
from trainer.w2v import Word2VecTrainer
from trainer.ft import FastTextTrainer
import logging
logging.basicConfig(level=logging.INFO)


def train_w2v(random_walks):
    logging.info("Training word2vec model...")
    w2v_trainer = Word2VecTrainer(embedding_dimension=128, window_size=8)
    saved_path = 'saved_models/ExEm_w2v.model'
    w2v_trainer.train(random_walks, saved_path)
    logging.info("Word2vec model trained. You can find it in {}".format(saved_path))
    
    
def train_ft(random_walks):
    logging.info("Training FastText model...")
    ft_trainer = FastTextTrainer(embedding_dimension=128, window_size=8, epochs=10)
    saved_path = 'saved_models/ExEm_ft.model'
    ft_trainer.train(random_walks, saved_path)
    logging.info("FastText model trained. You can find it in {}".format(saved_path))


if __name__ == "__main__":
    
    logging.info("Loading dataset...")
    texts = DatasetLoader.load_inspec_dataset()
    logging.info("Dataset loaded.")
    
    logging.info("Preprocessing dataset...")
    preprocesser = Preprocesser('en_core_web_lg')
    lemma_ids_texts = preprocesser.preprocess(texts)
    logging.info("Dataset preprocessed.")
    
    logging.info("Building coappearance graph...")
    graph = CoappearanceGraph(window_size=8)
    graph.build_graph_from_preprocessed_texts(lemma_ids_texts)
    logging.info("Coappearance graph built.")
    
    logging.info("Generating random walks...")
    random_walker = RandomWalker(graph, number_of_walks_per_node=10, walk_length=32)
    walks = random_walker.get_random_walks()
    logging.info("Random walks generated.")
    
    train_ft(walks)