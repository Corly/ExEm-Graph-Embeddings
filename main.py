from datasets.dataset_loader import DatasetLoader
from preprocess.spacy_preprocesser import SpacyPreprocesser
from preprocess.bert_preprocesser import BertPreprocesser
from preprocess.preprocesser import Preprocesser
from graph.graph import CoappearanceGraph
from random_walk.random_walker import RandomWalker
from trainer.w2v import Word2VecTrainer
from trainer.ft import FastTextTrainer
import argparse
import json
from typing import Dict

import logging
logging.basicConfig(level=logging.INFO)



def get_file_name(model_type: str) -> str:
    import time
    return "ExEm_" + model_type + "_" + str(int(time.time()))


def write_description_json(save_directory: str, model_name: str, config: Dict) -> None:
    with open(f"{save_directory}/{model_name}_description.json", "w") as f:
        f.write(json.dumps(config))


def train_w2v(random_walks, config) -> None:
    logging.info("Training word2vec model...")
    w2v_trainer = Word2VecTrainer(embedding_dimension=config["embedding_dimension"], window_size=config["window_size"])
    model = w2v_trainer.train(random_walks)
    file_name = get_file_name(config["model"])
    model.save(f"{config['save_directory']}/{file_name}.model")
    write_description_json(config["save_directory"], file_name, config)
    logging.info(f"Word2vec model trained. You can find it in {config['save_directory']}/{file_name}.model")
    
    
def train_ft(random_walks, config) -> None:
    logging.info("Training FastText model...")
    ft_trainer = FastTextTrainer(embedding_dimension=config["embedding_dimension"], window_size=config["window_size"], epochs=config["epochs"])
    model = ft_trainer.train(random_walks)
    file_name = get_file_name(config["model"])
    model.save(f"{config['save_directory']}/{file_name}.model")
    write_description_json(config["save_directory"], file_name, config)
    logging.info(f"FastText model trained. You can find it in {config['save_directory']}/{file_name}.model")


def initialize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a ExEm graph embedding model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", type=str, default="Inspec", help="Dataset name. Supported datasets: Inspec")
    parser.add_argument("-p", "--preprocesser", type=str, required=True, help="Preprocesser name. Supported preprocessers: Spacy, BERT")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name. Supported models: W2V, FT")
    parser.add_argument("-rsw", "--remove_stopwords", action="store_true", help="Remove stopwords from the text")
    parser.add_argument("-rp", "--remove_punctuation", action="store_true", help="Remove punctuation from the text")
    parser.add_argument("-ws", "--window_size", type=int, default=10, help="Window size for the random walk")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Number of epochs for the FastText model")
    parser.add_argument("-nwn", "--num_walks_per_node", type=int, default=10, help="Number of random walks per dominant node")
    parser.add_argument("-wl", "--walk_length", type=int, default=80, help="Length of the random walk")
    parser.add_argument("-em", "--embedding_dimension", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("-sd", "--save_directory", type=str, default="saved_models", help="Directory path to save the model")
    return parser


if __name__ == "__main__":
    parser = initialize_parser()
    args = parser.parse_args()
    config = vars(args)
    
    if config["dataset"] not in ["Inspec"]:
        exit("Dataset {} not supported".format(config["dataset"]))
    if config["preprocesser"] not in ["Spacy", "BERT"]:
        exit("Preprocesser {} not supported".format(config["preprocesser"]))
    if config["model"] not in ["W2V", "FT"]:
        exit("Model {} not supported".format(config["model"]))
    logging.info("Loading dataset...")
    texts = DatasetLoader.load_inspec_dataset()
    logging.info("Dataset loaded.")
    
    
    logging.info("Preprocessing dataset...")
    preprocesser: Preprocesser = None
    if config["preprocesser"] == "Spacy":
        preprocesser = SpacyPreprocesser(spacy_model_name='en_core_web_lg')
    elif config["preprocesser"] == "BERT":
        preprocesser = BertPreprocesser(spacy_model_name='en_core_web_lg', bert_model_name='distilbert-base-uncased')
    lemma_ids_texts = preprocesser.preprocess(texts, remove_stopwords=config["remove_stopwords"], remove_punctuation=config["remove_punctuation"])
    logging.info("Dataset preprocessed.")
    
    logging.info("Building coappearance graph...")
    graph = CoappearanceGraph(window_size=config["window_size"])
    graph.build_graph_from_preprocessed_texts(lemma_ids_texts)
    logging.info("Coappearance graph built.")
    
    logging.info("Generating random walks...")
    random_walker = RandomWalker(graph, number_of_walks_per_node=config["num_walks_per_node"], walk_length=config["walk_length"])
    walks = random_walker.get_random_walks()
    logging.info("Random walks generated.")
    
    if config["model"] == "W2V":
        train_w2v(walks, config)
    elif config["model"] == "FT":
        train_ft(walks, config)
