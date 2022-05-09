from typing import List
from preprocess.preprocesser import Preprocesser

class SpacyPreprocesser(Preprocesser):
    
    def __init__(self, spacy_model_name: str):
        import spacy
        self.spacy_model = spacy.load(spacy_model_name)
    
    def preprocess(self, texts: List[str], remove_stopwords: bool = False, 
                        remove_punctuation: bool = False) -> List[List[int]]:
        lemma_ids_texts: List[List[int]] = []
        for text in texts:
            doc = self.spacy_model(text)
            lemma_ids_text = [token.lemma for token in doc
                                if not (token.is_stop and remove_stopwords) 
                                    and not (token.is_punct and remove_punctuation)
                                    and not token.is_space]
            if lemma_ids_text:
                lemma_ids_texts.append(lemma_ids_text)
        return lemma_ids_texts