from typing import List
from abc import ABC, abstractmethod


class Preprocesser(ABC):
    
    def __init__(self, spacy_model_name: str):
        import spacy
        self.spacy_model = spacy.load(spacy_model_name)
        
    def remove_stopwords(self, text: str) -> str:
        doc = self.spacy_model(text)
        text_without_stopwords = ' '.join([token.text_with_ws for token in doc if not token.is_stop])
        return text_without_stopwords
    
    def remove_punctuation(self, text: str) -> str:
        doc = self.spacy_model(text)
        text_without_punctuation = ' '.join([token.text_with_ws for token in doc if not token.is_punct])
        return text_without_punctuation
    
    def remove_stopwords_and_punctuation(self, text: str) -> str:
        doc = self.spacy_model(text)
        text_without_stopwords_and_punctuation = ' '.join([token.text_with_ws for token in doc if not token.is_stop and not token.is_punct])
        return text_without_stopwords_and_punctuation
    
    @abstractmethod
    def preprocess(self, texts: List[str], remove_stopwords: bool, remove_punctuation: bool) -> List[str]:
        pass
