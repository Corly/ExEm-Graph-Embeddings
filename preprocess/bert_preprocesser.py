from typing import List
from preprocess.preprocesser import Preprocesser
from transformers import AutoTokenizer


class BertPreprocesser(Preprocesser):
    
    def __init__(self, spacy_model_name: str, bert_model_name: str):
        import spacy
        self.spacy_model = spacy.load(spacy_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    def preprocess(self, texts: List[str], remove_stopwords: bool = False, 
                        remove_punctuation: bool = False) -> List[List[int]]:
        lemma_ids_texts: List[List[int]] = []
        for text in texts:
            transformed_text = text
            if remove_stopwords and remove_punctuation:
                transformed_text = self.remove_stopwords_and_punctuation(transformed_text)
            elif remove_stopwords:
                transformed_text = self.remove_stopwords(transformed_text)
            elif remove_punctuation:
                transformed_text = self.remove_punctuation(transformed_text)
            
            tokenized_text = self.bert_tokenizer.tokenize(transformed_text)
            bert_token_ids_text = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            if bert_token_ids_text:
                lemma_ids_texts.append(bert_token_ids_text)
        return lemma_ids_texts