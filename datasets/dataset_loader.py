import os


class DatasetLoader:
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_inspec_dataset():
        texts = []
        for file_name in os.listdir('datasets/Inspec/docsutf8'):
            with open(f'datasets/Inspec/docsutf8/{file_name}', 'r', encoding='utf-8') as f:
                texts.append(f.read())
        return texts