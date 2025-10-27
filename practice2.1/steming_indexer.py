from nltk.stem import PorterStemmer
from stop_words_indexer import StopWordsIndexer
#from portestemmer import PorterStemmer


class StemmingIndexer(StopWordsIndexer):
    def __init__(self, stop_words_file):
        super().__init__(stop_words_file)
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        """Surcharger pour ajouter le stemming"""
        tokens = super().preprocess_text(text)  # Filtrage des stop words d'abord
        # Appliquer le stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens