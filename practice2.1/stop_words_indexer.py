
from advanced_indexer import AdvancedInvertedIndex


class StopWordsIndexer(AdvancedInvertedIndex):
    def __init__(self, stop_words_file):
        super().__init__()
        self.stop_words = self.load_stop_words(stop_words_file)
    
    def load_stop_words(self, filename):
        """Charger la liste des stop words"""
        with open(filename, 'r', encoding='utf-8') as file:
            stop_words = set(line.strip().lower() for line in file if line.strip())
        return stop_words
    
    def preprocess_text(self, text):
        """Surcharger pour filtrer les stop words"""
        tokens = super().preprocess_text(text)
        # Filtrer les stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens