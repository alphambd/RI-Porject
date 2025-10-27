
from indexer import InvertedIndex


class AdvancedInvertedIndex(InvertedIndex):
    def __init__(self):
        super().__init__()
        self.total_chars = 0
        self.total_tokens = 0
        self.document_lengths = []
    
    def compute_statistics(self):
        """Calculer les statistiques avancées"""
        # 1. Longueur moyenne des documents
        avg_doc_length = self.total_tokens / len(self.doc_ids) if self.doc_ids else 0
        
        # 2. Longueur moyenne des termes
        avg_term_length = self.total_chars / self.total_tokens if self.total_tokens else 0
        
        # 3. Taille du vocabulaire
        vocabulary_size = len(self.dictionary)
        
        return {
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length,
            'vocabulary_size': vocabulary_size,
            'total_documents': len(self.doc_ids),
            'total_tokens': self.total_tokens
        }
    
    def add_document(self, doc_id, text):
        """Surcharger pour collecter les statistiques"""
        tokens = self.preprocess_text(text)
        self.total_tokens += len(tokens)
        self.total_chars += sum(len(token) for token in tokens)
        self.document_lengths.append(len(tokens))
        
        # Appeler la méthode parent
        super().add_document(doc_id, text)