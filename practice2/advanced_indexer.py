import re
from collections import defaultdict, Counter
import gzip
import time
from porterstemmer import PorterStemmer

class AdvancedInvertedIndex:
    def __init__(self):
        # Structure principale
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.doc_lengths = {}
        
        # Statistiques globales
        self.total_tokens = 0
        self.total_chars = 0
        self.total_documents = 0
        
        # Options de traitement
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words_set = set()
        self.stemmer = PorterStemmer()
    
    def reset(self):
        """Réinitialise complètement l'index"""
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.doc_lengths = {}
        self.total_tokens = 0
        self.total_chars = 0
        self.total_documents = 0
    
    def load_stop_words(self, stop_words_file="data/stop-words-english4.txt"):
        """Charge la liste des stop words depuis un fichier"""
        try:
            with open(stop_words_file, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"- {len(self.stop_words_set)} stop words chargés")
        except FileNotFoundError:
            print(f"- Fichier {stop_words_file} non trouvé")
            self.stop_words_set = set()
    
    def simple_tokenize(self, text):
        """Tokenisation simple sans chiffres ni caractères spéciaux"""
        # Suppression des caractères non alphabétiques (conserve les espaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenisation et filtrage des tokens vides
        tokens = [token for token in text.split() if len(token) > 0]
        
        return tokens
    
    def process_tokens(self, tokens):
        """Transforme les tokens en terms avec tous les traitements"""
        # Conversion minuscules
        tokens = [token.lower() for token in tokens]
        
        # Filtrer les stop words si activé
        if self.stop_word_active and self.stop_words_set:
            tokens = [token for token in tokens if token not in self.stop_words_set]
        
        # Appliquer le stemming si activé
        if self.stemmer_active:
            tokens = [self.stemmer.stem(token, 0, len(token)-1) for token in tokens]
        
        return tokens
    
    def read_file(self, filename, is_zipped=True):
        """Lit un fichier compressé ou normal"""
        try:
            if is_zipped:
                with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                    return file.read()
            else:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                    return file.read()
        except Exception as e:
            print(f"- Erreur lecture {filename}: {e}")
            return None
    
    def build_index(self, filename, is_zipped=True, verbose=False):
        """Construit l'index depuis un fichier - STYLE UNIFIÉ"""
        #start_time = time.time()
        
        content = self.read_file(filename, is_zipped)
        if content is None:
            return None
        
        # Extraction des documents
        doc_pattern = r'<doc><docno>([^<]+)</docno>(.*?)</doc>'
        matches = re.findall(doc_pattern, content, re.DOTALL)
        
        if verbose:
            print(f"- Indexation de {len(matches)} documents...")
        
        start_time = time.time()
        # Indexation de chaque document
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            
            # Tokenisation simple
            tokens = self.simple_tokenize(doc_text)
            
            # Traitement des tokens (stop words si TRUE, stemming si TRUE)
            terms = self.process_tokens(tokens)
            
            # Mise à jour des statistiques
            doc_length = len(terms)
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = doc_length
            self.total_tokens += doc_length
            self.total_chars += sum(len(term) for term in terms)
            self.total_documents += 1
            
            # Construction du dictionnaire inversé
            term_freq = Counter(terms)
            for term, freq in term_freq.items():
                self.dictionary[term][doc_id] = freq
        
        indexing_time = time.time() - start_time
        
        if verbose:
            print(f"- Index construit avec succès en {indexing_time:.2f}s")
        
        return indexing_time
    
    def get_global_statistics(self):
        """Calcule les statistiques globales demandées"""
        if self.total_documents == 0:
            return {
                'avg_document_length': 0,
                'avg_term_length': 0,
                'vocabulary_size': 0,
                'total_documents': 0,
                'total_tokens': 0
            }
        
        avg_doc_length = self.total_tokens / self.total_documents
        avg_term_length = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        vocabulary_size = len(self.dictionary)
        
        return {
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length,
            'vocabulary_size': vocabulary_size,
            'total_documents': self.total_documents,
            'total_tokens': self.total_tokens
        }