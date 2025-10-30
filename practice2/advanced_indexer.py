import re
from collections import defaultdict, Counter
import gzip
import time
from portestemmer import PorterStemmer


class AdvancedInvertedIndex:
    def __init__(self):
        # Structure principale
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}
        
        # Statistiques globales
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        
        # Options de traitement
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words_set = set()
        
        # Pour suivre les termes uniques par document
        self.unique_terms_per_doc = []
    
    def reset(self):
        """Réinitialise complètement l'index"""
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        self.unique_terms_per_doc = []
    
    def load_stop_words(self, stop_words_file="stop-words-english4.txt"):
        """Charge la liste des stop words depuis un fichier"""
        try:
            with open(stop_words_file, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            #print(f" {len(self.stop_words_set)} stop words chargés")
        except FileNotFoundError:
            print(f"  Fichier {stop_words_file} non trouvé")
            self.stop_words_set = set()
    
    def simple_tokenize(self, text):
        """Tokenisation simple sans chiffres ni caractères spéciaux - EXACTEMENT comme demandé"""
        # Conversion minuscules
        text = text.lower()
        
        # Suppression des caractères non alphabétiques (conserve les espaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenisation et filtrage des tokens vides
        tokens = [token for token in text.split() if len(token) > 0]
        
        return tokens
    
    def apply_stemming(self, tokens):
        """Applique le stemming de Porter sur une liste de tokens"""
        if not self.stemmer_active:
            return tokens
        
        p = PorterStemmer()
        stemmed_tokens = []
        
        for token in tokens:
            # Appliquer le stemming de Porter
            stemmed_token = p.stem(token, 0, len(token) - 1)
            stemmed_tokens.append(stemmed_token)
        
        return stemmed_tokens
    
    def compute_statistics(self):
        """Calcule les statistiques APRÈS l'indexation pour optimiser le temps"""
        self._total_tokens = 0
        self._total_chars = 0
        self._document_lengths = []
        
        for doc_id, tokens in self._tokens_cache.items():
            doc_length = len(tokens)
            self._total_tokens += doc_length
            self._total_chars += sum(len(token) for token in tokens)
            self._document_lengths.append(doc_length)
    
    def add_document(self, doc_id, text):
        """Ajoute un document avec tous les traitements"""
        self.doc_ids.append(doc_id)
        self.all_documents[doc_id] = text
        
        # Tokenisation simple
        tokens = self.simple_tokenize(text)
        
        # Filtrer les stop words si activé
        if self.stop_word_active and self.stop_words_set:
            tokens = [token for token in tokens if token not in self.stop_words_set]
        
        # Appliquer le stemming si activé
        if self.stemmer_active:
            tokens = self.apply_stemming(tokens)
        
        # Mise à jour des statistiques
        doc_length = len(tokens)
        self.total_tokens += doc_length
        self.total_chars += sum(len(token) for token in tokens)
        self.document_lengths.append(doc_length)
        self.unique_terms_per_doc.append(len(set(tokens)))
        
        # Mise à jour du dictionnaire inversé
        term_freq = Counter(tokens)
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
    
    def build_from_file(self, filename, verbose=False, print_index=False):
        """Construit l'index depuis un fichier avec mesure précise du temps"""
        #start_time = time.time()
        
        try:
            # Lecture du fichier compressé
            with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        except Exception as e:
            print(f" Erreur lecture {filename}: {e}")
            return None
        
        start_time = time.time()
        # Extraction des documents - pattern corrigé
        doc_pattern = r'<doc><docno>([^<]+)</docno>(.*?)</doc>'
        matches = re.findall(doc_pattern, content, re.DOTALL)
        
        # Indexation de chaque document
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.add_document(doc_id, doc_text)
        
        end_time = time.time()
        indexing_time = end_time - start_time
        
        if verbose:
            print(f" {filename}: {len(matches)} documents indexés en {indexing_time:.2f}s")
        
        # Affichage de l'index uniquement pour petites collections
        if print_index and len(matches) <= 10:
            self.display_index(limit=10)
        
        return indexing_time
    
    def get_global_statistics(self):
        """Calcule les statistiques globales demandées dans l'exercice 2.1"""
        if not self.doc_ids:
            return {
                'avg_document_length': 0,
                'avg_term_length': 0,
                'vocabulary_size': 0,
                'total_documents': 0,
                'total_tokens': 0
            }
        
        avg_doc_length = self.total_tokens / len(self.doc_ids) if self.doc_ids else 0
        avg_term_length = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        vocabulary_size = len(self.dictionary)
        
        return {
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length,
            'vocabulary_size': vocabulary_size,
            'total_documents': len(self.doc_ids),
            'total_tokens': self.total_tokens
        }
    
