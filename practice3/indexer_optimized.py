import re
from collections import defaultdict
import gzip
import time
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Télécharger les données NLTK si nécessaire (à faire une fois)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NLTKOptimizedInvertedIndex:
    def __init__(self):
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.doc_lengths = {}
        self.doc_count = 0
        self.total_terms = 0
        self.total_tokens_bp = 0
        self.distinct_tokens_bp = set()
        self.total_chars_tokens = 0
        self.avg_doc_length = 0

        # NLTK components - BEAUCOUP plus rapides
        self.stemmer = PorterStemmer()
        self.nltk_stop_words = set(stopwords.words('english'))
        
        self.stop_word_active = False
        self.stemmer_active = False
        
        # Regex pré-compilées
        self.doc_pattern = re.compile(r'<doc><docno>([^<]+)</docno>(.*?)</doc>', re.DOTALL)
        self.clean_pattern = re.compile(r'[^a-zA-Z\s]')

    def load_stop_words(self, stop_words_file=None):
        """Utilise les stopwords NLTK par défaut (plus rapide)"""
        if stop_words_file:
            try:
                with open(stop_words_file, 'r', encoding='utf-8') as file:
                    self.nltk_stop_words = set(line.strip().lower() for line in file if line.strip())
                print(f"- {len(self.nltk_stop_words)} stop words chargés")
            except FileNotFoundError:
                print(f"- Utilisation des stopwords NLTK par défaut ({len(self.nltk_stop_words)} mots)")
        else:
            print(f"- Stopwords NLTK par défaut ({len(self.nltk_stop_words)} mots)")

    def apply_tokenization(self, text):
        """
        Tokenisation OPTIMISÉE avec NLTK
        """
        # Nettoyage basique
        text_clean = self.clean_pattern.sub(' ', text)
        # Tokenisation NLTK (optimisée en C)
        tokens = word_tokenize(text_clean)
        # Filtrer les tokens vides et trop courts
        return [token for token in tokens if len(token) > 1]

    def process_tokens_batch(self, tokens):
        """
        Traitement par lots OPTIMISÉ avec NLTK
        """
        if not tokens:
            return []
        
        # Case folding en une opération
        tokens_lower = [token.lower() for token in tokens]
        
        # Stop words avec set NLTK (très rapide)
        if self.stop_word_active:
            tokens_filtered = [token for token in tokens_lower if token not in self.nltk_stop_words]
        else:
            tokens_filtered = tokens_lower
        
        # Stemming NLTK batch (beaucoup plus rapide que l'implémentation Python pure)
        if self.stemmer_active and tokens_filtered:
            # NLTK PorterStemmer est optimisé en C
            return [self.stemmer.stem(token) for token in tokens_filtered]
        
        return tokens_filtered

    def read_file(self, filename, is_zipped):
        """Lecture optimisée"""
        try:
            if is_zipped:
                with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                    return file.read()
            else:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                    return file.read()
        except Exception as e:
            print(f"- Erreur lecture: {e}")
            return None

    def build_index(self, filename, is_zipped=False):
        """Construction d'index ULTRA rapide avec NLTK"""
        start_time = time.time()

        print(f"Lecture de {filename}...")
        content = self.read_file(filename, is_zipped)
        if content is None:
            return None

        # Extraction des documents
        matches = self.doc_pattern.findall(content)
        print(f"Indexation de {len(matches)} documents avec NLTK...")

        # Variables pour optimisation
        batch_size = 500
        processed = 0
        
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()

            # Tokenisation NLTK
            tokens = self.apply_tokenization(doc_text)
            
            # Statistiques tokens
            token_count = len(tokens)
            self.total_tokens_bp += token_count
            self.distinct_tokens_bp.update(tokens)
            self.total_chars_tokens += sum(len(token) for token in tokens)

            # Traitement NLTK optimisé
            terms = self.process_tokens_batch(tokens)
            doc_length = len(terms)
            
            # Mise à jour des structures
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = doc_length
            self.total_terms += doc_length

            # Construction d'index sans Counter (plus rapide)
            term_freq = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
            
            for term, freq in term_freq.items():
                self.dictionary[term][doc_id] = freq
            
            # Progression
            processed += 1
            if processed % batch_size == 0:
                print(f"  Documents traités: {processed}/{len(matches)}")

        # Métadonnées finales
        self.doc_count = len(self.doc_ids)
        self.avg_doc_length = self.total_terms / self.doc_count if self.doc_count > 0 else 0

        indexing_time = time.time() - start_time
        print(f"Index NLTK construit en {indexing_time:.2f} secondes!")

        return indexing_time

    def get_collection_statistics(self, indexing_time):
        """Calcul des statistiques"""
        distinct_tokens = len(self.distinct_tokens_bp)
        avg_token_length = (
            sum(len(token) for token in self.distinct_tokens_bp) / distinct_tokens
            if distinct_tokens > 0 else 0
        )

        distinct_terms = len(self.dictionary)
        total_chars_terms = sum(len(term) for term in self.dictionary.keys())
        avg_term_length = total_chars_terms / distinct_terms if distinct_terms > 0 else 0

        return {
            'indexing_time': indexing_time,
            'total_tokens': self.total_tokens_bp,
            'distinct_tokens': distinct_tokens,
            'avg_token_length': avg_token_length,
            'total_terms': self.total_terms,
            'distinct_terms': distinct_terms,
            'avg_doc_length': self.avg_doc_length,
            'avg_term_length': avg_term_length
        }