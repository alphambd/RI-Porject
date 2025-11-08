import re
from collections import defaultdict, Counter
import gzip
import time
from porterstemmer import PorterStemmer


class WeightedInvertedIndex:
    def __init__(self, stop_words_set):
        self.dictionary = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_ids = []
        self.doc_lengths = {}  # doc_id -> length in terms
        self.doc_count = 0
        self.total_terms = 0
        self.total_tokens_bp = 0  # tokens avant traitement, bp = before processing
        self.distinct_tokens_bp = set()  # tokens distincts avant traitement
        self.total_chars_tokens = 0  # total caractères pour tokens
        self.avg_doc_length = 0
        self.stop_words_set = stop_words_set

        # Options
        self.stop_words_set = set()
        self.stemmer = PorterStemmer()
        self.stop_word_active = False
        self.stemmer_active = False

    """
    def load_stop_words(self, stop_words_file="data/stop-words-english4.txt"):

        try:
            with open(stop_words_file, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"- {len(self.stop_words_set)} stop words chargés")
        except FileNotFoundError:
            print(f"- Fichier {stop_words_file} non trouvé")
    """

    def apply_tokenization(self, text):
        """
        Extraction des tokens bruts du texte.
        Retire ponctuation et chiffres si nécessaire.
        """
        # Retirer chiffres et caractères spéciaux (laisser seulement a-z et A-Z)
        text = re.sub(r'[^A-Za-z\s]', ' ', text)

        # Séparer sur les espaces
        tokens = [t for t in text.split() if len(t) > 0]

        return tokens

    def process_tokens(self, tokens):
        """
        Transforme les tokens en terms prêts pour l'index.
        - Case folding
        - Suppression des stop-words
        - Stemming
        """
        # Case folding
        tokens = [t.lower() for t in tokens]

        # stop words
        if self.stop_word_active:
            tokens = [token for token in tokens if token not in self.stop_words_set]

        # stemming
        if self.stemmer_active:
            tokens = [self.stemmer.stem(token, 0, len(token) - 1) for token in tokens]

        return tokens

    def read_file(self, filename, is_zipped):
        """Renvoie le contenu d'un fichier zippé ou non"""
        if is_zipped:
            try:
                with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
            except Exception as e:
                print(f"- Erreur lecture: {e}")
                return None
        else:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

        return content

    def build_index(self, filename, is_zipped=False):
        """Construit l'index depuis le fichier"""
        start_time = time.time()

        print(f"Lecture de {filename}...")
        content = self.read_file(filename, is_zipped)

        # Extraction des documents
        doc_pattern = r'<doc><docno>([^<]+)</docno>(.*?)</doc>'
        matches = re.findall(doc_pattern, content, re.DOTALL)

        print(f"Indexation de {len(matches)} documents...")

        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()

            # Tokenisation SIMPLE (sans traitement supplémentaire)
            tokens = self.apply_tokenization(doc_text)

            # Mise à jour des statistiques pour les TOKENS
            self.total_tokens_bp += len(tokens)
            self.distinct_tokens_bp.update(tokens)
            self.total_chars_tokens += sum(len(token) for token in tokens)

            # Traitement TOKENS (case folding, stop-words si TRUE, stemming si TRUE) pour obtenir les TERMS
            terms = self.process_tokens(tokens)

            # Mise à jour des statistiques TERMS (après traitement)
            doc_length = len(terms)
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = doc_length
            self.total_terms += doc_length

            # Construction du dictionnaire
            term_freq = Counter(terms)
            for term, freq in term_freq.items():
                self.dictionary[term][doc_id] = freq

        self.doc_count = len(self.doc_ids)
        self.avg_doc_length = self.total_terms / self.doc_count if self.doc_count > 0 else 0

        end_time = time.time()
        indexing_time = end_time - start_time

        print(f"Index construit avec succès !")

        return indexing_time

    def get_collection_statistics(self, indexing_time):
        """Calcule TOUTES les statistiques demandées dans l'exercice 1"""

        # Statistiques pour TOKENS (avant traitement)
        total_tokens = self.total_tokens_bp
        distinct_tokens = len(self.distinct_tokens_bp)

        # Correction : calcul de la longueur moyenne des tokens selon les tokens distincts
        avg_token_length = (
            sum(len(token) for token in self.distinct_tokens_bp) / distinct_tokens
            if distinct_tokens > 0 else 0
        )

        # Statistiques pour TERMS (après traitement)
        total_terms = self.total_terms
        distinct_terms = len(self.dictionary)
        total_chars_terms = sum(len(term) for term in self.dictionary.keys())
        avg_term_length = total_chars_terms / distinct_terms if distinct_terms > 0 else 0

        # Longueur moyenne des documents
        avg_doc_length = self.avg_doc_length

        return {
            'indexing_time': indexing_time,
            'total_tokens': total_tokens,
            'distinct_tokens': distinct_tokens,
            'avg_token_length': avg_token_length,
            'total_terms': total_terms,
            'distinct_terms': distinct_terms,
            'avg_doc_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }
