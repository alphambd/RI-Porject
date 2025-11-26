import re
from collections import defaultdict, Counter
import gzip
import time
from porterstemmer import PorterStemmer
from snowballstemmer import stem_word

class WeightedInvertedIndex:
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

        # Configuration simplifiée - SUPPRESSION des attributs de compatibilité
        self.stop_words_set = set()
        self.stemmer_func = None
        self.tokenization_method = "basic"
        self.stop_list_name = "nostop"
        self.stemmer_name = "nostem"
        # SUPPRIMÉ: stop_word_active et stemmer_active

    def get_doc_ids(self):
        """Retourne la liste des IDs de documents"""
        return self.doc_ids
    # === FONCTIONS DE TOKENIZATION ===
    
    def _tokenize_basic(self, text):
        """Tokenization basique: seulement lettres"""
        text = re.sub(r'[^A-Za-z\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]
    
    def _tokenize_extended(self, text):
        """Tokenization étendue: lettres et chiffres"""
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]
    
    def _tokenize_hyphen(self, text):
        """Tokenization qui garde les traits d'union"""
        text = re.sub(r'[^A-Za-z\-\s]', ' ', text)
        tokens = []
        for token in text.split():
            if '-' in token and len(token) > 1:
                tokens.extend([token] + token.split('-'))
            else:
                tokens.append(token)
        return [t for t in tokens if len(t) > 0]
    
    def _tokenize_apostrophe(self, text):
        """Tokenization qui garde les apostrophes"""
        text = re.sub(r'[^A-Za-z\'\s]', ' ', text)
        return [t for t in text.split() if len(t) > 0]

    def apply_tokenization(self, text):
        """Applique la tokenization configurée"""
        methods = {
            "basic": self._tokenize_basic,
            "extended": self._tokenize_extended, 
            "hyphen": self._tokenize_hyphen,
            "apostrophe": self._tokenize_apostrophe
        }
        tokenizer = methods.get(self.tokenization_method, self._tokenize_basic)
        return tokenizer(text)

    # === FONCTIONS DE STEMMING ===
    
    def configure_stemmer(self, stemmer_name="nostem"):
        """Configure l'algorithme de stemming - VERSION SIMPLIFIÉE"""
        self.stemmer_name = stemmer_name
        
        if stemmer_name == "nostem":
            self.stemmer_func = None
        elif stemmer_name == "porter":
            stemmer = PorterStemmer()
            self.stemmer_func = lambda word: stemmer.stem(word, 0, len(word) - 1)
        elif stemmer_name == "snowball":
            self.stemmer_func = stem_word
        else:
            print(f"Stemmer '{stemmer_name}' non supporté, utilisation de 'nostem'")
            self.stemmer_func = None
        print(f"- Stemmer configuré: {stemmer_name}")

    # === FONCTIONS DE STOP-WORDS ===
    
    def _load_stop_words(self, stop_list_name="stop671"):
        """Charge différentes listes de stop-words"""
        stop_files = {
            "stop635": "data/stop-words-english1.txt",
            "stop174": "data/stop-words-english2.txt",
            "stop32": "data/stop-words-english3-google.txt", 
            "stop671": "data/stop-words-english4.txt",
            
        }
        # TO HANDLE
        file_path = stop_files.get(stop_list_name, "data/stop-words-english4.txt")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"- {len(self.stop_words_set)} stop words chargés depuis {stop_list_name}")
        except FileNotFoundError:
            print(f"- Fichier {file_path} non trouvé, utilisation liste vide")
            self.stop_words_set = set()

    def configure_stop_words(self, stop_list_name="nostop"):
        """Configure la liste de stop-words"""
        self.stop_list_name = stop_list_name
        if stop_list_name != "nostop":
            self._load_stop_words(stop_list_name)
        print(f"- Stop-words configurés: {stop_list_name}")

    def configure_tokenization(self, method="basic"):
        """Configure la méthode de tokenization"""
        self.tokenization_method = method
        print(f"- Méthode de tokenization configurée: {method}")

    def process_tokens(self, tokens):
        """Transforme les tokens en terms avec la configuration actuelle"""
        # Case folding
        tokens = [t.lower() for t in tokens]

        # Stop words
        if self.stop_list_name != "nostop":
            tokens = [token for token in tokens if token not in self.stop_words_set]

        # Stemming
        if self.stemmer_func:
            tokens = [self.stemmer_func(token) for token in tokens]

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

            # Tokenization avec la méthode configurée
            tokens = self.apply_tokenization(doc_text)

            # Mise à jour des statistiques pour les TOKENS
            self.total_tokens_bp += len(tokens)
            self.distinct_tokens_bp.update(tokens)
            self.total_chars_tokens += sum(len(token) for token in tokens)

            # Traitement TOKENS pour obtenir les TERMS
            terms = self.process_tokens(tokens)

            # Mise à jour des statistiques TERMS
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

        print(f"Index construit avec succès !\n")
        return indexing_time

    def get_collection_statistics(self, indexing_time):
        """Calcule TOUTES les statistiques demandées"""
        total_tokens = self.total_tokens_bp
        distinct_tokens = len(self.distinct_tokens_bp)

        avg_token_length = (
            sum(len(token) for token in self.distinct_tokens_bp) / distinct_tokens
            if distinct_tokens > 0 else 0
        )

        total_terms = self.total_terms
        distinct_terms = len(self.dictionary)
        total_chars_terms = sum(len(term) for term in self.dictionary.keys())
        avg_term_length = total_chars_terms / distinct_terms if distinct_terms > 0 else 0

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