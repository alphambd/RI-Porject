import re
from collections import defaultdict, Counter
import gzip
import os
import time
from portestemmer import PorterStemmer
import string

class AdvancedInvertedIndex:
    def __init__(self):
        # Structure principale
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}  # Stocke le texte des documents pour statistiques
        
        # Statistiques globales
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        
        # Options de traitement
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words_set = set()
        
        # Résultats par fichier pour les graphiques
        self.file_statistics = {}
    
    def reset(self):
        """Réinitialise complètement l'index - CRITIQUE pour les comparaisons"""
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        self.file_statistics = {}
    
    def load_stop_words(self, stop_words_file="stop-words-english4.txt"):
        """Charge la liste des stop words depuis un fichier"""
        try:
            with open(stop_words_file, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"✅ {len(self.stop_words_set)} stop words chargés")
        except FileNotFoundError:
            print(f"⚠️  Fichier {stop_words_file} non trouvé")
            self.stop_words_set = set()
    
    def preprocess_text(self, text):
        # Tokenisation et normalisation avancée
        # Conversion minuscules
        text = text.lower()
        
        # Nettoyage des caractères spéciaux (conserve les apostrophes pour le stemming)
        text = re.sub(r"[^a-z'\s]", " ", text)
        
        # Tokenisation
        tokens = text.split()
        
        return tokens
    """
    def preprocess_text(self, text):
        # Traitement basique du texte
        # Convertir en minuscules
        text = text.lower()
        # Traiter le cas des apostrophes (pour schindler's, singin', etc.)
        text = text.replace("’", "'")
        text = re.sub(r"'s?\b", "", text)
        # Supprimer la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenisation
        tokens = text.split()
        return tokens
    """
    def apply_stemming(self, text):
        """Applique le stemming de Porter sur un texte"""
        if not self.stemmer_active:
            return text
        
        p = PorterStemmer()
        output = ''
        word = ''
        
        for char in text:
            if char.isalpha():
                word += char
            else:
                if word:
                    # Appliquer le stemming
                    stemmed_word = p.stem(word, 0, len(word) - 1)
                    output += stemmed_word
                    word = ''
                output += char
        
        # Traiter le dernier mot si nécessaire
        if word:
            output += p.stem(word, 0, len(word) - 1)
        
        return output
    
    def add_document(self, doc_id, text, filename=None):
        """Ajoute un document avec tous les traitements"""
        # Stocker le document original pour statistiques
        self.all_documents[doc_id] = text
        self.doc_ids.append(doc_id)
        
        # Appliquer le stemming si activé
        if self.stemmer_active:
            text = self.apply_stemming(text)
        
        # Prétraitement de base
        tokens = self.preprocess_text(text)
        
        # Filtrer les stop words si activé
        if self.stop_word_active and self.stop_words_set:
            tokens = [token for token in tokens if token not in self.stop_words_set]
        
        # Mise à jour des statistiques globales
        self.total_tokens += len(tokens)
        self.total_chars += sum(len(token) for token in tokens)
        self.document_lengths.append(len(tokens))
        
        # Calcul des fréquences des termes
        term_freq = Counter(tokens)
        
        # Mise à jour du dictionnaire inversé
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
        
        # Statistiques par fichier
        if filename:
            if filename not in self.file_statistics:
                self.file_statistics[filename] = {
                    'documents': 0,
                    'tokens': 0,
                    'chars': 0,
                    'vocabulary': set()
                }
            
            self.file_statistics[filename]['documents'] += 1
            self.file_statistics[filename]['tokens'] += len(tokens)
            self.file_statistics[filename]['chars'] += sum(len(token) for token in tokens)
            self.file_statistics[filename]['vocabulary'].update(term_freq.keys())
    
    def build_from_file(self, filename, verbose=False, print_index=False):
        """Construit l'index depuis un fichier avec options de contrôle"""
        start_time = time.time()
        
        try:
            # Lecture du fichier compressé
            with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        except Exception as e:
            print(f"❌ Erreur lecture {filename}: {e}")
            return None
        
        # Extraction des documents
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)
        
        if not matches:
            print(f"⚠️  Aucun document trouvé dans {filename}")
            return None
        
        # Indexation de chaque document
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.add_document(doc_id, doc_text, filename)
        
        end_time = time.time()
        indexing_time = end_time - start_time
        
        # Affichage contrôlé (variants de l'exercice 1)
        if verbose:
            print(f"📊 {filename}: {len(matches)} documents indexés en {indexing_time:.2f}s")
        
        if print_index and len(matches) <= 50:  # Seulement pour petites collections
            self.display_index(limit=20)
        
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
        
        avg_doc_length = self.total_tokens / len(self.doc_ids)
        avg_term_length = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        vocabulary_size = len(self.dictionary)
        
        return {
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length,
            'vocabulary_size': vocabulary_size,
            'total_documents': len(self.doc_ids),
            'total_tokens': self.total_tokens
        }
    
    def get_file_statistics(self, filename):
        """Retourne les statistiques pour un fichier spécifique"""
        if filename not in self.file_statistics:
            return None
        
        stats = self.file_statistics[filename]
        avg_doc_length = stats['tokens'] / stats['documents'] if stats['documents'] > 0 else 0
        avg_term_length = stats['chars'] / stats['tokens'] if stats['tokens'] > 0 else 0
        
        return {
            'documents': stats['documents'],
            'tokens': stats['tokens'],
            'vocabulary_size': len(stats['vocabulary']),
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }
    
    def display_index(self, limit=0, with_tf=False):
        """Affiche l'index avec contrôle (variants exercice 1)"""
        sorted_terms = sorted(self.dictionary.keys())
        
        if limit > 0:
            sorted_terms = sorted_terms[:limit]
            print(f"📄 Affichage des {limit} premiers termes...")
        
        for term in sorted_terms:
            postings = self.dictionary[term]
            df = len(postings)
            
            if with_tf:
                print(f"{df}=df({term})")
                for doc_id, tf in sorted(postings.items()):
                    print(f"    {tf} {doc_id}")
            else:
                print(f"{df}=df({term})")
                for doc_id in sorted(postings.keys()):
                    print(f"    {doc_id}")
    
    # Méthodes utilitaires
    def get_postings(self, term):
        term = term.lower()
        return sorted(self.dictionary.get(term, {}).keys())
    
    def get_document_frequency(self, term):
        term = term.lower()
        return len(self.dictionary.get(term, {}))
    
    def get_vocabulary(self):
        return set(self.dictionary.keys())