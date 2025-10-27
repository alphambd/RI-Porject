import re  # for regex operations
from collections import defaultdict, Counter  # for dictionary and counting
import string
from practice2.portestemmer import PorterStemmer   # ton fichier PorterStemmer

class InvertedIndex:
    def __init__(self, stop_words_file=None):
        # Dictionnaire qui stocke pour chaque mot la liste des documents où il apparaît : term -> {doc_id: tf}
        self.dictionary = defaultdict(dict)
        # Liste de tous les identifiants de documents
        self.doc_ids = []

        # Options stop-words et stemmer
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words = set()

        # Charger stop-words si fichier fourni
        if stop_words_file:
            try:
                with open(stop_words_file, 'r', encoding='utf-8') as f:
                    self.stop_words = set(line.strip() for line in f)
                self.stop_word_active = True
            except FileNotFoundError:
                print("⚠️ Fichier stop-words non trouvé. Stop-words désactivés.")

        # Créer le stemmer
        self.stemmer = PorterStemmer()
        self.stemmer_active = True  # actif pour Exercice 2

    def preprocess_text(self, text):
        """Traitement basique du texte avec stop-words et stemming"""
        # Convertir en minuscules
        text = text.lower()
        # Traiter les apostrophes
        text = text.replace("’", "'")
        text = re.sub(r"'s?\b", "", text)
        # Supprimer la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenisation
        tokens = text.split()

        # Supprimer stop-words si activé
        if self.stop_word_active:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Appliquer le stemmer si activé
        if self.stemmer_active:
            tokens = [self.stemmer.stem(t, 0, len(t)-1) for t in tokens]

        return tokens

    def add_document(self, doc_id, text):
        """Ajouter un document à l'index"""
        tokens = self.preprocess_text(text)
        # Compter les fréquences des termes
        term_freq = Counter(tokens)

        # Mettre à jour le dictionnaire
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq

        # Ajouter l'identifiant du document
        self.doc_ids.append(doc_id)

    def build_from_file(self, filename):
        """Construire l'index à partir d'un fichier"""
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extraire les documents
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)

        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.add_document(doc_id, doc_text)

    def display_index(self, with_tf=False):
        """Afficher l'index inversé"""
        sorted_terms = sorted(self.dictionary.keys())

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

    def get_postings(self, term):
        term = term.lower()
        if term in self.dictionary:
            return sorted(self.dictionary[term].keys())
        return []

    def get_document_frequency(self, term):
        term = term.lower()
        return len(self.dictionary.get(term, {}))

    def print_dictionary(self):
        sorted_terms = sorted(self.dictionary.keys())
        for term in sorted_terms:
            print(f"{term}: {self.dictionary[term]}")
