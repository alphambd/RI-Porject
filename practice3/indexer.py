import re  # pour les expressions régulières (rechercher / nettoyer du texte)
from collections import defaultdict, Counter  # pour gérer les dictionnaires et compter les mots
import string
from practice2.portestemmer import PorterStemmer   # importer le stemmer (racine des mots)

class InvertedIndex:
    def __init__(self, stop_words_file=None):
        # Dictionnaire principal : chaque mot pointe vers les documents où il apparaît avec sa fréquence
        self.dictionary = defaultdict(dict)
        # Liste de tous les documents traités
        self.doc_ids = []

        # Options : activer ou non les stop-words et le stemming
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words = set()

        # Charger les stop-words (mots à ignorer) si un fichier est donné
        if stop_words_file:
            try:
                with open(stop_words_file, 'r', encoding='utf-8') as f:
                    # On stocke tous les stop-words dans un ensemble
                    self.stop_words = set(line.strip() for line in f)
                self.stop_word_active = True  # activer la suppression des stop-words
            except FileNotFoundError:
                print("⚠️ Fichier stop-words non trouvé. Stop-words désactivés.")

        # Créer et activer le stemmer (pour réduire les mots à leur racine)
        self.stemmer = PorterStemmer()
        self.stemmer_active = True  # actif pour l'exercice 2

    def preprocess_text(self, text):
        """Nettoyer le texte et appliquer stop-words + stemming"""
        # Tout mettre en minuscules
        text = text.lower()
        # Gérer les apostrophes (ex: l'homme -> lhomme)
        text = text.replace("’", "'")
        text = re.sub(r"'s?\b", "", text)
        # Supprimer la ponctuation (.,!?: etc.)
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Découper le texte en mots (tokens)
        tokens = text.split()

        # Supprimer les stop-words si activé
        if self.stop_word_active:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Appliquer le stemmer si activé (réduit les mots à leur racine)
        if self.stemmer_active:
            tokens = [self.stemmer.stem(t, 0, len(t)-1) for t in tokens]

        return tokens

    def add_document(self, doc_id, text):
        """Ajouter un document à l'index inversé"""
        # Prétraiter le texte (tokenisation, nettoyage, etc.)
        tokens = self.preprocess_text(text)
        # Compter la fréquence de chaque mot dans le document
        term_freq = Counter(tokens)

        # Mettre à jour le dictionnaire principal avec les fréquences
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq

        # Ajouter l'identifiant du document dans la liste
        self.doc_ids.append(doc_id)

    def build_from_file(self, filename):
        """Lire le fichier et construire l'index complet"""
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extraire les documents grâce à une expression régulière
        # <doc><docno>ID</docno>contenu</doc>
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)

        # Pour chaque document trouvé, on l'ajoute à l'index
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.add_document(doc_id, doc_text)

    def display_index(self, with_tf=False):
        """Afficher le contenu de l'index inversé"""
        # Trier les termes alphabétiquement pour un affichage clair
        sorted_terms = sorted(self.dictionary.keys())

        for term in sorted_terms:
            postings = self.dictionary[term]  # documents contenant le terme
            df = len(postings)  # nombre de documents où le terme apparaît

            if with_tf:
                # Afficher avec fréquence du terme (tf)
                print(f"{df}=df({term})")
                for doc_id, tf in sorted(postings.items()):
                    print(f"    {tf} {doc_id}")
            else:
                # Afficher seulement les documents
                print(f"{df}=df({term})")
                for doc_id in sorted(postings.keys()):
                    print(f"    {doc_id}")

    def get_postings(self, term):
        """Retourner la liste des documents contenant un mot"""
        term = term.lower()
        if term in self.dictionary:
            return sorted(self.dictionary[term].keys())
        return []

    def get_document_frequency(self, term):
        """Retourner combien de documents contiennent ce mot"""
        term = term.lower()
        return len(self.dictionary.get(term, {}))

    def print_dictionary(self):
        """Afficher tout le dictionnaire complet"""
        sorted_terms = sorted(self.dictionary.keys())
        for term in sorted_terms:
            print(f"{term}: {self.dictionary[term]}")
