import re
from collections import defaultdict, Counter
import string

class InvertedIndex:
    def __init__(self):
        """Initialise l’index inversé et la liste des documents"""
        # Dictionnaire : mot -> {doc_id: fréquence dans ce doc}
        self.dictionary = defaultdict(dict)
        # Liste des identifiants de tous les documents
        self.doc_ids = []

    def preprocess_text(self, text):
        """Nettoie le texte : minuscules, suppression ponctuation, découpage en mots"""
        text = text.lower()  # tout en minuscules
        text = text.translate(str.maketrans('', '', string.punctuation))  # supprime la ponctuation
        tokens = text.split()  # découpe en mots
        return tokens

    def add_document(self, doc_id, text):
        """Ajoute un document à l’index inversé"""
        tokens = self.preprocess_text(text)  # nettoie et découpe le texte
        term_freq = Counter(tokens)  # compte la fréquence de chaque mot
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq  # ajoute au dictionnaire

    def build_from_text(self, text):
        """Construit l’index à partir d’un texte complet"""
        # On cherche les documents dans le texte avec <doc><docno>...</docno>...</doc>
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, text)

        # Pour chaque document trouvé, on l'ajoute à l'index
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()  # supprime espaces superflus
            doc_text = doc_text.strip()
            self.doc_ids.append(doc_id)  # garde l'identifiant du doc
            self.add_document(doc_id, doc_text)  # ajoute le doc à l'index

    def display_index(self, with_tf=False):
        """Affiche l’index inversé, avec ou sans fréquence"""
        for term in sorted(self.dictionary.keys()):  # tri par ordre alphabétique
            postings = self.dictionary[term]
            df = len(postings)  # nombre de documents contenant le mot
            print(f"{df}=df({term})")  # affiche la fréquence documentaire
            for doc_id, tf in sorted(postings.items()):
                # si with_tf=True on affiche la fréquence du mot
                print(f"    {tf if with_tf else ''} {doc_id}".strip())

    def get_postings(self, term):
        """Renvoie la liste des documents contenant un mot"""
        term = term.lower()
        return sorted(self.dictionary.get(term, {}).keys())

    def get_document_frequency(self, term):
        """Renvoie combien de documents contiennent le mot"""
        term = term.lower()
        return len(self.dictionary.get(term, {}))
