import re # for regex operations
from collections import defaultdict, Counter # for dictionary and counting
import gzip
import time

#from practice2.portestemmer import PorterStemmer
from portestemmer import PorterStemmer


class InvertedIndex:
    def __init__(self):
        # Dictionnaire principal : term -> {doc_id: tf}
        self.dictionary = defaultdict(dict)
        # Liste des identifiants de documents
        self.doc_ids = []

        # Statistiques par fichier
        self.fils_doc_count_terms = defaultdict(list)
        self.fils_doc_count_words = defaultdict(list)
        self.fils_doc_count_chars = defaultdict(list)
        self.fils_doc_ids = defaultdict(list)

        # Options d'indexation
        self.stop_word_active = False
        self.stemmer_active = False

    def reset(self):
        """Réinitialise complètement l'index"""
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.fils_doc_count_terms = defaultdict(list)
        self.fils_doc_count_words = defaultdict(list)
        self.fils_doc_count_chars = defaultdict(list)
        self.fils_doc_ids = defaultdict(list)
        self.stop_word_active = False
        self.stemmer_active = False

    def avg_document_length(self):
        """Retourne la longueur moyenne des documents en nombre de mots"""
        total_words = sum(len(self.dictionary[term]) for term in self.dictionary)
        total_docs = len(self.doc_ids)
        return total_words / total_docs if total_docs > 0 else 0

    def avg_term_length(self):
        """Retourne la longueur moyenne des termes en nombre de caractères"""
        total_chars = sum(len(term) for term in self.dictionary)
        total_terms = len(self.dictionary)
        return total_chars / total_terms if total_terms > 0 else 0
    
    def get_vocabulary_size(self):
        """Retourne le nombre de termes uniques dans TOUTE la collection"""
        return len(self.dictionary)

    def preprocess_text(self, text):
        """Traitement basique du texte"""
        # Convertir en minuscules
        text = text.lower()
        # Traiter le cas des apostrophes (pour schindler's, singin', etc.)
        # ^[...] : ^ indique le début d'un ensemble de caractères (...)
        # [^...] : ^ indique la négation (tout sauf ...)
        text = text.replace("’", "'")
        text = re.sub(r"[^a-z\s]", "", text)
        # Tokenisation
        tokens = text.split()
        return tokens

    def add_document(self, doc_id, text, filename=None):
        """
        Ajoute un document à l'index inversé.
        Applique selon les paramètres :
        - suppression des stop words (si self.stop_word_active = True)
        - stemming via le PorterStemmer (si self.stemmer_active = True)
        Met ensuite à jour le dictionnaire et les statistiques globales.
        """

        # Appliquer le stemming si activé
        if self.stemmer_active:
            p = PorterStemmer()
            output, word = '', ''
            for c in text:
                if c.isalpha(): # si c'est une lettre unicode
                    word += c.lower()
                else: # si c'est un séparateur
                    if word: # si on a un mot à traiter
                        output += p.stem(word, 0, len(word) - 1)
                        word = ''
                    # garder le séparateur (espace, ponctuation, etc.)
                    output += c.lower()
            text = output

        # Prétraitement du texte
        tokens = self.preprocess_text(text)

        # Suppression des stop words si activée
        if self.stop_word_active:
            try:
                # Lecture du fichier de stop words une seule fois serait préférable 
                with open("stop-words-english4.txt", 'r') as file:
                    stopwords = set(file.read().split())  # ensemble = plus rapide que liste
            except FileNotFoundError:
                stopwords = set()
                print("⚠️  Fichier de stop words non trouvé. Aucun mot supprimé.")

            # On ne garde que les tokens qui NE sont PAS des stop words
            tokens = [token for token in tokens if token not in stopwords]
        
        # Calcul de la fréquence des termes
        term_freq = Counter(tokens)

        # Mise à jour des statistiques du fichier
        if filename:
            self.fils_doc_ids[filename].append(doc_id)
            self.fils_doc_count_words[filename] += len(tokens)
            self.fils_doc_count_chars[filename] += len("".join(tokens)) # sans espaces     
            self.fils_doc_count_terms[filename] += len(term_freq)

        # Mise à jour du dictionnaire inversé
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
    
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

    def build_from_file(self, filename, print_index=False):
        """Construire l'index à partir d'un fichier"""
        start_time = time.time()
        with gzip.open(filename, 'rb') as file:
            content = file.read()
        content = content.decode('utf-8')
        # Extraire le contenu des documents dans le fichier avec regex
        """
        le premier ([^<]+) : capture le contenu de <docno>...</docno> (ex : D0)
        le deuxième ([^<]+) : capture le contenu texte du document (ex: Citizen Kane)
        """
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)

        # Initialiser les statistiques pour ce fichier
        self.fils_doc_count_words[filename] = 0
        self.fils_doc_count_chars[filename] = 0
        self.fils_doc_count_terms[filename] = 0
        self.fils_doc_ids[filename] = []
        
        # Pour chaque document trouvé, on l'ajoute à l'index
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.fils_doc_ids[filename].append(doc_id)
            self.add_document(doc_id, doc_text, filename)

        end_time = time.time()
        indexing_time = end_time - start_time

        # Affichage de l'index si la collection est petite
        if print_index:  
            if len(matches) <= 50:
                print(f"📄 Index pour {filename} :")
                self.display_index(limit=20)
        else:
            print(f"  {filename}: {len(matches)} documents indexés en {indexing_time:.2f}s")
        
        return indexing_time
    
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
    
    def get_postings(self, term):
        """Récupérer la liste de postings pour un terme"""
        term = term.lower()
        if term in self.dictionary:
            return sorted(self.dictionary[term].keys())
        return []
    
    def get_document_frequency(self, term):
        """Récupérer la fréquence de document pour un terme"""
        term = term.lower()
        return len(self.dictionary.get(term, {}))
    
    def print_dictionary(self):
        """Afficher le dictionnaire"""
        sorted_terms = sorted(self.dictionary.keys())
        for term in sorted_terms:
            print(f"{term}: {self.dictionary[term]}")

    def print_dictionary_with_size(self, size_limit=0):
        """Afficher le dictionnaire avec une limite de taille"""
        if size_limit > 0 and size_limit < len(self.dictionary):
            sorted_terms = sorted(self.dictionary.keys())[:size_limit]
        else:
            sorted_terms = sorted(self.dictionary.keys())
        for term in sorted_terms:
            print(f"{term}: {self.dictionary[term]}")

    def get_data(self, filename):
        """Retourne les statistiques du fichier donné"""
        return (self.fils_doc_count_terms[filename], self.fils_doc_count_words[filename], self.fils_doc_count_chars[filename],self.fils_doc_ids[filename])