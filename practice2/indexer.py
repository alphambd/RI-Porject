import re # for regex operations
from collections import defaultdict, Counter # for dictionary and counting
import string
import gzip

from practice2.portestemmer import PorterStemmer


class InvertedIndex:
    def __init__(self):
        # Dictionnaire qui stocke pour chaque mot la liste des documents où il apparaît : term -> {doc_id: tf}
        self.dictionary = defaultdict(dict)
        # Dictionnaire qui stocke pour chaque fichier la liste des ids des documents
        self.fils_doc_count_terms = defaultdict(list)
        self.fils_doc_count_words = defaultdict(list)
        self.fils_doc_count_chars = defaultdict(list)
        self.fils_doc_ids = defaultdict(list)
        self.stop_word_active = False
        self.stemmer_active = False

    def preprocess_text(self, text):
        """Traitement basique du texte"""
        # Convertir en minuscules
        text = text.lower()
        # Traiter le cas des apostrophes (pour schindler's, singin', etc.)
        text = text.replace("’", "'")
        text = re.sub(r"[^a-z\s]", "", text)
        # Tokenisation
        tokens = text.split()
        return tokens

    def add_document(self, doc_id, text, filename):
        """Ajouter un document à l'index"""
        if self.stemmer_active:
            p = PorterStemmer()
            output = ''
            word = ''
            for c in text:
                if c.isalpha():
                    word += c.lower()
                else:
                    if word:
                        output += p.stem(word, 0, len(word) - 1)
                        word = ''
                    output += c.lower()
            text = output

        if self.stop_word_active:
            with open("stop-words-english4.txt", 'r') as file:
                stopwords = file.read().split()
            tokens = [token for token in self.preprocess_text(text) if token in stopwords]
        else :
            tokens =  self.preprocess_text(text)

        self.fils_doc_count_words[filename] += len(tokens)
        self.fils_doc_count_chars[filename] += len("".join(tokens))
        # Compter les fréquences des termes
        term_freq = Counter(tokens)
        self.fils_doc_count_terms[filename] += len(term_freq)

        # Mettre à jour le dictionnaire
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq

    def build_from_file(self, filename):
        """Construire l'index à partir d'un fichier"""
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

        self.fils_doc_count_words[filename] = 0
        self.fils_doc_count_chars[filename] = 0
        self.fils_doc_count_terms[filename] = 0
        self.fils_doc_ids[filename] = []

        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.fils_doc_ids[filename].append(doc_id)
            self.add_document(doc_id, doc_text, filename)

    def display_index(self, with_tf=False):
        """Afficher l'index inversé"""
        # Trier les termes alphabétiquement
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

    def get_data(self, filename):
        return (self.fils_doc_count_terms[filename], self.fils_doc_count_words[filename], self.fils_doc_count_chars[filename],self.fils_doc_ids[filename])