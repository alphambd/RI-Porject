import math
import time
import os
from collections import defaultdict
import re


class SMARTLTNIndex:
    def __init__(self):
        # Dictionnaire pour stocker les documents {doc_id: {terme: fréquence}}
        self.documents = {}
        # Statistiques des termes {terme: fréquence de document}
        self.term_stats = {}
        # Nombre total de documents
        self.N = 0
        # Ensemble de tous les termes uniques dans la collection
        self.vocabulary = set()

    def add_document(self, doc_id, terms):
        """Ajouter un document à l'index"""
        # Dictionnaire pour compter la fréquence des termes dans ce document
        term_freq = defaultdict(int)
        for term in terms:
            if term:  # Ignorer les termes vides
                term_freq[term] += 1
                self.vocabulary.add(term)

        # Stocker les fréquences des termes pour ce document
        self.documents[doc_id] = dict(term_freq)
        # Mettre à jour le nombre total de documents
        self.N = len(self.documents)

    def preprocess_text(self, text):
        """Prétraitement basique du texte : minuscules et tokenisation"""
        # Convertir en minuscules
        text = text.lower()
        # Tokenisation : garder seulement les caractères alphanumériques
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        return tokens

    def compute_term_statistics(self):
        """Calculer df_t (fréquence de document) pour chaque terme"""
        df_t = defaultdict(int)

        # Compter dans combien de documents apparaît chaque terme
        for doc_id, term_freq in self.documents.items():
            for term in term_freq.keys():
                df_t[term] += 1

        self.term_stats = df_t

    def ltn_weight(self, tf, df_t):
        """Calculer le poids SMART ltn pour un terme"""
        if tf == 0 or df_t == 0:
            return 0.0

        # l: composante log tf (1 + log(tf))
        log_tf = 1 + math.log(tf)

        # t: composante idf (log(N/df_t))
        idf = math.log(self.N / df_t)

        # n: pas de normalisation
        return log_tf * idf

    def query_weight(self, term):
        """Calculer le poids de requête pour un terme (en utilisant ltn pour la requête aussi)"""
        if term not in self.term_stats:
            return 0.0

        df_t = self.term_stats[term]

        # Pour la requête, on utilise la même formule ltn
        # Comme les termes de requête apparaissent typiquement une fois, tf=1
        log_tf = 1 + math.log(1)  # Ceci égale 1
        idf = math.log(self.N / df_t)

        return log_tf * idf

    def compute_document_weights(self):
        """Calculer les poids ltn pour tous les documents"""
        start_time = time.time()

        doc_weights = {}
        for doc_id, term_freq in self.documents.items():
            weights = {}
            for term, tf in term_freq.items():
                df_t = self.term_stats.get(term, 0)
                weights[term] = self.ltn_weight(tf, df_t)
            doc_weights[doc_id] = weights

        weighting_time = time.time() - start_time
        return doc_weights, weighting_time

    def compute_rsv(self, query_terms, doc_weights):
        """Calculer la valeur RSV (Retrieval Status Value) pour tous les documents"""
        rsv_scores = {}

        # Calculer les poids de la requête
        query_weights = {}
        for term in query_terms:
            query_weights[term] = self.query_weight(term)

        # Calculer RSV pour chaque document
        for doc_id, doc_weights_dict in doc_weights.items():
            rsv = 0.0
            for term in query_terms:
                if term in doc_weights_dict:
                    # RSV = produit scalaire des vecteurs document et requête
                    rsv += doc_weights_dict[term] * query_weights[term]
            rsv_scores[doc_id] = rsv

        return rsv_scores

    def load_documents_from_file(self, file_path):
        """Charger les documents depuis un fichier unique contenant tous les documents"""
        print(f"Chargement des documents depuis {file_path}...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier {file_path} non trouvé")

        # Lire tout le contenu du fichier
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()

        # Séparer les documents basé sur un motif - à adapter selon la structure de votre fichier
        # Motifs communs : IDs de documents, séparateurs, etc.
        documents = self.parse_documents(content)

        # Ajouter chaque document à l'index
        for doc_id, doc_content in documents.items():
            tokens = self.preprocess_text(doc_content)
            self.add_document(doc_id, tokens)

        print(f"Chargement de {len(self.documents)} documents terminé")
        return len(self.documents)

    def parse_documents(self, content):
        """
        Analyser le contenu pour extraire les documents individuels.
        Cette fonction doit être adaptée au format spécifique de votre fichier.
        """
        documents = {}

        # Essayer différents séparateurs de documents communs
        separators = [
            (r'Document\s+(\d+)', 'Document X'),  # Document 12345
            (r'DOCNO\s*:\s*(\d+)', 'DOCNO: X'),  # DOCNO: 12345
            (r'<DOC>\s*<DOCNO>(\d+)</DOCNO>', '<DOC><DOCNO>X</DOCNO>'),  # Style XML
            (r'\n(\d+)\s*\n', 'Nombre sur une ligne'),  # Nombre sur sa propre ligne
        ]

        for pattern, desc in separators:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                print(f"Trouvé {len(matches)} documents en utilisant le motif : {desc}")
                for i, match in enumerate(matches):
                    try:
                        doc_id = int(match.group(1))
                        start_pos = match.end()

                        # Trouver la fin de ce document (début du document suivant ou fin du fichier)
                        if i < len(matches) - 1:
                            end_pos = matches[i + 1].start()
                        else:
                            end_pos = len(content)

                        doc_content = content[start_pos:end_pos].strip()
                        documents[doc_id] = doc_content
                    except ValueError:
                        continue

                if documents:
                    break

        # Si aucun motif spécifique trouvé, traiter tout le contenu comme un seul document
        if not documents:
            print("Aucun séparateur de document trouvé. Traitement du fichier entier comme un seul document.")
            documents[1] = content

        return documents