import math
from collections import defaultdict

class RankedRetrieval:
    def __init__(self, index):
        self.index = index
        self.doc_count = index.doc_count
        self.avg_dl = index.avg_doc_length
        
        # Pré-calculer df pour tous les termes
        self.df = {}
        for term, doc_dict in index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        # Cache optionnel pour les normes (vide au début)
        self._doc_norms_cache = {}
    
    def _compute_document_norm(self, doc_id):
        """Calcule la norme cosinus d'un document spécifique"""
        sum_of_squares = 0.0
        
        # Parcourir tous les termes de ce document
        for term, doc_dict in self.index.dictionary.items():
            if doc_id in doc_dict:
                tf = doc_dict[doc_id]
                df = self.df[term]
                
                # Calcul du poids brut pour chaque terme
                w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
                w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
                raw_weight = w_tf * w_idf
                
                sum_of_squares += raw_weight ** 2
        
        norm = math.sqrt(sum_of_squares) if sum_of_squares > 0 else 1.0
        return norm

    def smart_ltn_weighting(self, term, doc_id):
        """SMART ltn weighting: logarithmic tf, idf, pad de normalization"""
        # ltn: (1 + log(tf)) * log(N/df)

        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0

        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
         
        w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
        w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        
        return w_tf * w_idf

    def smart_ltc_weighting(self, term, doc_id, use_cache=True):
        """SMART ltc weighting: logarithmic tf, idf, normalization cosinus """
        # ltn_values = 1 + log(tf)) * log(N/df)
        # ltc: ltn_values / sqrt(sum_of_squares(ltn_values)) 

        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        
        # Calcul du poids brut (identique à ltn)
        w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
        w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        raw_weight = w_tf * w_idf
        
        # Calcul de la norme avec cache optionnel
        if use_cache:
            if doc_id not in self._doc_norms_cache:
                self._doc_norms_cache[doc_id] = self._compute_document_norm(doc_id)
            doc_norm = self._doc_norms_cache[doc_id]
        else:
            doc_norm = self._compute_document_norm(doc_id)
        
        return raw_weight / doc_norm if doc_norm > 0 else 0.0
    
    def bm25_weighting(self, term, doc_id, k1=1.2, b=0.75):
        """BM25 weighting avec paramètres standard"""
        # BM25: log((N - df + 0.5) / (df + 0.5) + 1) * [ (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl))) ]
        # - N = nombre total de documents
        # - dl = longueur du document (nombre de termes)
        # - avgdl = longueur moyenne des documents
        # - k1 = paramètre de saturation TF (valeur par défaut: 1.2)
        # - b = paramètre de normalisation longueur (valeur par défaut: 0.75)

        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        doc_length = self.index.doc_lengths[doc_id]
        
        # Calcul BM25
        idf = math.log10((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / self.avg_dl)))
        
        return idf * tf_component
    
    def process_query_terms(self, query):
        """Traiter la requete pour extraire les termes"""
        tokens = self.index.apply_tokenization(query)
        tokens = self.index.process_tokens(tokens)
        return list(set(tokens)) # termes uniques

    def search_query(self, query, weighting_scheme="ltn", top_k=10):
        """Recherche une requête avec le schéma de pondération spécifié"""
        # Traitement de la requête
        query_terms = self.process_query_terms(query) 
        
        print(f" * Recherche: '{query}' -> termes: {query_terms}")
        
        # Calcul des scores pour tous les documents
        doc_scores = defaultdict(float)
        
        for doc_id in self.index.doc_ids:
            score = 0.0
            for term in query_terms:
                if weighting_scheme == "ltn":
                    term_weight = self.smart_ltn_weighting(term, doc_id)
                elif weighting_scheme == "ltc":
                    term_weight = self.smart_ltc_weighting(term, doc_id)
                elif weighting_scheme == "bm25":
                    term_weight = self.bm25_weighting(term, doc_id)
                else:
                    term_weight = self.smart_ltn_weighting(term, doc_id)
                
                score += term_weight
            
            if score > 0:
                doc_scores[doc_id] = score
        
        # Tri par score décroissant
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]
    
    def get_term_weight(self, term, doc_id, weighting_scheme):
        """Retourne le poids d'un terme spécifique dans un document"""
        if weighting_scheme == "ltn":
            return self.smart_ltn_weighting(term, doc_id)
        elif weighting_scheme == "ltc":
            return self.smart_ltc_weighting(term, doc_id)
        elif weighting_scheme == "bm25":
            return self.bm25_weighting(term, doc_id)
        else:
            print(f" - Erreur : option de weighting '{weighting_scheme}' invalide...")
            return 0.0