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
    
    def smart_ltn_weighting(self, term, doc_id):
        """SMART ltn weighting: logarithmic tf, idf, no normalization"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]  # tf raw
        df = self.df[term]
        
        # ltn: (1 + log(tf)) * log(N/df) - PAS de normalisation
        w_tf = 1.0 + math.log(tf) if tf > 0 else 0.0
        w_idf = math.log(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        
        return w_tf * w_idf
    
    def smart_ltc_weighting(self, term, doc_id):
        """SMART ltc weighting: logarithmic tf, idf, cosine normalization"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        doc_length = self.index.doc_lengths[doc_id]
        
        # ltc: [(1 + log(tf)) * log(N/df)] / sqrt(sum_of_squares) - normalisation cosinus complète
        w_tf = 1.0 + math.log(tf) if tf > 0 else 0.0
        w_idf = math.log(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        
        # Pour la normalisation cosinus complète, on aurait besoin de pré-calculer les normes des documents
        # Ici simplification: normalisation par longueur
        raw_score = w_tf * w_idf
        normalized_score = raw_score / doc_length if doc_length > 0 else 0.0
        
        return normalized_score
    
    def bm25_weighting(self, term, doc_id, k1=1.2, b=0.75):
        """BM25 weighting avec paramètres standard"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        doc_length = self.index.doc_lengths[doc_id]
        
        # Calcul BM25
        idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
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
            return 0.0