import math
import pickle
import os
from collections import defaultdict

class RankedRetrieval:
    def __init__(self, index, cache_dir="data/norm_cache"):
        self.index = index
        self.doc_count = index.doc_count
        self.avg_dl = index.avg_doc_length
        self.cache_dir = cache_dir
        
        # Créer le dossier cache s'il n'existe pas
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Pré-calculer df pour tous les termes
        self.df = {}
        for term, doc_dict in index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        # Initialiser le cache des normes cosine (vide au début)
        self._cosine_norms_cache = None
    
    def _get_cosine_norms_cache_filename(self):
        """Génère un nom de fichier de cache basé sur les caractéristiques de l'index"""
        index_hash = hash((
            self.doc_count,
            self.index.total_terms,
            len(self.index.dictionary),
            self.index.stop_word_active,
            self.index.stemmer_active
        ))
        return os.path.join(self.cache_dir, f"cosine_norms_{abs(index_hash)}.pkl")
    
    def _load_or_compute_cosine_norms(self):
        """Charge les normes cosine depuis le cache ou les calcule si nécessaire"""
        # Si déjà chargé, retourner le cache
        if self._cosine_norms_cache is not None:
            return self._cosine_norms_cache
            
        cache_file = self._get_cosine_norms_cache_filename()
        
        # Essayer de charger depuis le cache
        if os.path.exists(cache_file):
            print("Chargement des normes cosine depuis le cache...")
            try:
                with open(cache_file, 'rb') as f:
                    norms = pickle.load(f)
                print("Normes cosine chargées avec succès!")
                self._cosine_norms_cache = norms
                return norms
            except Exception as e:
                print(f"Erreur lors du chargement du cache: {e}")
        
        # Calculer les normes si le cache n'existe pas ou est corrompu
        print("Calcul des normes cosine des documents...")
        norms = self._precompute_all_cosine_norms()
        
        # Sauvegarder dans le cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(norms, f)
            print("Normes cosine sauvegardées dans le cache!")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du cache: {e}")
        
        self._cosine_norms_cache = norms
        return norms

    def _precompute_all_cosine_norms(self):
        """Version optimisée du pré-calcul des normes cosine"""
        doc_norms = {doc_id: 0.0 for doc_id in self.index.doc_ids}
        
        # Parcourir chaque terme une seule fois
        for term, doc_dict in self.index.dictionary.items():
            df = len(doc_dict)
            w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
            
            for doc_id, tf in doc_dict.items():
                w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
                raw_weight = w_tf * w_idf
                doc_norms[doc_id] += raw_weight ** 2
        
        # Prendre la racine carrée
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id]) if doc_norms[doc_id] > 0 else 1.0
        
        print(f"Calcul des normes cosine terminé pour {len(doc_norms)} documents!")
        return doc_norms

    def smart_ltn_weighting(self, term, doc_id):
        """SMART ltn weighting: logarithmic tf, idf, pas de normalization"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0

        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
         
        w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
        w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        
        return w_tf * w_idf

    def smart_ltc_weighting(self, term, doc_id):
        """SMART ltc weighting: logarithmic tf, idf, normalization cosinus"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        # Charger les normes cosine seulement si nécessaire (lazy loading)
        if self._cosine_norms_cache is None:
            self._load_or_compute_cosine_norms()
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        
        w_tf = 1.0 + math.log10(tf) if tf > 0 else 0.0
        w_idf = math.log10(self.doc_count / df) if df > 0 and self.doc_count > df else 0.0
        raw_weight = w_tf * w_idf
        
        # Utilise la norme cosine pré-calculée
        doc_norm = self._cosine_norms_cache.get(doc_id, 1.0)
        return raw_weight / doc_norm if doc_norm > 0 else 0.0
    
    def bm25_weighting(self, term, doc_id, k1=1.2, b=0.75):
        """BM25 weighting avec paramètres standard"""
        if term not in self.index.dictionary or doc_id not in self.index.dictionary[term]:
            return 0.0
        
        tf = self.index.dictionary[term][doc_id]
        df = self.df[term]
        doc_length = self.index.doc_lengths[doc_id]
        
        idf = math.log10((self.doc_count - df + 0.5) / (df + 0.5))
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / self.avg_dl)))
        
        return idf * tf_component
    
    def process_query_terms(self, query):
        """Traiter la requête pour extraire les termes"""
        tokens = self.index.apply_tokenization(query)
        tokens = self.index.process_tokens(tokens)
        #return list(set(tokens))
        return sorted(set(tokens))  # termes uniques triés

    def search_query(self, query, weighting_scheme="ltn", top_k=10):
        """Recherche une requête avec le schéma de pondération spécifié"""
        query_terms = self.process_query_terms(query) 
        
        print(f" * Recherche: '{query}' -> termes: {query_terms}")
        
        # Précharger les normes cosine seulement si nécessaire pour LTC
        if weighting_scheme == "ltc" and self._cosine_norms_cache is None:
            self._load_or_compute_cosine_norms()
        
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
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]
    
    def get_term_weight(self, term, doc_id, weighting_scheme):
        """Retourne le poids d'un terme spécifique dans un document"""
        # Précharger les normes cosine seulement si nécessaire pour LTC
        if weighting_scheme == "ltc" and self._cosine_norms_cache is None:
            self._load_or_compute_cosine_norms()
            
        if weighting_scheme == "ltn":
            return self.smart_ltn_weighting(term, doc_id)
        elif weighting_scheme == "ltc":
            return self.smart_ltc_weighting(term, doc_id)
        elif weighting_scheme == "bm25":
            return self.bm25_weighting(term, doc_id)
        else:
            return 0.0

    def clear_cosine_norms_cache(self):
        """Effacer le cache des normes cosine"""
        cache_file = self._get_cosine_norms_cache_filename()
        if os.path.exists(cache_file):
            os.remove(cache_file)
            self._cosine_norms_cache = None
            print("Cache des normes cosine effacé!")