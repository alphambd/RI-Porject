import os
import time
import hashlib
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import math

from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument

class FieldWeightedIndex:
    """Index avec pondération par champs AVEC CACHE"""
    
    def __init__(self, cache_dir="data/cache/field_index"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.index = None
        self.field_tfs = defaultdict(lambda: defaultdict(dict))
        self.field_weights = {}
        self.doc_lengths = {}
        self.doc_ids = []
        self.df = {}
        self.field_stats = {}
    
    def _get_cache_key(self, xml_dir: str, fields_config: Dict, 
                      field_weights: Dict, config: Dict) -> str:
        """Génère une clé de cache unique"""
        params = {
            'xml_dir': xml_dir,
            'fields_config': fields_config,
            'field_weights': field_weights,
            'tokenization': config.get('tokenization', 'basic'),
            'stemmer': config.get('stemmer', 'nostem'),
            'stop_words': config.get('stop_words', 'nostop'),
            'file_count': len(self._get_xml_files(xml_dir, None))
        }
        params_str = str(sorted(params.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:16]
    
    def _save_to_cache(self, cache_key: str):
        """Sauvegarde l'index dans le cache"""
        cache_data = {
            'field_tfs': dict(self.field_tfs),
            'field_weights': self.field_weights,
            'doc_lengths': self.doc_lengths,
            'doc_ids': self.doc_ids,
            'df': self.df,
            'index_data': self.index.get_cache_data() if self.index else None,
            'field_stats': self.field_stats
        }
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f" Index sauvegardé dans le cache: {cache_file}")
    
    def _load_from_cache(self, cache_key: str) -> bool:
        """Charge l'index depuis le cache - CORRIGÉ"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_file):
            return False
        
        try:
            print(f" Chargement depuis le cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restaurer les données
            self.field_tfs = defaultdict(lambda: defaultdict(dict), cache_data['field_tfs'])
            self.field_weights = cache_data['field_weights']
            self.doc_lengths = cache_data['doc_lengths']
            self.doc_ids = cache_data['doc_ids']
            self.df = cache_data['df']
            self.field_stats = cache_data.get('field_stats', {})
            
            # CORRECTION: S'assurer que avg_doc_length est bien calculé
            if cache_data['index_data']:
                self.index = WeightedInvertedIndex()
                index_data = cache_data['index_data']
                
                # Restaurer les données de base
                self.index.dictionary = defaultdict(dict, index_data['dictionary'])
                self.index.doc_ids = index_data['doc_ids']
                self.index.doc_lengths = index_data['doc_lengths']
                self.index.doc_count = index_data['doc_count']
                self.index.total_terms = index_data['total_terms']
                
                # CORRECTION: avg_doc_length DOIT être calculé
                if self.index.doc_count > 0 and self.index.total_terms > 0:
                    self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
                else:
                    # Recalculer depuis doc_lengths si nécessaire
                    total_length = sum(self.index.doc_lengths.values())
                    self.index.avg_doc_length = total_length / self.index.doc_count if self.index.doc_count > 0 else 0
                
                # Configuration
                self.index.stop_list_name = index_data['config']['stop_list_name']
                self.index.stemmer_name = index_data['config']['stemmer_name']
                self.index.tokenization_method = index_data['config']['tokenization_method']
            
            print(f" Index chargé depuis le cache: {len(self.doc_ids)} documents")
            print(f"   avg_doc_length = {self.index.avg_doc_length:.2f}")
            return True
            
        except Exception as e:
            print(f" Erreur chargement cache: {e}")
            return False
        
    def configure(self, tokenization="basic", stemmer="nostem", stop_words="nostop"):
        """Configure l'index principal"""
        self.index = WeightedInvertedIndex()
        self.index.configure(
            tokenization=tokenization,
            stemmer=stemmer,
            stop_words=stop_words
        )
    
    def build_or_load_field_index(self, xml_dir: str, 
                                 fields_config: Dict[str, List[str]],
                                 field_weights: Dict[str, float],
                                 config: Dict,
                                 max_files: Optional[int] = None,
                                 force_rebuild: bool = False) -> int:
        """
        Construit OU charge depuis cache
        """
        # 1. Générer clé de cache
        cache_key = self._get_cache_key(xml_dir, fields_config, field_weights, config)
        
        # 2. Essayer de charger depuis cache
        if not force_rebuild and self._load_from_cache(cache_key):
            return len(self.doc_ids)
        
        # 3. Si cache manquant ou force_rebuild, construire
        print("🔄 Construction de l'index (cache manquant)...")
        self.configure(**config)
        self.field_weights = field_weights
        
        start_time = time.time()
        xml_files = self._get_xml_files(xml_dir, max_files)
        
        for i, xml_file in enumerate(xml_files):
            if i % 100 == 0:
                print(f"  Traitement {i}/{len(xml_files)}...")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
                continue
            
            doc_id = doc.doc_id
            self.doc_ids.append(doc_id)
            self.doc_lengths[doc_id] = 0
            
            if doc_id not in self.field_tfs:
                self.field_tfs[doc_id] = defaultdict(dict)
            
            all_terms = []
            
            for field_name, target_tags in fields_config.items():
                main_tag = target_tags[0]
                field_text = self._extract_field_simple(doc, main_tag)
                
                if field_text:
                    tokens = self.index.apply_tokenization(field_text)
                    terms = self.index.process_tokens(tokens)
                    
                    if terms:
                        term_counts = Counter(terms)
                        for term, tf in term_counts.items():
                            self.field_tfs[doc_id][field_name][term] = float(tf)
                        all_terms.extend(terms)
            
            self.doc_lengths[doc_id] = len(all_terms)
            
            for term in set(all_terms):
                count = all_terms.count(term)
                if doc_id not in self.index.dictionary[term]:
                    self.index.dictionary[term][doc_id] = 0
                self.index.dictionary[term][doc_id] = count
        
        # Finaliser
        self.index.doc_ids = self.doc_ids
        self.index.doc_lengths = self.doc_lengths
        self.index.doc_count = len(self.doc_ids)
        self.index.total_terms = sum(self.doc_lengths.values())
        
        if self.index.doc_count > 0:
            self.index.avg_doc_length = self.index.total_terms / self.index.doc_count
        
        # Calculer df
        for term, doc_dict in self.index.dictionary.items():
            self.df[term] = len(doc_dict)
        
        # Calculer statistiques par champ (pour BM25Fw)
        self._compute_field_stats()
        
        indexing_time = time.time() - start_time
        print(f" Index construit en {indexing_time:.2f}s: {len(self.doc_ids)} documents")
        
        # 4. Sauvegarder dans le cache
        self._save_to_cache(cache_key)
        
        return len(self.doc_ids)
    
    def _compute_field_stats(self):
        """Pré-calcule les statistiques par champ (pour cache)"""
        for field_name in self.field_weights.keys():
            total_field_length = 0
            field_doc_count = 0
            field_dfs = defaultdict(int)
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_tf_dict = self.field_tfs[doc_id][field_name]
                    field_length = sum(field_tf_dict.values())
                    total_field_length += field_length
                    field_doc_count += 1
                    
                    for term in field_tf_dict.keys():
                        field_dfs[term] += 1
            
            avg_field_length = total_field_length / field_doc_count if field_doc_count > 0 else 1
            
            self.field_stats[field_name] = {
                'avg_length': avg_field_length,
                'doc_count': field_doc_count,
                'dfs': dict(field_dfs)
            }
    
    # Cache pour les résultats de recherche
    def _get_search_cache_key(self, query: str, method: str, 
                            k1: float, b: float) -> str:
        """Clé de cache pour une recherche"""
        params = {
            'query': query,
            'method': method,
            'k1': k1,
            'b': b,
            'field_weights_hash': hash(tuple(sorted(self.field_weights.items())))
        }
        params_str = str(sorted(params.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
        
    def _extract_field_simple(self, doc: INEXDocument, target_tag: str) -> str:
        """
        Extrait le texte d'un CHAMP SIMPLE (non répétable)
        Version corrigée pour éviter les warnings
        """
        if doc.root is None:
            return ""
        
        # Recherche simple: trouver la première occurrence
        def find_element(elem, tag):
            elem_tag = doc._clean_tag(elem.tag)
            if elem_tag == tag:
                return elem
            
            for child in elem:
                # CORRECTION: Vérifier si child n'est pas None
                if child is not None:
                    result = find_element(child, tag)
                    if result is not None:
                        return result
            return None
        
        target_elem = find_element(doc.root, target_tag)
        if target_elem is not None:
            # Extraire tout le texte de cet élément
            text_parts = []
            try:
                for t in target_elem.itertext():
                    if t and t.strip():
                        text_parts.append(t.strip())
            except AttributeError:
                # Fallback pour ElementTree
                text = (target_elem.text or "").strip()
                for child in target_elem:
                    if child.text:
                        text += " " + child.text.strip()
                    if child.tail:
                        text += " " + child.tail.strip()
                text_parts.append(text)
            
            return ' '.join(text_parts)
        
        return ""
    
    def _get_xml_files(self, xml_dir: str, max_files: int = None) -> List[str]:
        """Liste les fichiers XML"""
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        if max_files:
            xml_files = xml_files[:max_files]
        
        return xml_files
    
    # ==================== BM25Fw ====================
    
    def search_bm25fw(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fw OPTIMISÉ : Late combination (Wilkinson94)
        """
        from collections import defaultdict
        
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # PRÉCALCULER les statistiques par champ (une seule fois)
        field_stats = {}
        for field_name in self.field_weights.keys():
            # Calculer avg_field_length pour ce champ
            total_field_length = 0
            field_doc_count = 0
            field_dfs = defaultdict(int)  # term -> df dans ce champ
            
            for doc_id in self.doc_ids:
                if field_name in self.field_tfs[doc_id]:
                    field_tf_dict = self.field_tfs[doc_id][field_name]
                    field_length = sum(field_tf_dict.values())
                    total_field_length += field_length
                    field_doc_count += 1
                    
                    # Compter les df
                    for term in field_tf_dict.keys():
                        field_dfs[term] += 1
            
            avg_field_length = total_field_length / field_doc_count if field_doc_count > 0 else 1
            
            field_stats[field_name] = {
                'avg_length': avg_field_length,
                'doc_count': field_doc_count,
                'dfs': field_dfs
            }
        
        # Dictionnaire pour scores combinés
        doc_scores = defaultdict(float)
        
        # OPTIMISATION: Traiter seulement les documents qui ont au moins un terme de la requête
        relevant_docs = set()
        
        # Pour chaque terme de la requête, trouver les documents qui le contiennent
        for term in query_terms:
            for doc_id in self.doc_ids:
                for field_name in self.field_weights.keys():
                    if term in self.field_tfs[doc_id].get(field_name, {}):
                        relevant_docs.add(doc_id)
                        break  # Passer au document suivant
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        # Pour chaque document pertinent, calculer BM25Fw
        for doc_id in relevant_docs:
            total_score = 0.0
            
            # Pour chaque champ
            for field_name, weight in self.field_weights.items():
                field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                
                if not field_tf_dict:
                    continue
                
                field_length = sum(field_tf_dict.values())
                field_stat = field_stats[field_name]
                avg_field_length = field_stat['avg_length']
                field_dfs = field_stat['dfs']
                
                field_score = 0.0
                
                # Pour chaque terme de la requête
                for term in query_terms:
                    tf = field_tf_dict.get(term, 0)
                    
                    if tf > 0:
                        df_field = field_dfs.get(term, 0)
                        
                        if df_field > 0:
                            # BM25 pour ce champ
                            idf = math.log((field_stat['doc_count'] - df_field + 0.5) / (df_field + 0.5))
                            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (field_length / avg_field_length)))
                            field_score += idf * tf_component
                
                total_score += weight * field_score
            
            if total_score > 0:
                doc_scores[doc_id] = total_score
        
        # OPTIMISATION: Si peu de documents, ajouter un score minimal aux autres
        if len(doc_scores) < 1500:
            # Pour les documents non pertinents, donner un petit score
            for doc_id in self.doc_ids:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.00001
        
        # Trier et limiter
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    def search_bm25fw_cached(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """BM25Fw avec cache des résultats"""
        cache_key = self._get_search_cache_key(query, "bm25fw", k1, b)
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.pkl")
        
        # Essayer le cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f" Résultats chargés depuis cache pour: {query[:30]}...")
                return results
            except:
                pass
        
        # Calculer
        print(f" Calcul BM25Fw pour: {query[:30]}...")
        results = self.search_bm25fw(query, k1, b)
        
        # Sauvegarder dans le cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except:
            pass
        
        return results
    
    # ==================== BM25Fr ====================
    
    def search_bm25fr(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr OPTIMISÉ : Early combination (Robertson94)
        """
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # OPTIMISATION: Pré-calculer les documents pertinents
        relevant_docs = set()
        for term in query_terms:
            if term in self.index.dictionary:
                relevant_docs.update(self.index.dictionary[term].keys())
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        doc_scores = defaultdict(float)
        
        # Pour chaque document pertinent seulement
        for doc_id in relevant_docs:
            combined_tf = defaultdict(float)
            
            # Étape 1: Combiner les TF pondérés
            for term in query_terms:
                tf_star = 0.0
                for field_name, weight in self.field_weights.items():
                    field_tf_dict = self.field_tfs[doc_id].get(field_name, {})
                    tf_in_field = field_tf_dict.get(term, 0)
                    tf_star += weight * tf_in_field
                
                if tf_star > 0:
                    combined_tf[term] = tf_star
            
            # Étape 2: BM25 sur les TF combinés
            doc_score = 0.0
            for term, tf_star in combined_tf.items():
                # DF global
                df = self.df.get(term, 0)
                """
                if df > 0:
                    # BM25 avec tf_star
                    idf = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
                    tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (self.doc_lengths[doc_id] / self.index.avg_doc_length)))
                    doc_score += idf * tf_component
                """
                if df > 0:
                    # CORRECTION: Vérifier avg_doc_length
                    if self.index.avg_doc_length <= 0:
                        self.index.avg_doc_length = 1.0  # Valeur par défaut
                    
                    idf = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
                    
                    # Protection contre division par zéro
                    doc_length = self.doc_lengths.get(doc_id, 1)
                    length_ratio = doc_length / self.index.avg_doc_length
                    
                    denominator = tf_star + k1 * (1 - b + b * length_ratio)
                    if denominator > 0:
                        tf_component = (tf_star * (k1 + 1)) / denominator
                        doc_score += idf * tf_component
                        
            if doc_score > 0:
                doc_scores[doc_id] = doc_score
        
        # OPTIMISATION: Si peu de documents, ajouter un score minimal
        if len(doc_scores) < 1500:
            for doc_id in self.doc_ids:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.00001
        
        # Trier
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    def search_bm25fr_optimized(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        BM25Fr ULTRA OPTIMISÉ : Early combination
        """
        # Traiter la requête
        tokens = self.index.apply_tokenization(query)
        query_terms = self.index.process_tokens(tokens)
        
        if not query_terms:
            return []
        
        # OPTIMISATION 1: Filtrer les termes qui existent dans la collection
        valid_terms = [term for term in query_terms if term in self.df]
        if not valid_terms:
            return []
        
        # OPTIMISATION 2: Précalculer l'IDF pour chaque terme
        idf_cache = {}
        for term in valid_terms:
            df = self.df[term]
            idf_cache[term] = math.log((self.index.doc_count - df + 0.5) / (df + 0.5))
        
        # OPTIMISATION 3: Documents qui contiennent au moins un terme
        relevant_docs = set()
        for term in valid_terms:
            if term in self.index.dictionary:
                relevant_docs.update(self.index.dictionary[term].keys())
        
        print(f"  Documents pertinents: {len(relevant_docs)}/{len(self.doc_ids)}")
        
        # OPTIMISATION 4: Batch processing
        batch_size = 500
        doc_scores = {}
        
        doc_list = list(relevant_docs)
        for i in range(0, len(doc_list), batch_size):
            batch = doc_list[i:min(i+batch_size, len(doc_list))]
            
            for doc_id in batch:
                combined_tf = 0.0
                total_tf_star = 0.0
                
                # Pour chaque terme
                for term in valid_terms:
                    tf_star = 0.0
                    # Combiner les TF pondérés
                    for field_name, weight in self.field_weights.items():
                        field_tf_dict = self.field_tfs.get(doc_id, {}).get(field_name, {})
                        tf_in_field = field_tf_dict.get(term, 0)
                        tf_star += weight * tf_in_field
                    
                    if tf_star > 0:
                        combined_tf += tf_star
                        
                        # BM25 partiel
                        idf = idf_cache[term]
                        tf_component = (tf_star * (k1 + 1)) / (tf_star + k1 * (1 - b + b * (self.doc_lengths[doc_id] / self.index.avg_doc_length)))
                        total_tf_star += idf * tf_component
                
                if total_tf_star > 0:
                    doc_scores[doc_id] = total_tf_star
        
        # OPTIMISATION 5: Si besoin, ajouter des documents avec score minimal
        if len(doc_scores) < 1500:
            needed = 1500 - len(doc_scores)
            other_docs = [d for d in self.doc_ids if d not in doc_scores]
            
            # Prendre les premiers autres documents
            for doc_id in other_docs[:needed]:
                doc_scores[doc_id] = 0.00001
        
        # Trier
        sorted_results = sorted(doc_scores.items(), key=lambda x: -x[1])
        return sorted_results[:1500]
    
    def search_bm25fr_cached(self, query: str, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
        """BM25Fr avec cache des résultats"""
        cache_key = self._get_search_cache_key(query, "bm25fr", k1, b)
        cache_file = os.path.join(self.cache_dir, f"search_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f" Résultats chargés depuis cache pour: {query[:30]}...")
                return results
            except:
                pass
        
        print(f" Calcul BM25Fr pour: {query[:30]}...")
        results = self.search_bm25fr(query, k1, b)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except:
            pass
        
        return results
    
def generate_field_weighted_run_cached(generator, run_id: str, run_type: str,
                                     xml_dir: str, queries: Dict[int, str],
                                     config: Dict, run_params: Dict,
                                     fields_config: Dict = None,
                                     field_weights: Dict = None):
    """
    Version avec CACHE
    """
    print(f"\n{'='*70}")
    print(f"GÉNÉRATION RUN {run_id} - {run_type.upper()} (AVEC CACHE)")
    print('='*70)
    
    # Configuration
    if fields_config is None:
        fields_config = {'title': ['title'], 'body': ['bdy']}
    if field_weights is None:
        field_weights = {'title': 3.0, 'body': 1.0}
    
    # Créer l'index AVEC CACHE
    field_index = FieldWeightedIndex(cache_dir="data/cache/field_weighted")
    
    start_time = time.time()
    print("Chargement/construction de l'index...")
    
    # Cette méthode charge depuis cache si disponible
    doc_count = field_index.build_or_load_field_index(
        xml_dir=xml_dir,
        fields_config=fields_config,
        field_weights=field_weights,
        config=config,
        max_files=run_params.get('max_files', None),
        force_rebuild=False  # Mettre à True pour forcer le recalcul
    )
    
    # Générer le fichier
    team_name = "AlphaAnaClement"
    fields_str = '-'.join(fields_config.keys())
    filename = f"{team_name}_{run_id}_{run_type}_fields-{fields_str}_{config['stop_words']}_{config['stemmer']}_k{run_params.get('k1', 1.2)}_b{run_params.get('b', 0.75)}.txt"
    filename = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    results_count = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for query_id, query_text in queries.items():
            print(f"\n[Query {query_id}] {query_text[:50]}...")
            query_start = time.time()
            
            # Utiliser les méthodes CACHÉES
            if run_type == 'bm25fw':
                results = field_index.search_bm25fw_cached(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            else:  # bm25fr
                results = field_index.search_bm25fr_cached(
                    query_text,
                    k1=run_params.get('k1', 1.2),
                    b=run_params.get('b', 0.75)
                )
            
            # Écrire résultats
            rank = 1
            for doc_id, score in results[:1500]:
                f.write(
                    f"{query_id} Q0 {doc_id} {rank} "
                    f"{score:.6f} {team_name} /article[1]\n"
                )
                rank += 1
                results_count += 1
            
            query_time = time.time() - query_start
            print(f"  {len(results)} articles, temps: {query_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f" RUN {run_type.upper()} TERMINÉ (AVEC CACHE)")
    print(f" Fichier: {filename}")
    print(f" Documents indexés: {doc_count}")
    print(f"   Temps total: {total_time:.2f}s")
    print('='*70)
    
    return filename

