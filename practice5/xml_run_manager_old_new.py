import os
import time
import pickle
from collections import defaultdict
from typing import List, Dict, Optional
import hashlib

from indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval


class INEXRunGenerator:
    """Générateur de runs INEX unifié pour tous les exercices"""
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.team_name = "AlphaAnaClement"
    
    def _get_cache_key(self, config_type: str, params: Dict) -> str:
        """Génère une clé de cache unique"""
        params_str = str(sorted(params.items()))
        key = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{config_type}_{key}"
    """
    def create_or_load_index(self, xml_dir: str, index_type: str, 
                            config: Dict, max_files: int = None) -> Dict:
        
        #Crée ou charge un index depuis le cache
        
        cache_key = self._get_cache_key(f"{index_type}_index", config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Vérifier le cache
        if os.path.exists(cache_file):
            print(f"Chargement {index_type} depuis cache...")
            try:
                index = WeightedInvertedIndex.load_from_file(cache_file)
                
                # Si c'est un index d'éléments, restaurer target_tags
                if index_type == 'element' and 'target_tags' in config:
                    index.target_tags = set(config['target_tags'])
                
                return {
                    'index': index,
                    'indexing_time': 0,
                    'config': config
                }
            except Exception as e:
                print(f"Cache corrompu ({e}), recalcul...")
        
        # Créer un nouvel index
        index = WeightedInvertedIndex()
        index.configure(**config)
        
        start_time = time.time()
        
        if index_type == 'article':
            indexing_time = index.build_index_from_xml_collection(xml_dir, max_files)
        else:  # 'element'
            target_tags = config.get('target_tags', ['sec', 'p', 'bdy'])
            indexing_time = index.build_index_from_xml_elements(xml_dir, target_tags, max_files)
        
        # Sauvegarder dans le cache
        try:
            index.save_to_file(cache_file)
            print(f"Index {index_type} sauvegardé dans le cache")
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")
        
        return {
            'index': index,
            'indexing_time': indexing_time,
            'config': config
        }
    """
    def create_or_load_index(self, xml_dir: str, index_type: str, 
                            config: Dict, max_files: int = None) -> Dict:
        """
        Crée ou charge un index depuis le cache
        Retourne: {'index': index, 'indexing_time': temps, 'config': config}
        """
        cache_key = self._get_cache_key(f"{index_type}_index", config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Vérifier le cache
        if os.path.exists(cache_file):
            print(f"Chargement {index_type} depuis cache...")
            try:
                index = WeightedInvertedIndex.load_from_file(cache_file)
                return {
                    'index': index,
                    'indexing_time': 0,  # Temps de chargement négligeable
                    'config': config
                }
            except Exception as e:
                print(f"Cache corrompu ({e}), recalcul...")
        
        # Créer un nouvel index
        index = WeightedInvertedIndex()
        index.configure(**config)
        
        start_time = time.time()
        
        if index_type == 'article':
            indexing_time = index.build_index_from_xml_collection(xml_dir, max_files)
        else:  # 'element'
            target_tags = config.get('target_tags', ['sec', 'p', 'bdy'])
            indexing_time = index.build_index_from_xml_elements(xml_dir, target_tags, max_files)
        
        # Sauvegarder dans le cache
        try:
            index.save_to_file(cache_file)
            print(f"Index {index_type} sauvegardé dans le cache")
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")
        
        return {
            'index': index,
            'indexing_time': indexing_time,
            'config': config
        }
    # ==================== MÉTHODES POUR EXERCICES 1-2 ====================
    
    def generate_article_run(self, run_id: str, xml_dir: str, 
                            queries: Dict[int, str], config: Dict,
                            weighting_scheme: str = "ltn",
                            k1: float = 1.2, b: float = 0.75) -> str:
        """
        Génère un run pour articles (exercices 1-2)
        """
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN ARTICLES {run_id} - {weighting_scheme.upper()}")
        print('='*70)
        
        # Créer/charger index
        index_data = self.create_or_load_index(xml_dir, 'article', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Générer nom de fichier
        filename = self._generate_article_filename(run_id, config, weighting_scheme, k1, b)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche
                top_docs = ranker.search_query(
                    query_text,
                    weighting_scheme=weighting_scheme,
                    top_k=1500,
                    k1=k1,
                    b=b
                )
                
                # Écrire résultats
                rank = 1
                for doc_id, score in top_docs:
                    f.write(
                        f"{query_id} Q0 {doc_id} {rank} "
                        f"{score:.6f} {self.team_name} /article[1]\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  {len(top_docs)} résultats")
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print('='*70)
        
        return filename
    
    def _generate_article_filename(self, run_id: str, config: Dict,
                                weighting_scheme: str, k1: float, b: float) -> str:
        """Génère nom de fichier pour articles"""
        os.makedirs("data/runs", exist_ok=True)
        
        stemmer = config.get('stemmer', 'nostem')
        stop_words = config.get('stop_words', 'nostop')
        test_type = config.get('test_type', '')
        
        parts = [
            self.team_name,
            run_id  # run_id contient déjà le type de test si nécessaire
        ]
        
        # Ajouter test_type seulement s'il n'est pas vide
        if test_type and test_type.strip():
            parts.append(test_type.strip())
        
        parts.extend([
            weighting_scheme,
            "article",
            stop_words,
            stemmer
        ])
        
        if weighting_scheme == "bm25":
            parts.extend([f"k1_{k1}", f"b_{b}"])
        
        # Filtrer les parties vides et créer le nom de fichier
        filtered_parts = [part for part in parts if part and str(part).strip()]
        filename = '_'.join(filtered_parts) + ".txt"
        return os.path.join("data/runs", filename)

    # ==================== MÉTHODES POUR EXERCICE 3 ====================
    
    def generate_element_run_exercise3(self, xml_dir: str, queries: Dict[int, str]) -> str:
        """
        Spécifique pour l'exercice 3 : éléments avec SMART ltn
        """
        print(f"\n{'='*70}")
        print("EXERCICE 3: Indexation XML éléments (bdy, sec, p) - SMART ltn")
        print('='*70)
        
        # Configuration exacte exercice 3
        config = {
            'tokenization': 'basic',
            'stemmer': 'nostem',
            'stop_words': 'nostop',
            'target_tags': ['bdy', 'sec', 'p']  # Pas 'article' !
        }
        
        # Créer index éléments
        index_data = self.create_or_load_index(xml_dir, 'element', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Nom de fichier spécifique exercice 3
        target_tags_str = '-'.join(sorted(config['target_tags']))
        filename = f"{self.team_name}_testXML_ltn_element-{target_tags_str}_nostop_nostem.txt"
        filename = os.path.join("data/runs", filename)
        
        os.makedirs("data/runs", exist_ok=True)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche avec SMART ltn
                top_docs = ranker.search_query(
                    query_text,
                    weighting_scheme='ltn',  # SMART ltn comme demandé
                    top_k=1500
                )
                
                rank = 1
                for elem_id, score in top_docs:
                    metadata = index.get_metadata(elem_id)
                    article_id = metadata.get('parent_doc_id', 'unknown')
                    xml_path = metadata.get('xml_path', '/article[1]')
                    
                    f.write(
                        f"{query_id} Q0 {article_id} {rank} "
                        f"{score:.6f} {self.team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  {len(top_docs)} éléments")
        
        print(f"\n{'='*70}")
        print(f"EXERCICE 3 TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print('='*70)
        
        return filename
    
    def generate_element_run(self, run_id: str, xml_dir: str, 
                            queries: Dict[int, str], config: Dict,
                            weighting_scheme: str = "ltn",
                            k1: float = 1.2, b: float = 0.75) -> str:
        """
        Génère un run pour éléments XML (exercice 4)
        """
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN ÉLÉMENTS {run_id} - {weighting_scheme.upper()}")
        print('='*70)
        
        # Créer/charger index
        index_data = self.create_or_load_index(xml_dir, 'element', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Générer nom de fichier
        target_tags = config.get('target_tags', ['sec', 'p', 'bdy'])
        target_tags_str = '-'.join(sorted(target_tags))
        
        parts = [
            self.team_name,
            run_id,
            weighting_scheme,
            f"element-{target_tags_str}",
            config.get('stop_words', 'nostop'),
            config.get('stemmer', 'nostem')
        ]
        
        if weighting_scheme == "bm25":
            parts.extend([f"k1_{k1}", f"b_{b}"])
        
        filename = '_'.join(parts) + ".txt"
        filename = os.path.join("data/runs", filename)
        
        os.makedirs("data/runs", exist_ok=True)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche
                top_docs = ranker.search_query(
                    query_text,
                    weighting_scheme=weighting_scheme,
                    top_k=1500,
                    k1=k1,
                    b=b
                )
                
                # Écrire résultats
                rank = 1
                for elem_id, score in top_docs:
                    metadata = index.get_metadata(elem_id)
                    article_id = metadata.get('parent_doc_id', 'unknown')
                    xml_path = metadata.get('xml_path', '/article[1]')
                    
                    f.write(
                        f"{query_id} Q0 {article_id} {rank} "
                        f"{score:.6f} {self.team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  {len(top_docs)} éléments")
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print('='*70)
        
        return filename
    
    # ==================== MÉTHODES POUR EXERCICES 4-6 ====================
    
    def generate_fetch_browse_run(self, 
                                 run_id: str,
                                 xml_dir: str,
                                 queries: Dict[int, str],
                                 fetch_config: Dict,
                                 browse_config: Dict,
                                 run_params: Dict = None) -> str:
        """
        Version optimisée pour Fetch & Browse (exercices avancés)
        """
        if run_params is None:
            run_params = {
                'top_articles': 1600,
                'max_elements': 1500,
                'weighting_scheme': 'ltn',
                'min_element_score': 0.00001
            }
        
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN {run_id} (Fetch & Browse)")
        print('='*70)
        
        total_start = time.time()
        
        # 1. Phase FETCH
        fetch_data = self.create_or_load_index(xml_dir, 'article', fetch_config)
        fetch_index = fetch_data['index']
        fetch_ranker = RankedRetrieval(fetch_index)
        
        # 2. Phase BROWSE
        browse_data = self.create_or_load_index(xml_dir, 'element', browse_config)
        browse_index = browse_data['index']
        browse_ranker = RankedRetrieval(browse_index)
        
        # 3. Préparer cache
        element_cache = {}
        article_to_elements = defaultdict(list)
        
        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))
            
            if parent_id:
                article_to_elements[parent_id].append(elem_id)
                
                # Extraire tag
                xml_path = metadata.get('xml_path', '')
                tag = metadata.get('tag', 'unknown')
                if tag == 'unknown':
                    if '/p[' in xml_path:
                        tag = 'p'
                    elif '/sec[' in xml_path:
                        tag = 'sec'
                    elif '/bdy[' in xml_path:
                        tag = 'bdy'
                    elif xml_path == '/article[1]' or xml_path.endswith('/article[1]'):
                        tag = 'article'
                
                priority = {'p': 4, 'sec': 3, 'bdy': 2, 'article': 1}.get(tag, 0)
                
                element_cache[elem_id] = {
                    'xml_path': xml_path,
                    'tag': tag,
                    'priority': priority
                }
        
        # 4. Générer le fichier
        filename = self._generate_fetch_browse_filename(run_id, fetch_config, browse_config, run_params)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # A. FETCH: Articles
                top_articles = fetch_ranker.search_query(
                    query_text,
                    weighting_scheme=run_params['weighting_scheme'],
                    top_k=run_params['top_articles']
                )
                
                print(f"  FETCH: {len(top_articles)} articles")
                
                # B. Collecter éléments
                article_results = defaultdict(list)
                query_terms = browse_ranker.process_query_terms(query_text)
                
                for article_id, article_score in top_articles:
                    str_article_id = str(article_id)
                    
                    if str_article_id in article_to_elements:
                        for elem_id in article_to_elements[str_article_id]:
                            score = 0.0
                            for term in query_terms:
                                weight = browse_ranker.get_term_weight(
                                    term, elem_id,
                                    weighting_scheme=run_params['weighting_scheme']
                                )
                                if weight:
                                    score += weight
                            
                            if score >= run_params.get('min_element_score', 0.00001):
                                elem_info = element_cache.get(elem_id, {})
                                
                                article_results[article_id].append({
                                    'element_id': elem_id,
                                    'score': score,
                                    'priority': elem_info.get('priority', 0),
                                    'tag': elem_info.get('tag', 'unknown'),
                                    'xml_path': elem_info.get('xml_path', '/article[1]')
                                })
                    
                    # Fallback: article entier
                    if not article_results[article_id]:
                        article_results[article_id].append({
                            'element_id': f"{article_id}_article",
                            'score': article_score,
                            'priority': 1,
                            'tag': 'article',
                            'xml_path': '/article[1]'
                        })
                
                # C. Sélectionner meilleur élément par article
                final_elements = []
                
                for article_id, elements in article_results.items():
                    if elements:
                        elements.sort(key=lambda x: (-x['priority'], -x['score']))
                        best = elements[0]
                        final_elements.append({
                            'article_id': article_id,
                            'score': best['score'],
                            'xml_path': best['xml_path'],
                            'tag': best['tag']
                        })
                
                # D. Trier et limiter
                final_elements.sort(key=lambda x: -x['score'])
                final_elements = final_elements[:run_params['max_elements']]
                
                # E. Écrire
                rank = 1
                for result in final_elements:
                    xml_path = result['xml_path']
                    
                    if not xml_path.startswith('/article['):
                        if '/article' in xml_path:
                            xml_path = f"/article[1]{xml_path.split('/article', 1)[-1]}"
                        else:
                            xml_path = f"/article[1]{xml_path}"
                    
                    f.write(
                        f"{query_id} Q0 {result['article_id']} {rank} "
                        f"{result['score']:.6f} {self.team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                query_time = time.time() - query_start
                print(f"  {len(final_elements)} éléments, temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Temps total: {total_time:.2f}s")
        print('='*70)
        
        return filename
    
    def _generate_fetch_browse_filename(self, run_id: str,
                                       fetch_config: Dict, browse_config: Dict,
                                       run_params: Dict) -> str:
        """Génère nom de fichier pour Fetch & Browse"""
        os.makedirs("data/runs", exist_ok=True)
        
        parts = [
            self.team_name,
            run_id,
            f"fetch-{fetch_config.get('stemmer', 'nostem')}-{fetch_config.get('stop_words', 'nostop')}",
            f"browse-{'_'.join(browse_config.get('target_tags', ['sec','p']))}",
            run_params['weighting_scheme']
        ]
        
        if run_params['weighting_scheme'] == 'bm25':
            parts.append(f"k{run_params.get('k1', 1.2)}")
            parts.append(f"b{run_params.get('b', 0.75)}")
        
        filename = "_".join(parts) + ".txt"
        return os.path.join("data/runs", filename)
    
    def validate_run_file(self, filename: str) -> bool:
        """Valide qu'un run respecte les règles INEX"""
        print(f"\n[VALIDATION de {filename}]")
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        violations = 0
        
        # Vérifier nombre de résultats
        if len(lines) != 7 * 1500:
            print(f"  ⚠️  Nombre de résultats: {len(lines)} (attendu: {7 * 1500})")
        
        # Vérifier format
        for i, line in enumerate(lines[:10]):
            parts = line.split()
            if len(parts) != 7:
                print(f"  ❌ Ligne {i+1}: Format incorrect")
                violations += 1
        
        if violations == 0:
            print(f"  ✅ RUN VALIDE")
        else:
            print(f"  ❌ {violations} violations détectées")
        
        return violations == 0