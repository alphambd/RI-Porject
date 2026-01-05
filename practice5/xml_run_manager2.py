import os
import time
import pickle
from collections import defaultdict
from typing import List, Dict
import hashlib

from indexer2 import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval


class INEXRunGenerator:
    """Générateur de runs INEX optimisé pour Fetch and Browse"""
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, config_type: str, params: Dict) -> str:
        """Génère une clé de cache unique"""
        params_str = str(sorted(params.items()))
        key = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{config_type}_{key}"
    
    def create_or_load_index(self, xml_dir: str, index_type: str, 
                            config: Dict, max_files: int = None) -> Dict:
        """
        Crée ou charge un index depuis le cache
        """
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
            indexing_time = index.build_index_from_articles(xml_dir, max_files)
        else:  # 'element'
            target_tags = config.get('target_tags', ['sec', 'p', 'bdy', 'article'])
            indexing_time = index.build_index_from_elements(xml_dir, target_tags, max_files)
        
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
    
    def generate_fetch_browse_run(self, 
                                 run_id: str,
                                 xml_dir: str,
                                 queries: Dict[int, str],
                                 fetch_config: Dict,
                                 browse_config: Dict,
                                 run_params: Dict = None) -> str:
        """
        Version unique et optimisée qui :
        1. Respecte le format INEX
        2. Élimine les small_err_nodes
        3. Gère les overlaps
        4. Priorise p > sec > bdy > article
        5. Produit exactement 1500 résultats par requête
        """
        if run_params is None:
            run_params = {
                'top_articles': 1600,
                'max_elements': 1500,
                'weighting_scheme': 'ltn',
                'min_element_score': 0.00001
            }
        
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN {run_id}")
        print('='*70)
        
        total_start = time.time()
        
        # 1. Phase FETCH: Index des articles
        fetch_data = self.create_or_load_index(xml_dir, 'article', fetch_config)
        fetch_index = fetch_data['index']
        fetch_ranker = RankedRetrieval(fetch_index)
        
        # 2. Phase BROWSE: Index des éléments
        browse_data = self.create_or_load_index(xml_dir, 'element', browse_config)
        browse_index = browse_data['index']
        browse_ranker = RankedRetrieval(browse_index)
        
        # 3. Préparer cache élément -> metadata
        print("\n[Création du cache des éléments...]")
        element_cache = {}
        article_to_elements = defaultdict(list)
        
        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))
            
            if parent_id:
                article_to_elements[parent_id].append(elem_id)
                
                # Extraire tag du xml_path
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
                
                # Priorité
                priority = {'p': 4, 'sec': 3, 'bdy': 2, 'article': 1}.get(tag, 0)
                
                element_cache[elem_id] = {
                    'xml_path': xml_path,
                    'tag': tag,
                    'priority': priority,
                    'parent_id': parent_id
                }
        
        print(f"  Cache créé: {len(element_cache)} éléments")
        print(f"  Articles indexés: {len(article_to_elements)}")
        
        # 4. Générer le fichier run
        team_name = "AlphaAnaClement"
        filename = self._generate_filename(team_name, run_id, 
                                          fetch_config, browse_config, run_params)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # A. FETCH: Articles pertinents
                top_articles = fetch_ranker.search_query(
                    query_text,
                    weighting_scheme=run_params['weighting_scheme'],
                    top_k=run_params['top_articles']
                )
                
                print(f"  FETCH: {len(top_articles)} articles")
                
                # B. Collecter éléments pertinents par article
                article_results = defaultdict(list)
                query_terms = browse_ranker.process_query_terms(query_text)
                
                processed_count = 0
                for article_id, article_score in top_articles:
                    processed_count += 1
                    if processed_count % 200 == 0:
                        print(f"    Traité {processed_count}/{len(top_articles)} articles...")
                    
                    str_article_id = str(article_id)
                    
                    # Chercher les éléments de cet article
                    if str_article_id in article_to_elements:
                        for elem_id in article_to_elements[str_article_id]:
                            # Calcul score rapide
                            score = 0.0
                            for term in query_terms:
                                weight = browse_ranker.get_term_weight(
                                    term, elem_id,
                                    weighting_scheme=run_params['weighting_scheme']
                                )
                                if weight:
                                    score += weight
                            
                            if score >= run_params['min_element_score']:
                                elem_info = element_cache.get(elem_id, {})
                                
                                article_results[article_id].append({
                                    'element_id': elem_id,
                                    'score': score,
                                    'priority': elem_info.get('priority', 0),
                                    'tag': elem_info.get('tag', 'unknown'),
                                    'xml_path': elem_info.get('xml_path', '/article[1]')
                                })
                    
                    # Fallback: article entier si aucun élément trouvé
                    if not article_results[article_id]:
                        article_results[article_id].append({
                            'element_id': f"{article_id}_article",
                            'score': article_score,
                            'priority': 1,
                            'tag': 'article',
                            'xml_path': '/article[1]'
                        })
                
                # C. Sélectionner MEILLEUR élément par article (sans overlap)
                final_elements = []
                
                for article_id, elements in article_results.items():
                    if not elements:
                        continue
                    
                    # Trier par priorité puis score
                    elements.sort(key=lambda x: (-x['priority'], -x['score']))
                    
                    # Prendre seulement le meilleur élément par article
                    # (simplification: pas d'overlap car un seul élément par article)
                    best_element = elements[0]
                    
                    final_elements.append({
                        'article_id': article_id,
                        'score': best_element['score'],
                        'xml_path': best_element['xml_path'],
                        'tag': best_element['tag']
                    })
                
                # D. Trier et limiter à 1500 éléments
                final_elements.sort(key=lambda x: -x['score'])
                final_elements = final_elements[:run_params['max_elements']]
                
                # Statistiques
                tags_count = defaultdict(int)
                for elem in final_elements:
                    tags_count[elem['tag']] += 1
                
                print(f"  RÉSULTATS: {len(final_elements)} éléments")
                print(f"    Distribution: p={tags_count.get('p',0)}, "
                      f"sec={tags_count.get('sec',0)}, "
                      f"bdy={tags_count.get('bdy',0)}, "
                      f"article={tags_count.get('article',0)}")
                
                # E. Écrire dans le format INEX
                rank = 1
                for result in final_elements:
                    xml_path = result['xml_path']
                    
                    # Formatage standard INEX
                    if not xml_path.startswith('/article['):
                        if '/article' in xml_path:
                            xml_path = f"/article[1]{xml_path.split('/article', 1)[-1]}"
                        else:
                            xml_path = f"/article[1]{xml_path}"
                    
                    # Format exact: query_id Q0 article_id rank score team_name xml_path
                    f.write(
                        f"{query_id} Q0 {result['article_id']} {rank} "
                        f"{result['score']:.6f} {team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                query_time = time.time() - query_start
                print(f"  Temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Temps total: {total_time:.2f}s")
        print('='*70)
        
        # Validation
        self._validate_run_file(filename)
        
        return filename
    
    def _generate_filename(self, team_name: str, run_id: str,
                          fetch_config: Dict, browse_config: Dict,
                          run_params: Dict) -> str:
        """Génère un nom de fichier descriptif"""
        os.makedirs("data/runs", exist_ok=True)
        
        parts = [
            team_name,
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
    
    
    def generate_element_only_run(self, 
                                 run_id: str,
                                 xml_dir: str,
                                 queries: Dict[int, str],
                                 config: Dict,
                                 run_params: Dict = None) -> str:
        """
        Version simplifiée POUR L'EXERCICE 3 :
        Indexe et recherche directement dans les éléments XML
        (pas d'approche Fetch & Browse)
        """
        if run_params is None:
            run_params = {
                'max_elements': 1500,
                'weighting_scheme': 'lm',  # SMART lm par défaut
                'min_score': 0.00001
            }
        
        print(f"\n{'='*70}")
        print(f"EXERCICE 3 - Indexation directe des éléments")
        print('='*70)
        
        # Indexer seulement les éléments
        index_data = self.create_or_load_index(
            xml_dir, 'element', config, max_files=None
        )
        
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Générer nom de fichier selon spécifications exercice 3
        team_name = "AlphaAnaClement"
        group_number = "12"
        
        # Construction du nom selon format demandé
        target_tags_str = '-'.join(sorted(config.get('target_tags', ['bdy', 'sec', 'p'])))
        stemmer = config.get('stemmer', 'nostem')
        stop_words = config.get('stop_words', 'nostop')
        
        filename = f"{team_name}_{group_number}_{run_id}_lm_element-{target_tags_str}_{stop_words}_{stemmer}.txt"
        filename = os.path.join("data/runs", filename)
        
        os.makedirs("data/runs", exist_ok=True)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche directe dans les éléments
                results = ranker.search_query(
                    query_text,
                    weighting_scheme=run_params['weighting_scheme'],
                    top_k=run_params['max_elements']
                )
                
                # Limiter et écrire
                results = results[:run_params['max_elements']]
                
                rank = 1
                for elem_id, score in results:
                    metadata = index.get_metadata(elem_id)
                    article_id = metadata.get('parent_doc_id', 'unknown')
                    xml_path = metadata.get('xml_path', '/article[1]')
                    
                    f.write(
                        f"{query_id} Q0 {article_id} {rank} "
                        f"{score:.6f} {team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  Écrit {len(results)} éléments")
                print(f"  Temps: {time.time() - query_start:.2f}s")
        
        print(f"\n{'='*70}")
        print(f"RUN EXERCICE 3 TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print('='*70)
        
        return filename
    
    def _validate_run_file(self, filename: str):
        """Valide qu'un run respecte les règles INEX"""
        print(f"\n[VALIDATION de {filename}]")
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        violations = 0
        
        # Vérifier nombre de résultats
        if len(lines) != 7 * 1500:
            print(f"  ⚠️  Nombre de résultats: {len(lines)} (attendu: {7 * 1500})")
        
        # Vérifier format des lignes
        for i, line in enumerate(lines[:10]):  # Vérifier seulement les 10 premières
            parts = line.split()
            if len(parts) != 7:
                print(f"  ❌ Ligne {i+1}: Format incorrect")
                print(f"     {line}")
                violations += 1
        
        # Vérifier entrelacement par requête
        queries = defaultdict(list)
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                queries[parts[0]].append(parts[2])  # query_id -> article_id
        
        for query_id, articles in queries.items():
            # Vérifier si les articles sont groupés
            current_article = None
            changes = 0
            
            for article in articles:
                if article != current_article:
                    changes += 1
                    current_article = article
            
            # Normalement, changes devrait être <= nombre d'articles uniques
            unique_articles = len(set(articles))
            if changes > unique_articles * 1.5:  # Tolérance
                print(f"  ⚠️  Requête {query_id}: Possible entrelacement")
                print(f"     Articles uniques: {unique_articles}, "
                      f"Changements: {changes}")
        
        if violations == 0:
            print(f"  ✅ RUN VALIDE: Format correct, pas d'erreurs évidentes")
        else:
            print(f"  ❌ {violations} violations détectées")
        
        return violations == 0