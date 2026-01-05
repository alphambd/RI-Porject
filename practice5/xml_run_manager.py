import os
import time
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import hashlib

from indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

class INEXRunGenerator:
    def __init__(self, cache_dir="data/cache", team_name="AlphaAnaClement"):
        self.cache_dir = cache_dir
        self.team_name = team_name
        os.makedirs(cache_dir, exist_ok=True)
    
    # ==================== GESTION CACHE & INDEX ====================
    
    def _get_cache_key(self, config_type: str, params: Dict) -> str:
        """Génère une clé de cache unique"""
        params_str = str(sorted(params.items()))
        key = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{config_type}_{key}"
    
    def create_or_load_index(self, xml_dir: str, index_type: str, 
                            config: Dict, max_files: int = None) -> Dict:
        """
        Crée ou charge un index depuis le cache.
        Vérifie si la configuration target_tags correspond au cache.
        """
        cache_key = self._get_cache_key(f"{index_type}_index", config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Vérifier si le cache existe ET si target_tags correspond
        cache_valid = False
        if os.path.exists(cache_file):
            try:
                # Vérifier la configuration du cache
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                cached_config = cache_data.get('config', {})
                cached_target_tags = set(cached_config.get('target_tags', []))
                current_target_tags = set(config.get('target_tags', []))
                
                # Le cache est valide si les target_tags sont SUBSET ou EQUAL
                # (on peut charger un cache avec moins de tags, mais pas l'inverse)
                if current_target_tags.issubset(cached_target_tags):
                    print(f"Chargement {index_type} depuis cache...")
                    index = WeightedInvertedIndex.load_from_file(cache_file)
                    cache_valid = True
                else:
                    print(f"Cache incompatible: tags mismatch")
                    print(f"  Cache: {sorted(cached_target_tags)}")
                    print(f"  Requis: {sorted(current_target_tags)}")
                    
            except Exception as e:
                print(f"Cache corrompu ({e}), recalcul...")
        
        if not cache_valid:
            print(f"Création nouvel index {index_type}...")
            # Créer un nouvel index
            index = WeightedInvertedIndex()
            
            # Configurer sans target_tags d'abord
            base_config = {k: v for k, v in config.items() if k != 'target_tags'}
            index.configure(**base_config)
            
            # Ajouter target_tags si présent
            if 'target_tags' in config:
                index.target_tags = set(config['target_tags'])
            
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
        else:
            # Cache valide chargé
            return {
                'index': index,
                'indexing_time': 0,
                'config': config
            }
        
    # ==================== FONCTIONS UTILITAIRES ====================
    
    def _create_element_cache(self, browse_index):
        """
        Crée un cache des éléments avec DEBUG.
        """
        article_to_elements = defaultdict(list)
        element_details = {}
        
        print(f"[DEBUG] Création cache pour {len(browse_index.doc_ids)} éléments")
        
        sample_count = 0
        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))
            
            # DEBUG: Afficher quelques exemples
            if sample_count < 5:
                print(f"  Élément: {elem_id[:20]}... → Parent: {parent_id}")
                sample_count += 1
            
            if parent_id:
                article_to_elements[parent_id].append(elem_id)
                
                xml_path = metadata.get('xml_path', '')
                tag = self._extract_tag_from_xpath(xml_path, metadata.get('tag', 'unknown'))
                
                element_details[elem_id] = {
                    'xml_path': xml_path,
                    'tag': tag,
                    'parent_id': parent_id,
                    'priority': self._get_tag_priority(tag)
                }
        
        print(f"[DEBUG] Cache créé: {len(article_to_elements)} articles uniques")
        print(f"[DEBUG] Exemples d'articles: {list(article_to_elements.keys())[:5]}")
        
        return article_to_elements, element_details

    def _extract_tag_from_xpath(self, xml_path: str, default_tag: str = 'unknown') -> str:
        """Extrait le tag principal d'un chemin XML"""
        if '/p[' in xml_path:
            return 'p'
        elif '/sec[' in xml_path:
            return 'sec'
        elif '/bdy[' in xml_path:
            return 'bdy'
        elif xml_path == '/article[1]' or xml_path.endswith('/article[1]'):
            return 'article'
        else:
            return default_tag
    
    def _get_tag_priority(self, tag: str) -> int:
        """Retourne la priorité selon la hiérarchie p > sec > bdy > article"""
        priorities = {'p': 4, 'sec': 3, 'bdy': 2, 'article': 1}
        return priorities.get(tag, 0)
    
    def _normalize_xml_path(self, xml_path: str) -> str:
        """Normalise un chemin XML pour format INEX"""
        if xml_path.startswith('/article['):
            return xml_path
        
        if '/article' in xml_path:
            # Extraire la partie après /article
            parts = xml_path.split('/article', 1)
            return f"/article[1]{parts[-1]}"
        
        # Chemin relatif, ajouter /article[1] au début
        if xml_path.startswith('/'):
            return f"/article[1]{xml_path}"
        
        return f"/article[1]/{xml_path}"
    
    def _get_xpath_indices(self, xml_path: str) -> Tuple[int, ...]:
        """Extrait les indices d'un XPath pour tri par ordre document"""
        indices = []
        for part in xml_path.split('/'):
            if part and '[' in part:
                try:
                    idx = int(part.split('[')[1].split(']')[0])
                    indices.append(idx)
                except (ValueError, IndexError):
                    indices.append(0)
        return tuple(indices)
    
    # ==================== PHASE BROWSE - SÉLECTION ÉLÉMENTS ====================
    
    def _score_elements_for_article(self,
                                  article_id: str,
                                  query_terms: List[str],
                                  element_ids: List[str],
                                  element_details: Dict,
                                  browse_ranker,
                                  weighting_scheme: str,
                                  min_element_score: float,
                                  k1: float = 1.2,
                                  b: float = 0.75) -> List[Dict]:
        """Calcule les scores pour tous les éléments d'un article"""
        scored_elements = []
        
        for elem_id in element_ids:
            elem_info = element_details.get(elem_id, {})
            
            # Calcul du score
            score = 0.0
            for term in query_terms:
                weight = browse_ranker.get_term_weight(
                    term, elem_id,
                    weighting_scheme=weighting_scheme,
                    k1=k1 if weighting_scheme == 'bm25' else None,
                    b=b if weighting_scheme == 'bm25' else None
                )
                score += weight if weight else 0.0
            
            # Filtrer par seuil minimum
            if score >= min_element_score:
                scored_elements.append({
                    'element_id': elem_id,
                    'score': score,
                    'tag': elem_info.get('tag', 'unknown'),
                    'xml_path': elem_info.get('xml_path', '/article[1]'),
                    'priority': elem_info.get('priority', 0)
                })
        
        return scored_elements
    
    def _select_best_element_hierarchy(self,
                                     elements: List[Dict],
                                     strategy: str = 'hierarchical',
                                     max_elements_per_article: int = 1,
                                     avoid_overlaps: bool = True) -> List[Dict]:
        """
        Sélectionne le(s) meilleur(s) élément(s) selon différentes stratégies.
        """
        if not elements:
            return []
        
        if strategy == 'hierarchical':
            return self._select_by_hierarchy(elements, max_elements_per_article, avoid_overlaps)
        elif strategy == 'top_score':
            return self._select_by_top_score(elements, max_elements_per_article, avoid_overlaps)
        elif strategy == 'coverage':
            return self._select_by_coverage(elements, max_elements_per_article)
        else:
            return self._select_by_hierarchy(elements, max_elements_per_article, avoid_overlaps)
    
    def _select_by_hierarchy(self, elements: List[Dict], max_elements: int, avoid_overlaps: bool) -> List[Dict]:
        """Stratégie hiérarchique: p > sec > bdy > article"""
        # Trier par priorité puis score
        elements.sort(key=lambda x: (-x['priority'], -x['score']))
        
        selected = []
        taken_paths = set()
        
        for elem in elements:
            if len(selected) >= max_elements:
                break
            
            xml_path = elem['xml_path']
            
            # Vérifier overlaps si demandé
            if avoid_overlaps:
                conflict = False
                for taken in taken_paths:
                    if self._paths_overlap(xml_path, taken):
                        conflict = True
                        break
                if conflict:
                    continue
            
            selected.append(elem)
            taken_paths.add(xml_path)
        
        return selected
    
    def _select_by_top_score(self, elements: List[Dict], max_elements: int, avoid_overlaps: bool) -> List[Dict]:
        """Stratégie: meilleurs scores uniquement"""
        elements.sort(key=lambda x: -x['score'])
        
        selected = []
        taken_paths = set()
        
        for elem in elements:
            if len(selected) >= max_elements:
                break
            
            xml_path = elem['xml_path']
            
            if avoid_overlaps:
                conflict = False
                for taken in taken_paths:
                    if self._paths_overlap(xml_path, taken):
                        conflict = True
                        break
                if conflict:
                    continue
            
            selected.append(elem)
            taken_paths.add(xml_path)
        
        return selected
    
    def _select_by_coverage(self, elements: List[Dict], max_elements: int) -> List[Dict]:
        """Stratégie: couverture du document (premiers éléments)"""
        elements.sort(key=lambda x: self._get_xpath_indices(x['xml_path']))
        return elements[:max_elements]
    
    def _paths_overlap(self, path1: str, path2: str) -> bool:
        """Vérifie si deux chemins XML ont une relation parent-enfant"""
        if path1 == path2:
            return True
        if path1.startswith(path2 + '/') or path2.startswith(path1 + '/'):
            return True
        return False
    
    # ==================== FONCTION PRINCIPALE FETCH & BROWSE ====================
    
    def generate_fetch_browse(self,
                                          run_id: str,
                                          xml_dir: str,
                                          queries: Dict[int, str],
                                          fetch_config: Dict,
                                          browse_config: Dict,
                                          run_params: Dict = None) -> str:
        """
        Version optimisée de fetch & browse avec paramètres configurables.
        """
        # Paramètres par défaut
        default_params = {
            'top_articles': 1000,
            'max_elements': 1500,
            'max_elements_per_article': 1,
            'weighting_scheme': 'ltn',
            'min_element_score': 0.00001,
            'selection_strategy': 'hierarchical',
            'avoid_overlaps': True,
            'fallback_to_article': True,
            'bm25_k1': 5,
            'bm25_b': 0.3
        }
        
        if run_params:
            default_params.update(run_params)
        run_params = default_params
        
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN {run_id}")
        print(f"Stratégie: {run_params['selection_strategy']}")
        print('='*70)
        
        total_start = time.time()
        
        # 1. PHASE FETCH
        print("[PHASE FETCH] Chargement index articles...")
        fetch_data = self.create_or_load_index(xml_dir, 'article', fetch_config)
        fetch_index = fetch_data['index']
        fetch_ranker = RankedRetrieval(fetch_index)
        
        # 2. PHASE BROWSE
        print("[PHASE BROWSE] Chargement index éléments...")
        if 'target_tags' in browse_config and 'article' not in browse_config['target_tags']:
            browse_config['target_tags'].append('article')
        
        browse_data = self.create_or_load_index(xml_dir, 'element', browse_config)
        browse_index = browse_data['index']
        browse_ranker = RankedRetrieval(browse_index)
        
        
        # 3. Cache des éléments
        print("[CACHE] Création du cache des éléments...")
        article_to_elements, element_details = self._create_element_cache(browse_index)
        print(f"  Articles dans cache: {len(article_to_elements)}")
        print(f"  Éléments totaux: {len(element_details)}")
        
        # 4. Générer nom de fichier
        filename = self._generate_run_filename(run_id, fetch_config, browse_config, run_params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        results_count = 0
        query_stats = {}
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                
                print(f"\n[Query {query_id}] {query_text[:60]}...")
                
                # A. FETCH: Récupérer top articles
                top_articles = fetch_ranker.search_query(
                    query_text,
                    weighting_scheme=run_params['weighting_scheme'],
                    top_k=run_params['top_articles'],
                    k1=run_params.get('bm25_k1', 1.2) if run_params['weighting_scheme'] == 'bm25' else None,
                    b=run_params.get('bm25_b', 0.75) if run_params['weighting_scheme'] == 'bm25' else None
                )
                
                print(f"  FETCH: {len(top_articles)} articles")
                
                # B. Prétraitement termes de la requête
                query_terms = browse_ranker.process_query_terms(query_text)
                
                # C. BROWSE: Traiter chaque article
                all_selected_elements = []
                articles_processed = 0
                articles_with_elements = 0
                
                for article_id, article_score in top_articles:
                    articles_processed += 1
                    
                    str_article_id = str(article_id)
                    
                    # Récupérer éléments de cet article
                    element_ids = article_to_elements.get(str_article_id, [])
                    
                    if element_ids:
                        # Calculer scores pour tous les éléments
                        scored_elements = self._score_elements_for_article(
                            article_id=str_article_id,
                            query_terms=query_terms,
                            element_ids=element_ids,
                            element_details=element_details,
                            browse_ranker=browse_ranker,
                            weighting_scheme=run_params['weighting_scheme'],
                            min_element_score=run_params['min_element_score'],
                            k1=run_params.get('bm25_k1', 1.2),
                            b=run_params.get('bm25_b', 0.75)
                        )
                        
                        # Sélectionner meilleur(s) élément(s)
                        selected = self._select_best_element_hierarchy(
                            elements=scored_elements,
                            strategy=run_params['selection_strategy'],
                            max_elements_per_article=run_params['max_elements_per_article'],
                            avoid_overlaps=run_params['avoid_overlaps']
                        )
                        
                        if selected:
                            articles_with_elements += 1
                            for elem in selected:
                                all_selected_elements.append({
                                    'article_id': article_id,
                                    'xml_path': elem['xml_path'],
                                    'score': elem['score'],
                                    'tag': elem['tag']
                                })
                    
                    # Fallback: article entier
                    if (run_params['fallback_to_article'] and 
                        article_score >= run_params['min_element_score'] and
                        not element_ids):
                        all_selected_elements.append({
                            'article_id': article_id,
                            'xml_path': '/article[1]',
                            'score': article_score,
                            'tag': 'article'
                        })
                
                # D. Trier et limiter les résultats
                all_selected_elements.sort(key=lambda x: -x['score'])
                final_elements = all_selected_elements[:run_params['max_elements']]
                
                # E. Regrouper par article pour écriture ordonnée
                grouped_by_article = defaultdict(list)
                for elem in final_elements:
                    grouped_by_article[elem['article_id']].append(elem)
                
                # Trier les articles par score maximum
                sorted_articles = sorted(
                    grouped_by_article.items(),
                    key=lambda x: max(e['score'] for e in x[1]),
                    reverse=True
                )
                
                # F. Écrire résultats groupés par article
                rank = 1
                for article_id, elements in sorted_articles:
                    # Trier les éléments par ordre document puis score
                    elements.sort(key=lambda x: (
                        self._get_xpath_indices(x['xml_path']),
                        -x['score']
                    ))
                    
                    for elem in elements:
                        xml_path = self._normalize_xml_path(elem['xml_path'])
                        
                        f.write(
                            f"{query_id} Q0 {article_id} {rank} "
                            f"{elem['score']:.6f} {self.team_name} {xml_path}\n"
                        )
                        rank += 1
                        results_count += 1
                
                # Statistiques de la requête
                tags_count = defaultdict(int)
                for elem in final_elements:
                    tags_count[elem['tag']] += 1
                
                query_time = time.time() - query_start
                query_stats[query_id] = {
                    'articles_processed': articles_processed,
                    'articles_with_elements': articles_with_elements,
                    'elements_found': len(final_elements),
                    'tags': dict(tags_count),
                    'time': query_time
                }
                
                print(f"  BROWSE: {articles_with_elements} articles avec éléments")
                print(f"  RÉSULTATS: {len(final_elements)} éléments")
                print(f"  Temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # 5. Afficher statistiques finales
        self._print_final_stats(filename, results_count, total_time, query_stats, run_params)
        
        # 6. Valider le run
        self._validate_run_file(filename)
        
        return filename
    
    def _generate_run_filename(self, run_id: str, fetch_config: Dict, 
                             browse_config: Dict, run_params: Dict) -> str:
        """Génère un nom de fichier descriptif pour le run"""
        os.makedirs("data/runs", exist_ok=True)
        
        # Extraire tags cibles
        target_tags = browse_config.get('target_tags', ['sec', 'p', 'bdy'])
        if 'article' in target_tags:
            target_tags.remove('article')
        tags_str = '-'.join(sorted(target_tags)) if target_tags else 'none'
        
        # Construction du nom
        parts = [
            self.team_name,
            run_id,
            f"fetch-{fetch_config.get('stemmer', 'nostem')}-{fetch_config.get('stop_words', 'nostop')}",
            f"browse-{tags_str}",
            run_params['weighting_scheme'],
            run_params['selection_strategy'][:3],  # Abréviation
            f"arts{run_params['top_articles']}",
            f"el{run_params['max_elements_per_article']}"
        ]
        
        if run_params['weighting_scheme'] == 'bm25':
            k1 = run_params.get('bm25_k1', 1.2)
            b = run_params.get('bm25_b', 0.75)
            parts.append(f"k{k1}b{b}")
        
        filename = '_'.join(parts) + ".txt"
        return os.path.join("data/runs", filename)
    
    def _print_final_stats(self, filename: str, results_count: int, total_time: float,
                         query_stats: Dict, run_params: Dict):
        """Affiche les statistiques finales du run"""
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {os.path.basename(filename)}")
        print(f"Temps total: {total_time:.2f}s")
        print(f"Résultats totaux: {results_count}")
        
        # Statistiques agrégées
        total_elements = 0
        tag_distribution = defaultdict(int)
        
        for qid, stats in query_stats.items():
            total_elements += stats['elements_found']
            for tag, count in stats['tags'].items():
                tag_distribution[tag] += count
        
        print(f"\nDistribution des tags:")
        for tag in ['p', 'sec', 'bdy', 'article']:
            count = tag_distribution.get(tag, 0)
            percentage = (count / total_elements * 100) if total_elements > 0 else 0
            print(f"  {tag}: {count} ({percentage:.1f}%)")
        
        print('='*70)
    
    def _validate_run_file(self, filename: str) -> bool:
        """Valide la conformité du fichier run"""
        print(f"\n[VALIDATION du run]")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            violations = 0
            
            # Vérifier format de base
            for i, line in enumerate(lines[:10]):
                parts = line.split()
                if len(parts) != 7:
                    print(f"  Err. Ligne {i+1}: format incorrect")
                    violations += 1
                    break
            
            # Vérifier nombre de résultats
            by_query = defaultdict(list)
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    query_id = parts[0]
                    by_query[query_id].append(line)
            
            print(f"  Requêtes traitées: {len(by_query)}")
            for query_id, query_lines in by_query.items():
                print(f"  Requête {query_id}: {len(query_lines)} résultats")
            
            if violations == 0:
                print(f"   RUN VALIDE")
                return True
            else:
                print(f"  Att. {violations} violations")
                return False
                
        except Exception as e:
            print(f"   Erreur... validation: {e}")
            return False
    
    # ==================== MÉTHODE POUR EXERCICES 1-2 ====================
    
    def generate_article_run(self, 
                        xml_dir: str,
                        queries: Dict[int, str],
                        config: Dict = None,
                        run_id: str = "article_run",
                        weighting_scheme: str = "ltn",
                        k1: float = None,
                        b: float = None) -> str:
        """
        Génère un run pour articles (exercices 1-2).
        Ajout des paramètres de pondération.
        """
        if config is None:
            config = {
                'tokenization': 'basic',
                'stemmer': 'nostem',
                'stop_words': 'nostop'
            }
        
        print(f"\n{'='*70}")
        print(f"EXERCICE 1-2: Run articles - {weighting_scheme.upper()}")
        print(f"Config: stemmer={config.get('stemmer', 'nostem')}, "
            f"stop={config.get('stop_words', 'nostop')}")
        print('='*70)
        
        # Créer index articles
        index_data = self.create_or_load_index(xml_dir, 'article', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Générer nom de fichier
        stemmer = config.get('stemmer', 'nostem')
        stop_words = config.get('stop_words', 'nostop')
        
        # Construire nom selon format INEX
        filename = f"{self.team_name}_{run_id}_{weighting_scheme}_article_{stop_words}_{stemmer}"
        
        # Ajouter paramètres BM25 si nécessaire
        if weighting_scheme == 'bm25':
            k1_val = k1 if k1 is not None else 1.2
            b_val = b if b is not None else 0.75
            filename += f"_k_{k1_val}_b_{b_val}"
        
        filename += ".txt"
        filename = os.path.join("data/runs", filename)
        os.makedirs("data/runs", exist_ok=True)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche sur articles avec la bonne pondération
                if weighting_scheme == 'bm25':
                    top_articles = ranker.search_query(
                        query_text,
                        weighting_scheme='bm25',
                        top_k=1500,
                        k1=k1 if k1 is not None else 1.2,
                        b=b if b is not None else 0.75
                    )
                else:
                    top_articles = ranker.search_query(
                        query_text,
                        weighting_scheme=weighting_scheme,
                        top_k=1500
                    )
                
                # **CORRECTION: S'assurer d'avoir 1500 résultats**
                if len(top_articles) < 1500:
                    print(f"   Seulement {len(top_articles)} résultats, ajout de compléments...")
                    
                    all_docs = set(index.doc_ids)
                    used_docs = set(doc_id for doc_id, _ in top_articles)
                    remaining = list(all_docs - used_docs)[:1500-len(top_articles)]
                    
                    for doc_id in remaining:
                        top_articles.append((doc_id, 0.000001))
                
                # Écrire résultats
                rank = 1
                for article_id, score in top_articles[:1500]:
                    f.write(
                        f"{query_id} Q0 {article_id} {rank} "
                        f"{score:.6f} {self.team_name} /article[1]\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  {len(top_articles[:1500])} articles écrits")
        
        print(f"\n{'='*70}")
        print(f"RUN ARTICLES TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Vérification: {results_count} lignes (attendues: {7*1500}=10500)")
        print('='*70)
        
        return filename

    # ==================== MÉTHODE POUR EXERCICES 3 fetch and browse ====================

    def generate_exercise3_fetch_browse(self, xml_dir, queries, with_article = False):
        """Version fetch & browse adaptée pour exercice 3"""
        
        # Config EXACTE exercice 3
        fetch_config = {'stemmer': 'nostem', 'stop_words': 'nostop'}
        browse_config = {
            'stemmer': 'nostem', 
            'stop_words': 'nostop',
            'target_tags': ['bdy', 'sec', 'p']  # Pas 'article'!
        }
        
        # Paramètres STRICTS
        run_params = {
            'top_articles': 2000,  # Large pour être sûr
            'max_elements': 1500,
            'max_elements_per_article': 1,          # 1 seul élément
            'weighting_scheme': 'ltn',              # SMART ltn
            'fallback_to_article': with_article,    # si False : PAS d'article
            # Forcer hiérarchie p > sec > bdy
            'selection_strategy': 'hierarchical',
            'avoid_overlaps': True,
        }
        
        # Générer avec fetch & browse
        filename = self.generate_fetch_browse_run_optimized(
            run_id="testXML_adapted",  # ← Contient testXML
            xml_dir=xml_dir,
            queries=queries,
            fetch_config=fetch_config,
            browse_config=browse_config,
            run_params=run_params
        )
        
        # Renommer pour correspondre au format demandé
        new_filename = filename.replace("_adapted", "")
        os.rename(filename, new_filename)
        
        return new_filename


    # ==================== MÉTHODE POUR EXERCICE 3 sans fetch and browse ====================
    
    def generate_element_run_simple(self,
                                  xml_dir: str,
                                  queries: Dict[int, str],
                                  config: Dict = None) -> str:
        """
        Version simplifiée pour l'exercice 3: recherche directe sur éléments.
        """
        if config is None:
            config = {
                'tokenization': 'basic',
                'stemmer': 'nostem',
                'stop_words': 'nostop',
                'target_tags': ['bdy', 'sec', 'p']
            }
        
        print(f"\n{'='*70}")
        print("EXERCICE 3: Indexation XML éléments (bdy, sec, p)")
        print('='*70)
        
        # Créer index éléments
        index_data = self.create_or_load_index(xml_dir, 'element', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)
        
        # Générer nom de fichier
        target_tags_str = '-'.join(sorted(config['target_tags']))
        filename = f"{self.team_name}_testXML_ltn_element-{target_tags_str}_nostop_nostem.txt"
        filename = os.path.join("data/runs", filename)
        os.makedirs("data/runs", exist_ok=True)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # Recherche directe sur éléments
                top_elements = ranker.search_query(
                    query_text,
                    weighting_scheme='ltn',
                    top_k=1500
                )
                
                rank = 1
                for elem_id, score in top_elements:
                    metadata = index.get_metadata(elem_id)
                    article_id = metadata.get('parent_doc_id', 'unknown')
                    xml_path = metadata.get('xml_path', '/article[1]')
                    xml_path = self._normalize_xml_path(xml_path)
                    
                    f.write(
                        f"{query_id} Q0 {article_id} {rank} "
                        f"{score:.6f} {self.team_name} {xml_path}\n"
                    )
                    rank += 1
                    results_count += 1
                
                print(f"  {len(top_elements)} éléments")
        
        print(f"\n{'='*70}")
        print(f"EXERCICE 3 TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print('='*70)
        
        return filename
    