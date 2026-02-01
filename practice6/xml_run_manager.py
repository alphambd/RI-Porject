import os
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import hashlib
import xml.etree.ElementTree as ET
import re

from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument
from ranked_retrieval import RankedRetrieval


class INEXRunGenerator:
    def __init__(self, cache_dir="data/cache", team_name="AlphaAnaClement"):
        self.cache_dir = cache_dir
        self.team_name = team_name

        self.pagerank_scores = None # pour le score pagerank

        os.makedirs(cache_dir, exist_ok=True)

    # ==================== GESTION CACHE ====================

    def _get_cache_key(self, config_type: str, params: Dict) -> str:
        """Génère une clé de cache unique basée sur la configuration."""
        params_str = str(sorted(params.items()))
        key = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"{config_type}_{key}"

    def create_or_load_index(self, xml_dir: str, index_type: str,
                            config: Dict, max_files: int = None) -> Dict:
        """
        Crée ou charge un index depuis le cache.
        Vérifie la compatibilité des target_tags.
        """
        cache_key = self._get_cache_key(f"{index_type}_index", config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                cached_config = cache_data.get('config', {})
                cached_tags = set(cached_config.get('target_tags', []))
                current_tags = set(config.get('target_tags', []))

                if current_tags.issubset(cached_tags):
                    print(f"Chargement {index_type} depuis cache...")
                    index = WeightedInvertedIndex.load_from_file(cache_file)
                    return {'index': index, 'indexing_time': 0, 'config': config}
                else:
                    print(f"Cache incompatible: tags mismatch")
            except Exception as e:
                print(f"Cache corrompu ({e}), recalcul...")

        print(f"Création nouvel index {index_type}...")
        index = WeightedInvertedIndex()

        base_config = {k: v for k, v in config.items() if k != 'target_tags'}
        index.configure(**base_config)

        if 'target_tags' in config:
            index.target_tags = set(config['target_tags'])

        start_time = time.time()

        if index_type == 'article':
            indexing_time = index.build_index_from_xml_collection(xml_dir, max_files)
        else:
            target_tags = config.get('target_tags', ['sec', 'p', 'bdy'])
            indexing_time = index.build_index_from_xml_elements(xml_dir, target_tags, max_files)

        try:
            index.save_to_file(cache_file)
            print(f"Index {index_type} sauvegardé dans le cache")
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")

        return {'index': index, 'indexing_time': indexing_time, 'config': config}

    def compute_or_load_pagerank(self, xml_dir: str,
                                damping: float = 0.85,
                                max_iter: int = 50):
        """
        Calcule le PageRank sur la collection INEX (articles).
        """
        if self.pagerank_scores is not None:
            return self.pagerank_scores

        print("\n[PageRank] Construction du graphe...")
        graph = extract_inex_link_graph(xml_dir)

        print("[PageRank] Calcul des scores...")
        pr = compute_pagerank(
        #pr = compute_pagerank_optimized(
            graph,
            damping=damping,
            max_iter=max_iter
        )

        print("[PageRank] Normalisation...")
        self.pagerank_scores = normalize_scores(pr)

        print("[PageRank] Terminé ✓")
        return self.pagerank_scores

    # ==================== FONCTIONS UTILITAIRES ====================

    def _create_element_cache(self, browse_index):
        """Crée un cache des éléments avec leurs métadonnées."""
        article_to_elements = defaultdict(list)
        element_details = {}

        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))

            if parent_id:
                article_to_elements[parent_id].append(elem_id)
                element_details[elem_id] = {
                    'xml_path': metadata.get('xml_path', '/article[1]'),
                    'tag': metadata.get('tag', 'unknown'),
                    'element_text_size':metadata.get('element_text_size', 0),
                    'parent_id': parent_id
                }

        return article_to_elements, element_details

    def _get_xpath_indices(self, xml_path: str) -> Tuple[int, ...]:
        """
        Extrait les indices d'un XPath pour tri par ordre document.
        Exemple: /article[1]/bdy[1]/sec[3]/p[2] → (1, 1, 3, 2)
        """
        indices = []
        for part in xml_path.split('/'):
            if part and '[' in part and ']' in part:
                try:
                    # Extraire le contenu entre crochets
                    idx_str = part.split('[')[1].split(']')[0]
                    idx = int(idx_str)
                    indices.append(idx)
                except (ValueError, IndexError):
                    indices.append(1)  # Fallback
            elif part:  # Partie sans indice (ne devrait pas arriver)
                indices.append(1)
        
        return tuple(indices)

    def _are_paths_overlapping(self, path1: str, path2: str) -> bool:
        """Vérifie si deux chemins XML se chevauchent (ancêtre/descendant)."""
        if path1 == path2:
            return True

        norm1 = path1.strip('/')
        norm2 = path2.strip('/')
        parts1 = [p for p in norm1.split('/') if p]
        parts2 = [p for p in norm2.split('/') if p]

        min_len = min(len(parts1), len(parts2))
        for i in range(min_len):
            if parts1[i] != parts2[i]:
                return False

        return True

    def rerank_with_pagerank(self,
                            bm25_results,
                            pagerank_scores,
                            alpha: float = 0.9):
        """
        Combine BM25 et PageRank par interpolation linéaire.
        """
        reranked = []

        for doc_id, bm25_score in bm25_results:
            pr_score = pagerank_scores.get(str(doc_id), 0.0)
            final_score = alpha * bm25_score + (1 - alpha) * pr_score
            reranked.append((doc_id, final_score))

        reranked.sort(key=lambda x: -x[1])
        return reranked

    # ==================== SÉLECTION D'ÉLÉMENTS ====================

    def _score_elements_for_article(
        self,
        query_terms,
        element_ids,
        element_details,
        browse_ranker,
        weighting_scheme,
        min_element_score,
        k1 = 1.2,
        b = 0.75
    ):
        scored_elements = []

        for elem_id in element_ids:
            elem_info = element_details[elem_id]
            xml_path = elem_info['xml_path']
            tag = elem_info['tag']

            score = 0.0
            for term in query_terms:
                score += browse_ranker.get_term_weight(
                    term,
                    elem_id,
                    weighting_scheme=weighting_scheme,
                    k1=k1 if weighting_scheme == 'bm25' else None,
                    b=b if weighting_scheme == 'bm25' else None
                ) or 0.0

            if score >= min_element_score:
                scored_elements.append({
                    'element_id': elem_id,
                    'score': score,
                    'tag': tag,
                    'xml_path': xml_path
                })

        return scored_elements

    def _extract_tag_from_xpath(self, xml_path: str) -> str:
        """Extrait le tag final d'un chemin XML."""
        if not xml_path or xml_path == '/article[1]':
            return 'article'
        
        # Trouver le dernier segment du chemin
        parts = [p for p in xml_path.split('/') if p]
        if not parts:
            return 'article'
        
        last_part = parts[-1]
        # Extraire le tag (avant les crochets)
        if '[' in last_part:
            return last_part.split('[')[0]
        return last_part

    def _normalize_xml_path_for_output(self, xml_path: str) -> str:
        """Normalise un chemin XML pour le format INEX."""
        if not xml_path or xml_path == '/':
            return '/article[1]'
        
        # S'assurer qu'il commence par /
        if not xml_path.startswith('/'):
            xml_path = '/' + xml_path
        
        # S'assurer que /article[1] est présent
        if not xml_path.startswith('/article['):
            if xml_path.startswith('/article'):
                # Cas comme /article/bdy[1] -> /article[1]/bdy[1]
                xml_path = xml_path.replace('/article/', '/article[1]/', 1)
            else:
                xml_path = '/article[1]' + ('' if xml_path.startswith('/') else '/') + xml_path
        
        # Nettoyer les doubles slashes
        xml_path = xml_path.replace('//', '/')
        
        return xml_path

    def select_elements_score_plus_depth(
        self,
        elements: List[Dict],
        bonus_by_tag: Dict[str, float],
        max_elements: int = 2,
        avoid_overlaps: bool = True
    ) -> List[Dict]:
        """
        Sélectionne les éléments en combinant score et spécificité (profondeur).
        """
        if not elements:
            return []

        adjusted_elements = []

        for elem in elements:
            tag = elem.get('tag', 'article')
            bonus = bonus_by_tag.get(tag, 1.0)

            adjusted_score = elem['score'] * bonus

            adjusted_elem = elem.copy()
            adjusted_elem['adjusted_score'] = adjusted_score
            adjusted_elements.append(adjusted_elem)

        # Trier par score ajusté décroissant
        adjusted_elements.sort(key=lambda x: -x['adjusted_score'])

        selected = []
        selected_paths = []

        for elem in adjusted_elements:
            if len(selected) >= max_elements:
                break

            xml_path = elem['xml_path']

            if avoid_overlaps:
                conflict = False
                for taken in selected_paths:
                    if self._are_paths_overlapping(xml_path, taken):
                        conflict = True
                        break
                if conflict:
                    continue

            selected.append(elem)
            selected_paths.append(xml_path)

        return selected

    # ==================== MÉTHODE PRINCIPALE FETCH & BROWSE ====================

    def generate_fetch_browse(self, run_id: str, xml_dir: str,
                            queries: Dict[int, str], fetch_config: Dict,
                            browse_config: Dict, run_params: Dict = None, bonus_tags: Dict = None) -> str:
        """Implémente la stratégie fetch & browse optimisée."""
        default_params = {
            'top_articles': 1000,
            'max_elements': 1500,
            'max_elements_per_article': 2,
            'weighting_scheme': 'ltn',
            'min_element_score': 0.000001,
            'avoid_overlaps': True,
            'fallback_to_article': True,
            'bm25_k1': 1.2,
            'bm25_b': 0.75,

            'use_pagerank': False,
            'pagerank_alpha': 0.9
        }
        
        if run_params:
            default_params.update(run_params)
        params = default_params

        # Bonus par tag pour la sélection score + profondeur
        if bonus_tags is None:
            bonus_tags = {
                'bdy': 1.0,
                'sec': 1.5,
                'p':   1.8
            }
        bonus_by_tag = bonus_tags
        
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN {run_id}")
        print(f"Pondération: {params['weighting_scheme']}")
        print(f"Éléments/article: {params['max_elements_per_article']}")
        print(f"Éviter chevauchements: {params['avoid_overlaps']}")
        print('='*70)
        
        total_start = time.time()
        
        # PHASE FETCH
        print("[PHASE FETCH] Index articles...")
        fetch_data = self.create_or_load_index(xml_dir, 'article', fetch_config)
        fetch_index = fetch_data['index']
        fetch_ranker = RankedRetrieval(fetch_index)

        # Charger PageRank si demandé
        pagerank_scores = None
        if params['use_pagerank']:
            pagerank_scores = self.compute_or_load_pagerank(xml_dir)
        
        # PHASE BROWSE
        print("[PHASE BROWSE] Index éléments...")
        browse_data = self.create_or_load_index(xml_dir, 'element', browse_config)
        browse_index = browse_data['index']
        browse_ranker = RankedRetrieval(browse_index)
        
        # Cache des éléments
        print("[CACHE] Construction du cache...")
        article_to_elements, element_details = self._create_element_cache(browse_index)
        print(f"  Articles: {len(article_to_elements)}, Éléments: {len(element_details)}")
        
        # Génération du fichier
        filename = self._generate_run_filename(run_id, fetch_config, browse_config, params)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        results_count = 0
        query_stats = {}
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                print(f"\n[Query {query_id}] {query_text[:60]}...")
                
                # FETCH: articles pertinents
                top_articles = fetch_ranker.search_query(
                    query_text,
                    weighting_scheme=params['weighting_scheme'],
                    top_k=params['top_articles'],
                    k1=params['bm25_k1'] if params['weighting_scheme'] == 'bm25' else None,
                    b=params['bm25_b'] if params['weighting_scheme'] == 'bm25' else None
                )

                # RE-RANKING AVEC PAGERANK
                if params['use_pagerank']:
                    top_articles = self.rerank_with_pagerank(
                        top_articles,
                        pagerank_scores,
                        alpha=params['pagerank_alpha']
                    )
                print(f"  FETCH: {len(top_articles)} articles")
                
                # BROWSE: éléments pertinents
                query_terms = browse_ranker.process_query_terms(query_text)
                if not query_terms:
                    continue
                
                all_selected_elements = []
                articles_with_elements = 0
                
                for article_id, article_score in top_articles:
                    str_article_id = str(article_id)
                    element_ids = article_to_elements.get(str_article_id, [])
                    
                    if element_ids:
                        # Calculer les scores pour tous les éléments
                        scored_elements = self._score_elements_for_article(
                            #str_article_id, query_terms, element_ids,
                            query_terms, element_ids,
                            element_details, browse_ranker,
                            params['weighting_scheme'],
                            params['min_element_score'],
                            params['bm25_k1'], params['bm25_b']
                        )

                        # Sélectionner les meilleurs éléments
                        selected = self.select_elements_score_plus_depth(
                            scored_elements,
                            bonus_by_tag=bonus_by_tag,
                            max_elements=params['max_elements_per_article'],
                            avoid_overlaps=params['avoid_overlaps']
                        )
                        
                        if selected:
                            articles_with_elements += 1
                            for elem in selected:
                                all_selected_elements.append({
                                    'article_id': article_id,
                                    'xml_path': elem['xml_path'],
                                    'score': elem['score'],
                                    'tag': elem['tag'],
                                    'element_text_size': element_details[elem['element_id']]['element_text_size'],
                                })
                    
                    # Fallback: article entier si besoin
                    if (params['fallback_to_article'] and
                        article_score >= params['min_element_score'] and
                        not any(e['article_id'] == article_id for e in all_selected_elements)):
                        all_selected_elements.append({
                            'article_id': article_id,
                            'xml_path': '/article[1]',
                            'score': article_score,
                            'tag': 'article'
                        })
                
                # CORRECTION CRITIQUE : Éviter l'interleaving
                # 1. Limiter d'abord par score global
                all_selected_elements.sort(key=lambda x: -x['score'])
                final_elements = all_selected_elements[:params['max_elements']]
                
                # 2. REGROUPER PAR ARTICLE (IMPORTANT !)
                grouped_by_article = defaultdict(list)
                for elem in final_elements:
                    grouped_by_article[elem['article_id']].append(elem)
                
                # 3. Trier les articles par score MAXIMUM
                sorted_articles = sorted(
                    grouped_by_article.items(),
                    key=lambda x: max(e['score'] for e in x[1]),  # Score max de l'article
                    reverse=True
                )
                
                # 4. ÉCRIRE TOUS LES ÉLÉMENTS D'UN ARTICLE ENSEMBLE
                rank = 1
                character_total_doc = 0
                character_total_element = 0
                depth_total_element = 0
                total_element = 0
                total_article = 0
                for article_id, elements in sorted_articles:
                    # À l'intérieur d'un article, trier par chemin XML (ordre document)
                    elements.sort(key=lambda x: self._get_xpath_indices(x['xml_path']))

                    total_article += 1
                    character_total_doc += fetch_index.doc_lengths_char[article_id]
                    
                    for elem in elements:
                        xml_path = self._normalize_xml_path_for_output(elem['xml_path'])

                        # Compter les caractères (élément vs document)
                        if (elem['tag'] != 'article'):
                            character_total_element+=elem['element_text_size']
                        else:
                            character_total_element += fetch_index.doc_lengths_char[article_id]
                        total_element += 1
                        depth_total_element += xml_path.count('/')

                        f.write(f"{query_id} Q0 {article_id} {rank} "
                                f"{elem['score']:.6f} {self.team_name} {xml_path}\n")
                        rank += 1
                        results_count += 1
                
                # Statistiques de la requête
                tags_count = defaultdict(int)
                for elem in final_elements:
                    tags_count[elem['tag']] += 1
                
                query_time = time.time() - query_start
                query_stats[query_id] = {
                    'articles_with_elements': articles_with_elements,
                    'elements_found': len(final_elements),
                    'tags': dict(tags_count),
                    'time': query_time
                }
                
                print(f"  BROWSE: {articles_with_elements} articles avec éléments")
                print(f"  RATE TARGET: {round(character_total_element/character_total_doc*100,2)}% ")
                print(f"  AVERAGE DEPTH: {round(depth_total_element/total_element,2)} depth")
                print(f"  ARTICLE COUNT: {total_article} articles")
                print(f"  RÉSULTATS: {len(final_elements)} éléments")
                print(f"  Temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        # Afficher les statistiques
        self._print_run_stats(filename, results_count, total_time, query_stats)
        
        # Validation spécifique pour éviter l'interleaving
        self._validate_no_interleaving(filename)
        
        return filename

    def _generate_run_filename(self, run_id: str, fetch_config: Dict,
                               browse_config: Dict, run_params: Dict) -> str:
        """Génère un nom de fichier descriptif."""
        os.makedirs("data/runs", exist_ok=True)

        target_tags = browse_config.get('target_tags', ['sec', 'p', 'bdy'])
        if 'article' in target_tags:
            target_tags.remove('article')
        tags_str = '-'.join(sorted(target_tags)) if target_tags else 'none'

        parts = [
            self.team_name,
            run_id,
            f"fetch-{fetch_config.get('stemmer', 'nostem')}-{fetch_config.get('stop_words', 'nostop')}",
            f"browse-{tags_str}",
            run_params['weighting_scheme'],
            f"arts{run_params['top_articles']}",
            f"elem{run_params['max_elements_per_article']}"
        ]

        if run_params['weighting_scheme'] == 'bm25':
            k1 = run_params['bm25_k1']
            b = run_params['bm25_b']
            parts.append(f"k{k1:.1f}b{b:.2f}")

        filename = '_'.join(parts) + ".txt"
        return os.path.join("data/runs", filename)

    def _validate_no_interleaving(self, filename: str) -> bool:
        """Vérifie qu'il n'y a pas d'interleaving dans le run."""
        print(f"\n[VALIDATION INTERLEAVING]")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            errors = []
            current_article = None
            query_articles = {}  # Pour chaque requête, garder trace des articles vus
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                query_id = parts[0]
                article_id = parts[2]
                
                if query_id not in query_articles:
                    query_articles[query_id] = {'current': None, 'seen': set()}
                
                query_info = query_articles[query_id]
                
                if article_id != query_info['current']:
                    # Si on change d'article, vérifier qu'on n'y est pas déjà retourné
                    if article_id in query_info['seen']:
                        errors.append(f"Query {query_id}: Article {article_id} interleaved à la ligne {i+1}")
                    else:
                        if query_info['current']:
                            query_info['seen'].add(query_info['current'])
                        query_info['current'] = article_id
            
            if errors:
                print(f"  ERREURS D'INTERLEAVING DÉTECTÉES:")
                for error in errors[:5]:  # Montrer seulement les 5 premières
                    print(f"    {error}")
                if len(errors) > 5:
                    print(f"    ... et {len(errors)-5} autres erreurs")
                return False
            else:
                print("  AUCUN INTERLEAVING DÉTECTÉ ✓")
                return True
                
        except Exception as e:
            print(f"  Erreur validation: {e}")
            return False
        
    def _print_run_stats(self, filename: str, results_count: int,
                         total_time: float, query_stats: Dict):
        """Affiche les statistiques du run."""
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {os.path.basename(filename)}")
        print(f"Temps total: {total_time:.2f}s")
        print(f"Résultats totaux: {results_count}")

        total_elements = sum(stats['elements_found'] for stats in query_stats.values())
        tag_distribution = defaultdict(int)

        for stats in query_stats.values():
            for tag, count in stats['tags'].items():
                tag_distribution[tag] += count

        print("\nDistribution des tags:")
        for tag in ['p', 'sec', 'bdy', 'article']:
            count = tag_distribution.get(tag, 0)
            percentage = (count / total_elements * 100) if total_elements > 0 else 0
            print(f"  {tag}: {count} ({percentage:.1f}%)")

        print('='*70)

    # ==================== MÉTHODES POUR LES EXERCICES ====================

    def generate_article_run(self, xml_dir: str, queries: Dict[int, str],
                             config: Dict = None, run_id: str = "article_run",
                             weighting_scheme: str = "ltn",
                             k1: float = None, b: float = None) -> str:
        """Génère un run pour articles (exercices 1-2)."""
        if config is None:
            config = {
                'tokenization': 'basic',
                'stemmer': 'nostem',
                'stop_words': 'nostop'
            }
        
        # Paramètres par défaut BM25
        if weighting_scheme == 'bm25':
            if k1 is None:
                k1 = 1.2
            if b is None:
                b = 0.75
        
        print(f"\n{'='*70}")
        print(f"RUN ARTICLES - {weighting_scheme.upper()}")
        print('='*70)

        index_data = self.create_or_load_index(xml_dir, 'article', config)
        index = index_data['index']
        ranker = RankedRetrieval(index)

        stemmer = config.get('stemmer', 'nostem')
        stop_words = config.get('stop_words', 'nostop')
        filename = f"{self.team_name}_{run_id}_{weighting_scheme}_article_{stop_words}_{stemmer}"

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

                if weighting_scheme == 'bm25':
                    top_articles = ranker.search_query(
                        query_text, weighting_scheme='bm25', top_k=1500,
                        k1=k1 if k1 is not None else 1.2,
                        b=b if b is not None else 0.75
                    )
                else:
                    top_articles = ranker.search_query(
                        query_text, weighting_scheme=weighting_scheme, top_k=1500
                    )
                
                rank = 1
                for article_id, score in top_articles[:1500]:
                    f.write(f"{query_id} Q0 {article_id} {rank} "
                            f"{score:.6f} {self.team_name} /article[1]\n")
                    rank += 1
                    results_count += 1

                print(f"  {len(top_articles[:1500])} articles écrits")

        print(f"\nRUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Attendu: {7 * 1500}")
        return filename

    def generate_article_run_with_pagerank(
        self,
        xml_dir: str,
        queries: Dict[int, str],
        config: Dict,
        run_id: str = "article_PR",
        weighting_scheme: str = "bm25",
        top_k: int = 1500,
        pagerank_alpha: float = 0.85,
        k1: float = 1.2,
        b: float = 0.75
    ) -> str:
        """
        Exercice 4 – Articles runs exploiting links (PageRank).
        """

        print(f"\n{'='*70}")
        print("EXERCICE 4 — ARTICLE RUN + PAGERANK")
        print('='*70)

        # 1 Index articles
        index_data = self.create_or_load_index(xml_dir, "article", config)
        index = index_data["index"]
        ranker = RankedRetrieval(index)

        # 2 Construire le graphe + stats
        print("\n[LINK GRAPH]")
        
        graph = extract_inex_link_graph(xml_dir)
        """
        graph, stats = extract_inex_link_graph(xml_dir)

        print("Statistiques du graphe :")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        """
        # 3 Calcul PageRank
        print("\n[PAGERANK] Calcul...")
        #pr_scores = compute_pagerank(graph, damping=0.85, max_iter=50)
        #pr_scores = compute_pagerank_optimized(graph, damping=0.85, max_iter=50)
        pr_scores = compute_pagerank(graph, damping=0.85, max_iter=50)
        pr_scores = normalize_scores(pr_scores)

        print("[PAGERANK] Terminé ✓")

        # 4 Nom du fichier
        filename = (
            f"{self.team_name}_{run_id}_"
            f"{config.get('stop_words', 'nostop')}_"
            f"{config.get('stemmer', 'none')}_"
            f"{weighting_scheme}_article_pagerank.txt"
        )
        filename = os.path.join("data/runs", filename)
        os.makedirs("data/runs", exist_ok=True)

        # 5 Génération du run
        total_results = 0

        with open(filename, "w", encoding="utf-8") as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:60]}...")

                # Recherche BM25
                if weighting_scheme == "bm25":
                    results = ranker.search_query(
                        query_text,
                        weighting_scheme="bm25",
                        top_k=top_k,
                        k1=k1,
                        b=b
                    )
                else:
                    results = ranker.search_query(
                        query_text,
                        weighting_scheme=weighting_scheme,
                        top_k=top_k
                    )

                # Combinaison BM25 + PageRank
                reranked = []
                """
                for doc_id, bm25_score in results:
                    pr = pr_scores.get(str(doc_id), 0.0)
                    final_score = (
                        pagerank_alpha * bm25_score
                        + (1 - pagerank_alpha) * pr
                    )
                    reranked.append((doc_id, final_score))

                reranked.sort(key=lambda x: -x[1])
                """
                reranked = self.rerank_with_pagerank(results, pr_scores, pagerank_alpha)

                
                # Écriture du run (articles uniquement)
                rank = 1
                for doc_id, score in reranked[:top_k]:
                    f.write(
                        f"{query_id} Q0 {doc_id} {rank} "
                        f"{score:.6f} {self.team_name} /article[1]\n"
                    )
                    rank += 1
                    total_results += 1

                print(f"  {rank-1} articles écrits")

        print(f"\nRUN TERMINÉ: {filename}")
        print(f"Résultats totaux: {total_results}")
        print(f"Attendu: {len(queries) * top_k}")

        return filename

    def generate_article_run_with_anchors(
        self,
        xml_dir: str,
        queries: Dict[int, str],
        config: Dict,
        run_id: str = "article_anchor",
        top_k: int = 1500,
        alpha_content: float = 1.0,
        alpha_anchor: float = 0.7,
        k1: float = 1.2,
        b: float = 0.75
    ) -> str:
        """
        Exercice 5 — Articles runs exploiting anchor texts (BM25F).
        """

        print(f"\n{'='*70}")
        print("EXERCICE 5 — ARTICLE RUN + ANCHORS (BM25F)")
        print('='*70)

        # 1 Extraction des anchor texts
        print("\n[ANCHORS] Extraction des ancres entrantes...")
        anchor_texts = extract_anchor_texts(xml_dir)
        print(f"  Articles avec ancres: {len(anchor_texts)}")

        # 2 Index articles (classique)
        index_data = self.create_or_load_index(xml_dir, "article", config)
        index = index_data["index"]
        ranker = RankedRetrieval(index)

        # 3 Nom du fichier
        filename = (
            f"{self.team_name}_{run_id}_"
            f"bm25f_article_anchor.txt"
        )
        filename = os.path.join("data/runs", filename)
        os.makedirs("data/runs", exist_ok=True)

        total_results = 0

        with open(filename, "w", encoding="utf-8") as f:
            for query_id, query_text in queries.items():
                print(f"\n[Query {query_id}] {query_text[:60]}...")

                # BM25 sur contenu
                content_results = ranker.search_query(
                    query_text,
                    weighting_scheme="bm25",
                    top_k=top_k,
                    k1=k1,
                    b=b
                )

                scores = defaultdict(float)

                # 4 Score contenu
                for doc_id, score in content_results:
                    scores[doc_id] += alpha_content * score

                # 5 Score ancres
                query_terms = ranker.process_query_terms(query_text)

                for doc_id, anchor_text in anchor_texts.items():
                    anchor_score = 0.0
                    for term in query_terms:
                        tf = anchor_text.lower().count(term)
                        anchor_score += tf

                    if anchor_score > 0:
                        scores[int(doc_id)] += alpha_anchor * anchor_score

                # 6 Tri final
                ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

                rank = 1
                for doc_id, score in ranked:
                    f.write(
                        f"{query_id} Q0 {doc_id} {rank} "
                        f"{score:.6f} {self.team_name} /article[1]\n"
                    )
                    rank += 1
                    total_results += 1

                print(f"  {rank-1} articles écrits")

        print(f"\nRUN TERMINÉ: {filename}")
        print(f"Résultats totaux: {total_results}")
        print(f"Attendu: {len(queries) * top_k}")

        return filename

    def generate_element_run(self, xml_dir: str, queries: Dict[int, str],
                            config: Dict = None) -> str:
        """Génère un run pour éléments XML (exercice 3)."""
        if config is None:
            config = {
                'tokenization': 'basic',
                'stemmer': 'nostem',
                'stop_words': 'nostop',
                'target_tags': ['bdy', 'sec', 'p']
            }
        
        print(f"\n{'='*70}")
        print("EXERCICE 3: Indexation XML éléments")
        print('='*70)
                
        # Créer index avec fetch & browse
        fetch_config = {
            'tokenization': 'basic',
            'stemmer': 'nostem',
            'stop_words': 'nostop'
        }
        
        browse_config = config.copy()
        
        # Paramètres optimisés pour favoriser p et sec
        run_params = {
            'top_articles': 800,
            'max_elements': 1500,
            'max_elements_per_article': 5,  # Prendre plus d'éléments par article
            'weighting_scheme': 'ltn',
            'min_element_score': 0.000001,
            'avoid_overlaps': True,
            'fallback_to_article': False,  # Pas d'articles entiers pour l'exercice 3
            'bm25_k1': 1.2,
            'bm25_b': 0.75
        }
        
        filename = self.generate_fetch_browse(
            run_id="testXML",
            xml_dir=xml_dir,
            queries=queries,
            fetch_config=fetch_config,
            browse_config=browse_config,
            run_params=run_params
        )
        
        # Renommer pour correspondre au format demandé
        target_tags_str = '-'.join(sorted(config['target_tags']))
        new_filename = f"{self.team_name}_testXML_ltn_element-{target_tags_str}_nostop_nostem.txt"
        new_full_path = os.path.join("data/runs", new_filename)
        
        # Copier plutôt que renommer
        import shutil
        shutil.copy2(filename, new_full_path)
        
        print(f"\nEXERCICE 3 TERMINÉ: {new_filename}")
        
        # Afficher un échantillon du résultat
        print("\nÉCHANTILLON DU RUN:")
        try:
            with open(new_full_path, 'r') as f:
                lines = f.readlines()[:20]
                for line in lines:
                    print(f"  {line.strip()}")
        except:
            pass
        
        return new_full_path


# ==================== SÉLECTIONS POUR PAGERANK ====================
def extract_inex_link_graph(xml_dir: str):
    
    #Construit un graphe {doc_id: set(out_links)}
    
    graph = defaultdict(set)
    all_docs = set()

    for root, _, files in os.walk(xml_dir):
        for file in files:
            if not file.endswith(".xml"):
                continue

            doc_id = file.replace(".xml", "")
            all_docs.add(doc_id)

            file_path = os.path.join(root, file)
            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for link in root_xml.iter("link"):
                    href = link.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                    match = re.search(r"/(\d+)\.xml", href)
                    if match:
                        target_id = match.group(1)
                        graph[doc_id].add(target_id)

            except Exception:
                continue

    # S'assurer que tous les docs sont présents
    for d in all_docs:
        graph.setdefault(d, set())

    return graph


"""
def extract_inex_link_graph(xml_dir: str):
    #Construit le graphe des liens INEX entre articles
    #et calcule des statistiques détaillées.
    

    graph = defaultdict(set)
    all_docs = set()

    # Statistiques
    total_links = 0
    article_to_article_links = 0
    external_links = 0
    internal_refs = 0
    parse_errors = 0

    for root, _, files in os.walk(xml_dir):
        for file in files:
            if not file.endswith(".xml"):
                continue

            doc_id = file.replace(".xml", "")
            all_docs.add(doc_id)

            file_path = os.path.join(root, file)

            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for link in root_xml.iter("link"):
                    total_links += 1

                    href = link.attrib.get(
                        "{http://www.w3.org/1999/xlink}href", ""
                    )

                    # Lien interne (ancre ou xpointer)
                    if href.startswith("#"):
                        internal_refs += 1
                        continue

                    # Lien vers un autre article INEX
                    match = re.search(r"/(\d+)\.xml", href)
                    if match:
                        target_id = match.group(1)
                        article_to_article_links += 1
                        graph[doc_id].add(target_id)
                    else:
                        external_links += 1

            except Exception:
                parse_errors += 1
                continue

    # S'assurer que tous les articles sont présents
    for d in all_docs:
        graph.setdefault(d, set())

    stats = {
        "num_articles": len(all_docs),
        "total_links": total_links,
        "article_to_article_links": article_to_article_links,
        "external_links": external_links,
        "internal_refs": internal_refs,
        "parse_errors": parse_errors,
        "avg_out_degree": (
            sum(len(v) for v in graph.values()) / len(graph)
            if graph else 0.0
        ),
        "max_out_degree": max((len(v) for v in graph.values()), default=0),
        "min_out_degree": min((len(v) for v in graph.values()), default=0),
    }

    return graph, stats
"""

"""
def extract_inex_link_graph(xml_dir: str):
    graph = defaultdict(set)
    all_docs = set()

    # Statistiques
    total_links = 0
    article_to_article_links = 0
    external_links = 0
    internal_refs = 0
    parse_errors = 0
    # NOUVELLE STAT : liens avec format différent
    different_formats = defaultdict(int)

    for root, _, files in os.walk(xml_dir):
        for file in files:
            if not file.endswith(".xml"):
                continue

            doc_id = file.replace(".xml", "")
            all_docs.add(doc_id)

            file_path = os.path.join(root, file)

            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for link in root_xml.iter("link"):
                    total_links += 1

                    href = link.attrib.get(
                        "{http://www.w3.org/1999/xlink}href", ""
                    )

                    # Lien interne (ancre ou xpointer)
                    if href.startswith("#"):
                        internal_refs += 1
                        continue

                    # Lien vers un autre article INEX
                    # MODIFICATION : regex plus permissive
                    match = re.search(r"/(\d+)\.xml", href)
                    if not match:
                        # Essayer une autre regex
                        match = re.search(r"(\d+)\.xml$", href)
                    
                    if match:
                        target_id = match.group(1)
                        article_to_article_links += 1
                        graph[doc_id].add(target_id)
                        
                        # Analyser le format pour le rapport
                        if href.startswith("../"):
                            different_formats["../xxx/"] += 1
                        elif href.startswith("/"):
                            different_formats["/xxx/"] += 1
                        elif "/" not in href:
                            different_formats["direct"] += 1
                        else:
                            different_formats["autre"] += 1
                    else:
                        external_links += 1

            except Exception as e:
                parse_errors += 1
                print(f"Erreur parsing {file}: {e}")
                continue

    # S'assurer que tous les articles sont présents
    for d in all_docs:
        graph.setdefault(d, set())

    stats = {
        "num_articles": len(all_docs),
        "total_links": total_links,
        "article_to_article_links": article_to_article_links,
        "external_links": external_links,
        "internal_refs": internal_refs,
        "parse_errors": parse_errors,
        "link_formats": dict(different_formats),  # NOUVEAU
        "avg_out_degree": (
            sum(len(v) for v in graph.values()) / len(graph)
            if graph else 0.0
        ),
        "max_out_degree": max((len(v) for v in graph.values()), default=0),
        "min_out_degree": min((len(v) for v in graph.values()), default=0),
    }

    # VALIDATION CRITIQUE
    print("\n" + "="*50)
    print("VALIDATION DE L'EXTRACTION DES LIENS")
    print("="*50)
    print(f"Articles trouvés: {stats['num_articles']}")
    print(f"Liens totaux: {stats['total_links']}")
    print(f"Liens article→article: {stats['article_to_article_links']}")
    print(f"Liens externes: {stats['external_links']}")
    print(f"Références internes: {stats['internal_refs']}")
    
    # Vérifier si on a bien ~100,000 liens article→article
    expected = 100000
    actual = stats['article_to_article_links']
    if abs(actual - expected) / expected > 0.1:  # ±10%
        print(f"  ATTENTION: {actual} liens article→article (attendu ~{expected})")
        print("   Vérifiez l'extraction des liens!")
    else:
        print(f"  OK: {actual} liens article→article (proche de {expected})")
    
    return graph, stats
"""
def compute_pagerank(
    graph,
    damping=0.85,
    max_iter=50,
    tol=1e-6
):
    """
    PageRank standard avec téléportation
    """
    nodes = list(graph.keys())
    N = len(nodes)

    pr = {n: 1.0 / N for n in nodes}

    for _ in range(max_iter):
        new_pr = {}
        diff = 0.0

        for node in nodes:
            incoming_sum = 0.0

            for src in nodes:
                if node in graph[src]:
                    out_degree = len(graph[src])
                    if out_degree > 0:
                        incoming_sum += pr[src] / out_degree

            new_pr[node] = (1 - damping) / N + damping * incoming_sum
            diff += abs(new_pr[node] - pr[node])

        pr = new_pr
        if diff < tol:
            break

    return pr

def compute_pagerank_optimized(
    graph,
    damping=0.85,
    max_iter=50,
    tol=1e-6
):
    nodes = list(graph.keys())
    N = len(nodes)

    pr = {n: 1.0 / N for n in nodes}

    for _ in range(max_iter):
        new_pr = {n: (1 - damping) / N for n in nodes}
        diff = 0.0

        for src, out_links in graph.items():
            if not out_links:
                continue

            share = damping * pr[src] / len(out_links)
            for dst in out_links:
                new_pr[dst] += share

        for n in nodes:
            diff += abs(new_pr[n] - pr[n])

        pr = new_pr
        if diff < tol:
            break

    return pr

def normalize_scores(scores: dict):
    max_val = max(scores.values())
    min_val = min(scores.values())

    if max_val == min_val:
        return {k: 1.0 for k in scores}

    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }

def extract_anchor_texts(xml_dir: str):
    """
    Construit un mapping:
    article_id -> texte des ancres ENTRANTES
    """
    anchor_texts = defaultdict(list)

    for root, _, files in os.walk(xml_dir):
        for file in files:
            if not file.endswith(".xml"):
                continue

            source_id = file.replace(".xml", "")
            file_path = os.path.join(root, file)

            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                for link in root_xml.iter("link"):
                    href = link.attrib.get(
                        "{http://www.w3.org/1999/xlink}href", ""
                    )

                    match = re.search(r"/(\d+)\.xml", href)
                    if not match:
                        continue

                    target_id = match.group(1)

                    # Texte de l'ancre
                    anchor = "".join(link.itertext()).strip()
                    if anchor:
                        anchor_texts[target_id].append(anchor)

            except Exception:
                continue

    # Concaténer les ancres par article
    anchor_texts = {
        doc_id: " ".join(texts)
        for doc_id, texts in anchor_texts.items()
    }

    return anchor_texts

