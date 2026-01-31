import os
import time
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import hashlib

from advanced_indexer import WeightedInvertedIndex
from inex_document import INEXDocument
from ranked_retrieval import RankedRetrieval


class INEXRunGenerator:
    def __init__(self, cache_dir="data/cache", team_name="AlphaAnaClement"):
        self.cache_dir = cache_dir
        self.team_name = team_name
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

    def _extract_tag_from_xpath(self, xml_path: str) -> str:
        """Extrait le tag principal d'un chemin XML."""
        if '/p[' in xml_path:
            return 'p'
        elif '/sec[' in xml_path:
            return 'sec'
        elif '/bdy[' in xml_path:
            return 'bdy'
        elif xml_path.endswith('/article[1]') or xml_path == '/article[1]':
            return 'article'
        else:
            return 'unknown'

    def _normalize_xml_path(self, xml_path: str) -> str:
        """Normalise un chemin XML pour le format INEX."""
        if xml_path.startswith('/article['):
            return xml_path

        if '/article' in xml_path:
            parts = xml_path.split('/article', 1)
            return f"/article[1]{parts[-1]}"

        if xml_path.startswith('/'):
            return f"/article[1]{xml_path}"

        return f"/article[1]/{xml_path}"

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

    # ==================== SÉLECTION D'ÉLÉMENTS ====================

    def _score_elements_for_article(self, article_id: str, query_terms: List[str],
                                    element_ids: List[str], element_details: Dict,
                                    browse_ranker, weighting_scheme: str,
                                    min_element_score: float = 0.0,
                                    k1: float = 1.2, b: float = 0.75) -> List[Dict]:
        """Calcule les scores des éléments pour une requête."""
        scored_elements = []
        
        for elem_id in element_ids:
            elem_info = element_details.get(elem_id, {})
            xml_path = elem_info.get('xml_path', '/article[1]')
            tag = self._extract_tag_from_xpath(xml_path)
            
            # Calculer le score
            score = 0.0
            for term in query_terms:
                weight = browse_ranker.get_term_weight(
                    term, elem_id,
                    weighting_scheme=weighting_scheme,
                    k1=k1 if weighting_scheme == 'bm25' else None,
                    b=b if weighting_scheme == 'bm25' else None
                )
                if weight:
                    score += weight
            
            # IMPORTANT: Bonus différentiel selon le tag
            # Les éléments plus spécifiques (p) devraient avoir un petit avantage
            tag_bonus = {
                'p': 1, #1.2,    # +20% pour les paragraphes
                'sec': 1, #1.1,  # +10% pour les sections
                'bdy': 1.0, #,  # Pas de bonus pour les body
                'article': 1 #0.9  # Petit malus pour les articles entiers
            }
            
            score *= tag_bonus.get(tag, 1.0)
            
            # Bonus supplémentaire pour la profondeur (spécificité)
            depth = xml_path.count('/')
            if depth > 3:  # Plus profond que /article[1]/bdy[1]
                score *= (1.0 + (depth - 3) * 0.05)  # +5% par niveau
            
            if score >= min_element_score:
                scored_elements.append({
                    'element_id': elem_id,
                    'score': score,
                    'tag': tag,
                    'xml_path': xml_path,
                    'depth': depth,
                    'text_length': elem_info.get('text_length', 0)
                })
        
        return scored_elements

    def _select_best_elements_by_score(self, elements: List[Dict],
                                    max_elements_per_article: int = 2,
                                    avoid_overlaps: bool = True) -> List[Dict]:
        """
        Sélectionne les éléments avec les meilleurs scores,
        en évitant les chevauchements.
        """
        if not elements:
            return []
        
        # Trier par score décroissant
        elements.sort(key=lambda x: -x['score'])
        
        selected = []
        taken_paths = set()
        
        for elem in elements:
            if len(selected) >= max_elements_per_article:
                break
            
            xml_path = elem['xml_path']
            
            # Vérifier les chevauchements
            if avoid_overlaps:
                conflict = False
                for taken_path in taken_paths:
                    # Un chemin est en conflit s'il est ancêtre ou descendant
                    if (xml_path.startswith(taken_path + '/') or 
                        taken_path.startswith(xml_path + '/')):
                        conflict = True
                        break
                
                if conflict:
                    continue
            
            selected.append(elem)
            taken_paths.add(xml_path)
        
        return selected

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

    def _select_optimal_elements(self, elements: List[Dict],
                                 max_elements_per_article: int = 1,
                                 avoid_overlaps: bool = True) -> List[Dict]:
        """
        Sélectionne les éléments avec les meilleurs scores
        tout en évitant les chevauchements.
        """
        if not elements:
            return []

        elements.sort(key=lambda x: -x['score'])
        selected = []
        taken_paths = set()

        for elem in elements:
            if len(selected) >= max_elements_per_article:
                break

            xml_path = elem['xml_path']

            if avoid_overlaps:
                conflict = any(self._are_paths_overlapping(xml_path, taken)
                              for taken in taken_paths)
                if conflict:
                    continue

            selected.append(elem)
            taken_paths.add(xml_path)

        return selected

    # ==================== MÉTHODE PRINCIPALE FETCH & BROWSE ====================

    def generate_fetch_browse(self, run_id: str, xml_dir: str,
                            queries: Dict[int, str], fetch_config: Dict,
                            browse_config: Dict, run_params: Dict = None) -> str:
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
            'bm25_b': 0.75
        }
        
        if run_params:
            default_params.update(run_params)
        params = default_params
        
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
                            str_article_id, query_terms, element_ids,
                            element_details, browse_ranker,
                            params['weighting_scheme'],
                            params['min_element_score'],
                            params['bm25_k1'], params['bm25_b']
                        )
                        
                        # Sélectionner les meilleurs éléments
                        selected = self._select_best_elements_by_score(
                            scored_elements,
                            params['max_elements_per_article'],
                            params['avoid_overlaps']
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

    def _validate_run_file(self, filename: str) -> bool:
        """Valide la conformité du fichier run."""
        print(f"\n[VALIDATION]")

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            by_query = defaultdict(list)
            for line in lines:
                parts = line.split()
                if len(parts) >= 1:
                    query_id = parts[0]
                    by_query[query_id].append(line)

            print(f"  Requêtes traitées: {len(by_query)}")
            for query_id, query_lines in by_query.items():
                print(f"  Requête {query_id}: {len(query_lines)} résultats")

            valid = all(len(parts) == 7 for parts in (line.split() for line in lines[:10]))
            if valid:
                print("  RUN VALIDE")
            else:
                print("  Problèmes détectés dans le format")

            return valid

        except Exception as e:
            print(f"  Erreur de validation: {e}")
            return False

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
                """
                # Compléter à 1500 résultats si nécessaire
                if len(top_articles) < 1500:
                    all_docs = set(index.doc_ids)
                    used_docs = set(doc_id for doc_id, _ in top_articles)
                    remaining = list(all_docs - used_docs)[:1500 - len(top_articles)]
                    for doc_id in remaining:
                        top_articles.append((doc_id, 0.000001))
                """
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
        
        # D'abord, tester l'extraction
        #test_element_extraction(xml_dir, sample_size=5)
        
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

    def test_element_extraction(self, xml_dir: str, sample_size: int = 10):
        """Teste l'extraction des éléments pour débogage."""
        import random
        
        xml_files = []
        for root_dir, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_files.append(os.path.join(root_dir, file))
        
        # Prendre un échantillon aléatoire
        sample_files = random.sample(xml_files, min(sample_size, len(xml_files)))
        
        print(f"\n{'='*70}")
        print(f"TEST D'EXTRACTION D'ÉLÉMENTS")
        print(f"Échantillon de {len(sample_files)} fichiers")
        print('='*70)
        
        stats = defaultdict(lambda: defaultdict(int))
        
        for i, xml_file in enumerate(sample_files):
            print(f"\nFichier {i+1}: {os.path.basename(xml_file)}")
            
            doc = INEXDocument(xml_file)
            if not doc.parse():
                print("  Échec du parsing")
                continue
            
            # Tester la nouvelle méthode
            elements = doc.get_all_elements_with_full_paths({'bdy', 'sec', 'p'})
            
            for elem in elements:
                tag = elem['tag']
                path = elem['xml_path']
                text_len = len(elem['text'])
                
                stats[tag]['count'] += 1
                stats[tag]['total_length'] += text_len
                
                # Afficher quelques exemples
                if stats[tag]['count'] <= 2:  # Montrer 2 exemples par tag
                    print(f"  {tag}: {path} (longueur: {text_len} chars)")
            
            print(f"  Total éléments: {len(elements)}")
        
        # Afficher les statistiques
        print(f"\n{'='*70}")
        print("STATISTIQUES D'EXTRACTION:")
        total_elements = sum(stats[tag]['count'] for tag in stats)
        
        for tag in ['bdy', 'sec', 'p']:
            count = stats[tag]['count']
            avg_len = stats[tag]['total_length'] / count if count > 0 else 0
            percentage = (count / total_elements * 100) if total_elements > 0 else 0
            
            print(f"  {tag}:")
            print(f"    Nombre: {count} ({percentage:.1f}%)")
            print(f"    Longueur moyenne: {avg_len:.0f} caractères")
        
        print('='*70)

