import os
import time
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple
import hashlib

from new_indexer import WeightedInvertedIndex
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
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Reconstruire l'index depuis le cache
                index = WeightedInvertedIndex()
                
                # Restaurer les données
                index.dictionary = defaultdict(dict, cache_data['dictionary'])
                index.doc_ids = cache_data['doc_ids']
                index.doc_lengths = cache_data['doc_lengths']
                index.doc_count = cache_data['doc_count']
                index.total_terms = cache_data['total_terms']
                index.metadata_store = cache_data['metadata_store']
                
                # Restaurer la configuration
                stored_config = cache_data['config']
                index.configure(**stored_config)
                
                # Si c'est un index d'éléments, restaurer target_tags
                if index_type == 'element' and 'target_tags' in config:
                    index.target_tags = config['target_tags']
                
                return {
                    'index': index,
                    'indexing_time': 0,  # Temps zéro car chargé depuis cache
                    'config': config
                }
            except Exception as e:
                print(f"Cache corrompu ({e}), recalcul...")
        
        # Créer un nouvel index
        index = WeightedInvertedIndex()
        
        # Configurer sans target_tags d'abord
        base_config = {k: v for k, v in config.items() if k != 'target_tags'}
        index.configure(**base_config)
        
        # Ajouter target_tags si présent (après configure)
        if 'target_tags' in config:
            index.target_tags = config['target_tags']
        
        start_time = time.time()
        
        if index_type == 'article':
            indexing_time = index.build_index_from_articles(xml_dir, max_files)
        else:  # 'element'
            target_tags = config.get('target_tags', ['sec', 'p', 'bdy'])
            indexing_time = index.build_index_from_elements(xml_dir, target_tags, max_files)
        
        # Préparer les données pour le cache (sans xml_cache)
        cache_data = index.get_cache_data()  # Utiliser la nouvelle méthode
        
        # Sauvegarder dans le cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Index {index_type} sauvegardé dans le cache")
        except Exception as e:
            print(f"Erreur sauvegarde cache: {e}")
        
        return {
            'index': index,
            'indexing_time': indexing_time,
            'config': config
        }
        
    def generate_fetch_browse_run_old(self, 
                                run_id: str,
                                xml_dir: str,
                                queries: Dict[int, str],
                                fetch_config: Dict,
                                browse_config: Dict,
                                run_params: Dict = None) -> str:
        """
        Génère un run INEX avec Fetch and Browse - VERSION CORRIGÉE
        """
        if run_params is None:
            run_params = {
                'top_articles': 500,
                'score_threshold': 0.0001,
                'max_elements': 1500,
                'weighting_scheme': 'ltn'
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
        
        # 3. Pré-calculer mapping article -> éléments
        article_to_elements = defaultdict(list)
        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))
            if parent_id:
                article_to_elements[parent_id].append(elem_id)
        
        # 4. Générer le fichier run
        team_name = "AlphaAnaClement"
        filename = self._generate_filename(team_name, run_id, fetch_config, browse_config, run_params)
        
        results_count = 0
        
        with open(filename, 'w', encoding='utf-8') as f:
            for query_id, query_text in queries.items():
                query_start = time.time()
                
                print(f"\n[Query {query_id}] {query_text[:50]}...")
                
                # A. FETCH: Sélectionner les articles
                top_articles = fetch_ranker.search_query(
                    query_text,
                    weighting_scheme=run_params['weighting_scheme'],
                    top_k=run_params['top_articles']
                )
                
                print(f"  FETCH: {len(top_articles)} articles")
                
                # B. BROWSE: Collecter les éléments
                all_elements = []
                query_terms = browse_ranker.process_query_terms(query_text)
                
                for article_idx, (article_id, article_score) in enumerate(top_articles):
                    if article_idx % 100 == 0 and article_idx > 0:
                        print(f"    Article {article_idx}/{len(top_articles)}...")
                    
                    str_article_id = str(article_id)
                    
                    if str_article_id in article_to_elements:
                        article_elements = []
                        
                        for elem_id in article_to_elements[str_article_id]:
                            # Calcul du score
                            score = 0.0
                            for term in query_terms:
                                weight = browse_ranker.get_term_weight(
                                    term, elem_id,
                                    weighting_scheme=run_params['weighting_scheme'],
                                    k1=run_params.get('k1', 1.2),
                                    b=run_params.get('b', 0.75)
                                )
                                score += weight if weight else 0.0
                            
                            if score >= run_params['score_threshold']:
                                metadata = browse_index.get_metadata(elem_id)
                                article_elements.append({
                                    'element_id': elem_id,
                                    'score': score,
                                    'xml_path': metadata.get('xml_path', '/article[1]'),
                                    'article_id': article_id
                                })
                        
                        # Limiter par article
                        if article_elements:
                            article_elements.sort(key=lambda x: -x['score'])
                            all_elements.extend(article_elements[:50])  # 50 max par article
                            #all_elements.extend(article_elements)
                
                print(f"  BROWSE: {len(all_elements)} éléments collectés")
                
                # C. Filtrer les overlaps
                all_elements.sort(key=lambda x: -x['score'])
                filtered_elements = self._remove_overlaps(
                    all_elements,
                    max_elements=run_params['max_elements']
                )
                
                # D. Grouper par article
                elements_by_article = defaultdict(list)
                for elem in filtered_elements:
                    elements_by_article[elem['article_id']].append(elem)
                
                # E. Écrire les résultats - RANK COMMENCE À 1 POUR CHAQUE REQUÊTE
                rank = 1  # IMPORTANT: réinitialisé pour chaque requête
                
                # Trier les articles par ordre alphabétique/numérique pour consistance
                sorted_article_ids = sorted(elements_by_article.keys())
                
                for article_id in sorted_article_ids:
                    # Trier les éléments de cet article par score décroissant
                    elements_by_article[article_id].sort(key=lambda x: -x['score'])
                    
                    for elem in elements_by_article[article_id]:
                        metadata = browse_index.get_metadata(elem['element_id'])
                        correct_article_id = metadata.get('doc_id', article_id)
                        xml_path = elem['xml_path']
                        
                        # S'assurer du format XML
                        if not xml_path.startswith('/article['):
                            xml_path = f"/article[1]{xml_path.split('/article', 1)[-1]}" if '/article' in xml_path else f"/article[1]{xml_path}"
                        
                        # Écrire la ligne
                        f.write(
                            f"{query_id} Q0 {correct_article_id} {rank} "
                            f"{elem['score']:.6f} {team_name} {xml_path}\n"
                        )
                        rank += 1
                        results_count += 1
                
                query_time = time.time() - query_start
                print(f"  RÉSULTATS: {len(filtered_elements)} éléments, {len(elements_by_article)} articles")
                print(f"  Temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Temps total: {total_time:.2f}s")
        print('='*70)
        
        return filename

    def _get_document_position(self, xml_path):
        """
        Extrait la position dans le document à partir du XPath
        Retourne une liste d'indices [section, paragraph, etc.]
        """
        # Exemple: /article[1]/bdy[1]/sec[6]/p[8] → [1, 1, 6, 8]
        indices = []
        parts = xml_path.split('/')
        
        for part in parts:
            if '[' in part and ']' in part:
                # Extraire le nombre entre [ et ]
                try:
                    idx = int(part.split('[')[1].split(']')[0])
                    indices.append(idx)
                except:
                    indices.append(0)
        
        return indices

    def _compare_document_order(self, path1, path2):
        """
        Compare deux chemins selon l'ordre du document
        Retourne -1 si path1 avant path2, 1 si après, 0 si égaux
        """
        pos1 = self._get_document_position(path1)
        pos2 = self._get_document_position(path2)
        
        # Comparer niveau par niveau
        for i in range(min(len(pos1), len(pos2))):
            if pos1[i] < pos2[i]:
                return -1  # path1 avant
            elif pos1[i] > pos2[i]:
                return 1   # path1 après
        
        # Si tous les indices communs égaux, le plus court est le parent (vient avant)
        if len(pos1) < len(pos2):
            return -1
        elif len(pos1) > len(pos2):
            return 1
        
        return 0

    def _get_xpath_indices(self, xml_path):
        """
        Convertit un XPath en tuple d'indices pour le tri
        Ex: /article[1]/bdy[1]/sec[6]/p[8] → (1, 1, 6, 8)
        """
        indices = []
        parts = [p for p in xml_path.split('/') if p]
        
        for part in parts:
            if '[' in part:
                try:
                    # Extraire le numéro entre [ et ]
                    num_str = part.split('[')[1].split(']')[0]
                    indices.append(int(num_str))
                except:
                    indices.append(0)
            else:
                indices.append(0)
        
        return tuple(indices)

    def _sort_by_document_order(self, elements):
        """
        Trie une liste d'éléments par ordre naturel du document
        """
        # Ajouter les indices de position à chaque élément
        for elem in elements:
            elem['_indices'] = self._get_xpath_indices(elem['xml_path'])
        
        # Trier par indices
        elements.sort(key=lambda x: x['_indices'])
        
        # Nettoyer
        for elem in elements:
            del elem['_indices']
        
        return elements

    def _select_best_elements_no_overlap(self, article_elements_map):
        """
        Sélectionne les meilleurs éléments SANS OVERLAP
        Retourne un seul élément par chemin non-overlapping
        """
        final_elements = []
        
        for article_id, elements in article_elements_map.items():
            if not elements:
                continue
            
            # Trier par priorité puis score
            elements.sort(key=lambda x: (-x['priority'], -x['score']))
            
            # Filtrer les overlaps
            selected_for_article = []
            taken_paths = set()
            
            for elem in elements:
                xml_path = elem['xml_path']
                conflict = False
                
                # Vérifier les overlaps avec les chemins déjà pris
                for taken in taken_paths:
                    # Overlap si: parent-enfant ou même chemin
                    if (xml_path == taken or 
                        xml_path.startswith(taken + '/') or 
                        taken.startswith(xml_path + '/')):
                        conflict = True
                        break
                
                if not conflict:
                    selected_for_article.append(elem)
                    taken_paths.add(xml_path)
            
            # Prendre seulement le MEILLEUR élément de cet article (sans overlap)
            if selected_for_article:
                final_elements.append({
                    'article_id': article_id,
                    'score': selected_for_article[0]['score'],
                    'xml_path': selected_for_article[0]['xml_path'],
                    'tag': selected_for_article[0]['tag']
                })
        
        return final_elements
    
    def generate_fetch_browse_run(self, 
                                        run_id: str,
                                        xml_dir: str,
                                        queries: Dict[int, str],
                                        fetch_config: Dict,
                                        browse_config: Dict,
                                        run_params: Dict = None) -> str:
        """
        Version CORRIGÉE qui respecte :
        1. Pas d'overlap (parent-enfant)
        2. Grouping par article (pas d'entrelacement)
        3. Priorité p > sec > bdy > article
        """
        """
        if run_params is None:
            run_params = {
                'top_articles': 2000,  # Plus pour compenser les filtres
                'score_threshold': 0.0,
                'max_elements': 1500,
                'weighting_scheme': 'ltn',
                'min_element_score': 0.00001
            }
        """
        print(f"\n{'='*70}")
        print(f"GÉNÉRATION RUN {run_id}")
        print('='*70)
        
        total_start = time.time()
        
        # 1. Phase FETCH
        fetch_data = self.create_or_load_index(xml_dir, 'article', fetch_config)
        fetch_index = fetch_data['index']
        fetch_ranker = RankedRetrieval(fetch_index)
        
        # 2. Phase BROWSE
        if 'target_tags' in browse_config and 'article' not in browse_config['target_tags']:
            browse_config['target_tags'].append('article')
        
        browse_data = self.create_or_load_index(xml_dir, 'element', browse_config)
        browse_index = browse_data['index']
        browse_ranker = RankedRetrieval(browse_index)
        
        # 3. Cache des éléments
        article_to_elements = defaultdict(list)
        element_details = {}
        
        for elem_id in browse_index.doc_ids:
            metadata = browse_index.get_metadata(elem_id)
            parent_id = str(metadata.get('parent_doc_id', ''))
            
            if parent_id:
                article_to_elements[parent_id].append(elem_id)
                
                # Déterminer le tag
                xml_path = metadata.get('xml_path', '')
                tag = 'unknown'
                if '/p[' in xml_path:
                    tag = 'p'
                elif '/sec[' in xml_path:
                    tag = 'sec'
                elif '/bdy[' in xml_path:
                    tag = 'bdy'
                elif xml_path == '/article[1]' or xml_path.endswith('/article[1]'):
                    tag = 'article'
                else:
                    tag = metadata.get('tag', 'unknown')
                
                element_details[elem_id] = {
                    'xml_path': xml_path,
                    'tag': tag,
                    'parent_id': parent_id
                }
        
        # 4. Générer le fichier
        team_name = "AlphaAnaClement"
        filename = self._generate_filename(team_name, run_id, fetch_config, browse_config, run_params)
        
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
                
                # B. Collecter TOUS les éléments pertinents
                all_elements_by_article = defaultdict(list)
                query_terms = browse_ranker.process_query_terms(query_text)
                
                for article_id, article_score in top_articles[:1000]:  # Limiter pour performance
                    str_article_id = str(article_id)
                    
                    if str_article_id in article_to_elements:
                        for elem_id in article_to_elements[str_article_id]:
                            # Score
                            score = 0.0
                            for term in query_terms:
                                weight = browse_ranker.get_term_weight(
                                    term, elem_id,
                                    weighting_scheme=run_params['weighting_scheme']
                                )
                                score += weight if weight else 0.0
                            
                            if score >= run_params['min_element_score']:
                                elem_info = element_details.get(elem_id, {})
                                tag = elem_info.get('tag', 'unknown')
                                
                                # Priorité
                                priority = {'p': 4, 'sec': 3, 'bdy': 2, 'article': 1}.get(tag, 0)
                                
                                all_elements_by_article[article_id].append({
                                    'element_id': elem_id,
                                    'score': score,
                                    'priority': priority,
                                    'tag': tag,
                                    'xml_path': elem_info.get('xml_path', '/article[1]')
                                })
                    
                    # Fallback: article entier si rien trouvé
                    if not all_elements_by_article[article_id]:
                        all_elements_by_article[article_id].append({
                            'element_id': f"{article_id}_article",
                            'score': article_score,
                            'priority': 1,
                            'tag': 'article',
                            'xml_path': '/article[1]'
                        })
                
                # C. Pour chaque article, sélectionner SANS OVERLAP
                best_elements_by_article = {}
                
                for article_id, elements in all_elements_by_article.items():
                    if not elements:
                        continue
                    
                    # Trier par priorité puis score
                    elements.sort(key=lambda x: (-x['priority'], -x['score']))
                    
                    # Prendre le meilleur SANS OVERLAP
                    selected = []
                    taken_paths = set()
                    
                    for elem in elements:
                        xml_path = elem['xml_path']
                        conflict = False
                        
                        # Vérifier overlap avec chemins déjà pris
                        for taken in taken_paths:
                            if (xml_path == taken or 
                                xml_path.startswith(taken + '/') or 
                                taken.startswith(xml_path + '/')):
                                conflict = True
                                break
                        
                        if not conflict:
                            selected.append(elem)
                            taken_paths.add(xml_path)
                            if len(selected) >= 3:  # Max 3 éléments par article
                                break
                    
                    if selected:
                        best_elements_by_article[article_id] = selected
                
                # D. Trier les articles par score maximum
                articles_sorted = sorted(
                    best_elements_by_article.items(),
                    key=lambda x: max(e['score'] for e in x[1]),
                    reverse=True
                )
                
                # E. Prendre jusqu'à 1500 éléments au total
                final_elements = []
                remaining = run_params['max_elements']
                
                for article_id, elements in articles_sorted:
                    if remaining <= 0:
                        break
                    
                    # Prendre jusqu'à 3 éléments par article (ou moins si reste peu)
                    take_count = min(len(elements), 3, remaining)
                    final_elements.extend([
                        {
                            'article_id': article_id,
                            'score': elem['score'],
                            'xml_path': elem['xml_path'],
                            'tag': elem['tag']
                        }
                        for elem in elements[:take_count]
                    ])
                    remaining -= take_count
                
                # F. ÉCRIRE GROUPÉ PAR ARTICLE (pas d'entrelacement)
                # Regrouper à nouveau par article pour l'écriture
                grouped_by_article = defaultdict(list)
                for elem in final_elements:
                    grouped_by_article[elem['article_id']].append(elem)
                
                # Trier les articles par score maximum
                articles_for_writing = sorted(
                    grouped_by_article.items(),
                    key=lambda x: max(e['score'] for e in x[1]),
                    reverse=True
                )
                
                rank = 1
                for article_id, elements in articles_for_writing:
                    # Trier les éléments de cet article par score
                    #elements.sort(key=lambda x: -x['score'])
                    # Trier par ORDRE DU DOCUMENT d'abord, puis score
                    elements.sort(key=lambda x: (
                        self._get_document_position(x['xml_path']),  # Ordre naturel
                        -x['score']  # Même position → meilleur score d'abord
                    ))

                    for elem in elements:
                        xml_path = elem['xml_path']
                        
                        # Formatage
                        if not xml_path.startswith('/article['):
                            if '/article' in xml_path:
                                xml_path = f"/article[1]{xml_path.split('/article', 1)[-1]}"
                            else:
                                xml_path = f"/article[1]{xml_path}"
                        
                        f.write(
                            f"{query_id} Q0 {article_id} {rank} "
                            f"{elem['score']:.6f} {team_name} {xml_path}\n"
                        )
                        rank += 1
                        results_count += 1
                
                # Statistiques
                tags_count = defaultdict(int)
                for elem in final_elements:
                    tags_count[elem['tag']] += 1
                
                print(f"  RÉSULTATS: {len(final_elements)} éléments, {len(grouped_by_article)} articles")
                print(f"    Tags: p={tags_count.get('p',0)} sec={tags_count.get('sec',0)} "
                    f"bdy={tags_count.get('bdy',0)} article={tags_count.get('article',0)}")
                
                query_time = time.time() - query_start
                print(f"  Temps: {query_time:.2f}s")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*70}")
        print(f"RUN TERMINÉ: {filename}")
        print(f"Total résultats: {results_count}")
        print(f"Temps total: {total_time:.2f}s")
        print('='*70)
        
        # Validation automatique
        self._validate_run_file(filename)
        
        return filename

    
    def _validate_run_file(self, filename):
        """Valide qu'un run respecte les règles INEX"""
        print(f"\n[VALIDATION de {filename}]")
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Par requête
        by_query = defaultdict(list)
        for line in lines:
            parts = line.split()
            if len(parts) >= 7:
                query_id = parts[0]
                article_id = parts[2]
                xml_path = parts[6]
                by_query[query_id].append((article_id, xml_path))
        
        violations = 0
        
        for query_id, elements in by_query.items():
            # Vérifier entrelacement
            current_article = None
            article_changes = 0
            
            for article_id, _ in elements:
                if article_id != current_article:
                    article_changes += 1
                    current_article = article_id
            
            # Plus de changements que d'articles = entrelacement
            unique_articles = len(set(article_id for article_id, _ in elements))
            if article_changes > unique_articles:
                print(f"  ❌ Requête {query_id}: ENTRELACEMENT détecté")
                print(f"     Articles uniques: {unique_articles}, Changements: {article_changes}")
                violations += 1
            
            # Vérifier overlaps par article
            by_article = defaultdict(list)
            for article_id, xml_path in elements:
                by_article[article_id].append(xml_path)
            
            for article_id, paths in by_article.items():
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        p1, p2 = paths[i], paths[j]
                        if (p1.startswith(p2 + '/') or p2.startswith(p1 + '/') or p1 == p2):
                            print(f"  ❌ Requête {query_id}, Article {article_id}: OVERLAP")
                            print(f"     {p1}")
                            print(f"     {p2}")
                            violations += 1
        
        if violations == 0:
            print(f"  ✅ RUN VALIDE: Aucune violation détectée")
        else:
            print(f"  ❌ {violations} violations détectées")
        
        return violations == 0
    

    def _remove_overlaps(self, elements: List[Dict], max_elements: int = 1500) -> List[Dict]:
        """
        Élimine les éléments qui se chevauchent selon les règles INEX.
        
        Règle INEX : Pas d'overlap entre éléments retournés.
        En pratique : Pas de relation parent-enfant dans le même article.
        
        Args:
            elements: Liste d'éléments triés par score décroissant
            max_elements: Nombre maximum d'éléments à retourner (1500 pour INEX)
        
        Returns:
            Liste filtrée sans overlaps
        """
        filtered = []
        taken_paths_by_article = defaultdict(set)  # {article_id: set(chemins_déjà_pris)}
        
        for elem in elements:
            # Arrêter si on a atteint la limite
            if len(filtered) >= max_elements:
                break
            
            xml_path = elem['xml_path']       # Ex: /article[1]/bdy[1]/sec[2]/p[1]
            article_id = elem['article_id']   # ID de l'article parent
            score = elem['score']             # Score de pertinence

            # Récupérer les chemins déjà pris pour cet article
            taken_paths = taken_paths_by_article[article_id]
            
            conflict = False
            
            # Vérifier chaque chemin déjà pris dans le même article
            for taken in taken_paths:
                # =============================================
                # CAS 1 : CHEMINS IDENTIQUES (doublon exact)
                # =============================================
                # Ex: /article[1]/bdy[1]/sec[1] vs /article[1]/bdy[1]/sec[1]
                # → CONFLIT : même élément
                if xml_path == taken:
                    conflict = True
                    break
                
                # =============================================
                # CAS 2 : RELATION PARENT-ENFANT
                # =============================================
                # Condition: xml_path.startswith(taken + '/')
                # → xml_path est ENFANT de taken
                # Ex: taken = /article[1]/bdy[1]/sec[1]
                #     xml_path = /article[1]/bdy[1]/sec[1]/p[2]
                # → CONFLIT : parent (sec[1]) et enfant (p[2])
                #
                # Condition: taken.startswith(xml_path + '/')
                # → xml_path est PARENT de taken  
                # Ex: xml_path = /article[1]/bdy[1]/sec[1]
                #     taken = /article[1]/bdy[1]/sec[1]/p[2]
                # → CONFLIT : parent (sec[1]) et enfant (p[2])
                if xml_path.startswith(taken + '/') or taken.startswith(xml_path + '/'):
                    # Parent-enfant détecté → CONFLIT selon INEX
                    conflict = True
                    
                    # OPTION : Remplacer par le meilleur score
                    # Chercher l'élément existant dans filtered
                    existing_idx = next((i for i, e in enumerate(filtered) 
                                    if e['xml_path'] == taken and e['article_id'] == article_id), -1)
                    
                    if existing_idx >= 0 and score > filtered[existing_idx]['score']:
                        # Remplacer l'ancien élément par le nouveau (meilleur score)
                        filtered[existing_idx] = elem
                        # Mettre à jour le set des chemins pris
                        taken_paths.remove(taken)
                        taken_paths.add(xml_path)
                    
                    break  # Conflit traité, passer à l'élément suivant
            
            # =============================================
            # CAS 3 : PAS DE CONFLIT - Exemples acceptés
            # =============================================
            # Exemple 1 : FRÈRES (même parent, différents indices)
            #   taken = /article[1]/bdy[1]/sec[1]
            #   xml_path = /article[1]/bdy[1]/sec[2]
            #   → ACCEPTÉ : sec[1] et sec[2] sont frères
            #
            # Exemple 2 : COUSINS (parents différents)
            #   taken = /article[1]/bdy[1]/sec[1]/p[1]
            #   xml_path = /article[1]/bdy[1]/sec[2]/p[1]
            #   → ACCEPTÉ : p[1] de sec[1] et p[1] de sec[2]
            #
            # Exemple 3 : NIVEAUX DIFFÉRENTS MAIS PAS PARENT-ENFANT DIRECT
            #   taken = /article[1]/bdy[1]/sec[1]
            #   xml_path = /article[1]/bdy[1]/sec[2]/p[3]
            #   → ACCEPTÉ : pas de relation parent-enfant
            
            if not conflict:
                # Aucun conflit → ajouter l'élément aux résultats
                filtered.append(elem)
                # Enregistrer ce chemin comme pris pour cet article
                taken_paths_by_article[article_id].add(xml_path)
        
        return filtered

    def _generate_filename(self, team_name: str, run_id: str,
                      fetch_config: Dict, browse_config: Dict,
                      run_params: Dict) -> str:
        """Génère un nom de fichier descriptif"""
        os.makedirs("data/runs", exist_ok=True)
        
        # POUR L'EXERCICE 3 SPÉCIFIQUE
        if "testXML" in run_id:
            # Format: VictorAlbertJules_12_testXML_lm_element-bdy-sec-p_nostop_nostem.txt
            parts = [
                team_name,
                "12",  # Groupe 12
                "testXML",
                "lm",  # SMART lm
                "element-bdy-sec-p",
                "nostop",
                "nostem"
            ]
            filename = "_".join(parts) + ".txt"
        else:
            # Pour les autres runs
            parts = [
                team_name,
                run_id,
                f"fetch-{fetch_config['stemmer']}-{fetch_config['stop_words']}",
                f"browse-{browse_config.get('target_tags', ['sec','p'])}",
                run_params['weighting_scheme']
            ]
            
            if run_params['weighting_scheme'] == 'bm25':
                parts.append(f"k{run_params.get('k1', 1.2)}")
                parts.append(f"b{run_params.get('b', 0.75)}")
            
            filename = "_".join(parts) + ".txt"
        
        return os.path.join("data/runs", filename)

