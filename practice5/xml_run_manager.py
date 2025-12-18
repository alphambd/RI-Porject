import os
import time
import pickle
from advanced_indexer import WeightedInvertedIndex
from collections import defaultdict


CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================
# FONCTIONS de configuration avec CACHE
# ============================================

def get_cache_filename(config_type, params):
    """Génère un nom de fichier de cache basé sur la configuration"""
    config_hash = hash(frozenset(params.items()))
    return os.path.join(CACHE_DIR, f"{config_type}_{abs(config_hash)}.pkl")

def create_index_with_cache(data_file_path, config_params):
    """
    Crée un index avec gestion de cache pour éviter les recalculs
    
    Args:
        data_file_path: Chemin vers les fichiers XML
        config_params: Dictionnaire de configuration (tokenization, stemmer, stop_words)
    
    Returns:
        Dictionnaire contenant l'index et les statistiques
    """
    print(f"\nCréation d'index avec configuration: {config_params}")
    print("=" * 60)
    
    # Vérifier si l'index existe en cache
    cache_file = get_cache_filename("article_index", config_params)
    if os.path.exists(cache_file):
        print(f"Chargement de l'index depuis le cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print("SUCCES - Index chargé depuis le cache!")
            return cached_data
        except Exception as e:
            print(f"ERREUR de chargement du cache: {e}")
            print("Recalcul de l'index...")
    
    # Créer un nouvel index
    index = WeightedInvertedIndex()
    index.configure_tokenization(config_params['tokenization'])
    index.configure_stemmer(config_params['stemmer'])
    index.configure_stop_words(config_params['stop_words'])
    
    # Mesure du temps d'indexation
    start_time = time.time()
    indexing_time = index.build_index_from_xml_collection(data_file_path)
    end_time = time.time()
    
    if indexing_time is None:
        indexing_time = end_time - start_time
    
    # Obtenir les statistiques
    stats = index.get_collection_statistics(indexing_time)
    
    # Préparer les données pour le cache
    index_data = {
        'index': index,
        'indexing_time': indexing_time,
        'stats': stats,
        'config': config_params
    }
    
    # Sauvegarder dans le cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"SUCCES - Index sauvegardé dans le cache: {cache_file}")
    except Exception as e:
        print(f"ERREUR de sauvegarde du cache: {e}")
    
    return index_data

def create_element_index_with_cache(xml_dir, target_tags, config_params):
    """
    Crée un index d'éléments XML avec gestion de cache
    
    Args:
        xml_dir: Répertoire des fichiers XML
        target_tags: Liste des tags XML à indexer (ex: ['bdy', 'sec', 'p'])
        config_params: Dictionnaire de configuration
    
    Returns:
        Dictionnaire contenant l'index et les statistiques
    """
    print(f"\nCréation d'index éléments: tags={target_tags}, config={config_params}")
    print("=" * 70)
    
    # Vérifier si l'index existe en cache
    cache_params = config_params.copy()
    cache_params['target_tags'] = tuple(target_tags)  # Convertir en tuple pour le hash
    cache_file = get_cache_filename("element_index", cache_params)
    
    if os.path.exists(cache_file):
        print(f"Chargement de l'index éléments depuis le cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print("SUCCES - Index éléments chargé depuis le cache!")
            return cached_data
        except Exception as e:
            print(f"ERREUR de chargement du cache: {e}")
            print("Recalcul de l'index éléments...")
    
    # Créer un nouvel index d'éléments
    index = WeightedInvertedIndex()
    index.configure_tokenization(config_params['tokenization'])
    index.configure_stemmer(config_params['stemmer'])
    index.configure_stop_words(config_params['stop_words'])
    
    start_time = time.time()
    indexing_time = index.build_index_from_xml_elements(xml_dir, target_tags)
    end_time = time.time()
    
    if indexing_time is None:
        indexing_time = end_time - start_time
    
    stats = index.get_collection_statistics(indexing_time)
    
    # Préparer les données pour le cache
    element_data = {
        'index': index,
        'indexing_time': indexing_time,
        'stats': stats,
        'config': {**config_params, 'target_tags': target_tags}
    }
    
    # Sauvegarder dans le cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(element_data, f)
        print(f"SUCCES - Index éléments sauvegardé dans le cache: {cache_file}")
    except Exception as e:
        print(f"ERREUR de sauvegarde du cache: {e}")
    
    return element_data

# ============================================
# FONCTIONS utilitaires pour FETCH and BROWSE
# ============================================

def get_elements_for_article(article_id, query_terms, element_ranker, score_threshold=0.0,
                            weighting_scheme="ltn", k1=1.2, b=0.75):
    """
    Récupère les éléments d'un article avec leur score
    
    Args:
        article_id: ID de l'article
        query_terms: Termes de la requête
        element_ranker: Ranker pour les éléments
        score_threshold: Seuil minimal de score
        weighting_scheme: Schéma de pondération
        k1: Paramètre k1 pour BM25
        b: Paramètre b pour BM25
    
    Returns:
        Liste d'éléments avec leurs métadonnées et scores
    """
    article_elements = []
    
    for element_id in element_ranker.index.doc_ids:
        metadata = element_ranker.index.get_metadata(element_id)
        
        # Vérifier si l'élément appartient à cet article
        if str(metadata.get('parent_doc_id', '')) == str(article_id):
            # Calculer le score avec le schéma de pondération spécifié
            score = 0.0
            for term in query_terms:
                weight = element_ranker.get_term_weight_cached(
                    term, element_id, weighting_scheme, k1, b
                )
                if weight is not None:
                    score += weight
            
            # Filtrer par seuil de score
            if score >= score_threshold:
                article_elements.append({
                    'element_id': element_id,
                    'score': score,
                    'article_id': article_id,
                    'xml_path': metadata.get('xml_path', '/article[1]'),
                    'tag': metadata.get('tag', 'element')
                })
    
    return article_elements

def select_top_elements_without_overlap(elements, max_elements=1500):
    """
    Sélectionne les meilleurs éléments sans overlap
    
    Args:
        elements: Liste d'éléments triés par score décroissant
        max_elements: Nombre maximum d'éléments à sélectionner
    
    Returns:
        Liste filtrée sans overlaps
    """
    filtered_elements = []
    taken_paths = set()
    
    for elem in elements:
        if len(filtered_elements) >= max_elements:
            break
        
        xml_path = elem['xml_path']
        conflict = False
        
        # Vérifier les conflits avec les chemins déjà pris
        for taken in taken_paths:
            if (xml_path.startswith(taken + '/') or 
                taken.startswith(xml_path + '/')):
                conflict = True
                break
        
        if not conflict:
            filtered_elements.append(elem)
            taken_paths.add(xml_path)
    
    return filtered_elements

def group_elements_by_article(elements):
    """
    Groupe les éléments par article et les trie
    
    Args:
        elements: Liste d'éléments
        
    Returns:
        Dictionnaire {article_id: [éléments triés]}
    """
    
    elements_by_article = defaultdict(list)
    for elem in elements:
        elements_by_article[elem['article_id']].append(elem)
    
    # Trier les éléments dans chaque article par score
    for article_id in elements_by_article:
        elements_by_article[article_id].sort(key=lambda x: -x['score'])
    
    return elements_by_article

def write_results_to_file(f, query_id, elements_by_article, team_name, start_rank=1):
    """
    Écrit les résultats dans le fichier format INEX
    
    Args:
        f: File object
        query_id: ID de la requête
        elements_by_article: Éléments groupés par article
        team_name: Nom de l'équipe
        start_rank: Rang de départ
    
    Returns:
        Prochain rang disponible
    """
    # Trier les articles par score du meilleur élément
    sorted_articles = sorted(
        elements_by_article.items(),
        key=lambda x: max(e['score'] for e in x[1]) if x[1] else 0,
        reverse=True
    )
    
    current_rank = start_rank
    for article_id, article_elements in sorted_articles:
        for elem in article_elements:
            f.write(f"{query_id} Q0 {article_id} {current_rank} {elem['score']:.6f} {team_name} {elem['xml_path']}\n")
            current_rank += 1
    
    return current_rank

# ============================================
# APPROCHE : Fetch and Browse avec Pooling
# ============================================

def generate_fetch_browse_pooling(run_id, article_ranker, element_ranker, queries,
                                           top_articles=500, score_threshold=0.01,
                                           progress_interval=50):
    """
    Fetch and Browse avec pooling intelligent - Version optimisée
    
    Args:
        run_id: Identifiant du run
        article_ranker: Ranker pour les articles
        element_ranker: Ranker pour les éléments
        queries: Dictionnaire des requêtes
        top_articles: Nombre d'articles à considérer
        score_threshold: Seuil minimal de score (optimisation importante)
        progress_interval: Intervalle d'affichage de progression
    
    Returns:
        Nom du fichier généré
    """
    team_name = "AlphaAnaClement"
    filename = f"{team_name}_{run_id}_testXML_fetch-browse-pooling_opt_nostop_nostem.txt"
    runs_dir = "data/runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    print(f"\nGENERATION DU RUN - Fetch and Browse avec Pooling Optimisé")
    print(f"   Fichier: {filename}")
    print(f"   Paramètres: top_articles={top_articles}, score_threshold={score_threshold}")
    print("-" * 70)
    
    total_start_time = time.time()
    
    with open(os.path.join(runs_dir, filename), "w", encoding="utf-8") as f:
        for query_id, query_text in queries.items():
            query_start_time = time.time()
            print(f"  REQUETE {query_id}: '{query_text}'")
            
            # 1. Préparer les termes de la requête (une seule fois)
            query_terms = element_ranker.process_query_terms(query_text)
            print(f"    Termes de la requête: {len(query_terms)}")
            
            # 2. FETCH: Recherche des articles pertinents
            fetch_start = time.time()
            top_articles_list = article_ranker.search_query(
                query_text, 
                weighting_scheme="ltn",
                top_k=top_articles
            )
            fetch_time = time.time() - fetch_start
            print(f"    + PHASE FETCH: {len(top_articles_list)} articles trouvés en {fetch_time:.2f}s")
            
            # 3. BROWSE: Collecte des éléments pertinents avec seuil
            browse_start = time.time()
            global_pool = []
            elements_collected = 0
            articles_with_elements = 0
            
            for article_idx, (article_id, article_score) in enumerate(top_articles_list, 1):
                # Affichage de progression
                if article_idx % progress_interval == 0:
                    print(f"      Article {article_idx}/{len(top_articles_list)} - "
                          f"{len(global_pool)} éléments collectés")
                
                # Récupérer les éléments de cet article (avec seuil)
                article_elements = get_elements_for_article(
                    article_id, 
                    query_terms, 
                    element_ranker,
                    score_threshold=score_threshold
                )
                
                if article_elements:
                    articles_with_elements += 1
                    global_pool.extend(article_elements)
                    elements_collected += len(article_elements)
            
            browse_time = time.time() - browse_start
            print(f"    + PHASE BROWSE: {elements_collected} éléments de {articles_with_elements} articles en {browse_time:.2f}s")
            
            # 4. TRI: Trier le pool global par score
            sort_start = time.time()
            global_pool.sort(key=lambda x: -x['score'])
            sort_time = time.time() - sort_start
            print(f"    + PHASE TRI: {len(global_pool)} éléments triés en {sort_time:.2f}s")
            
            # 5. FILTRAGE: Éliminer les overlaps
            filter_start = time.time()
            filtered_elements = select_top_elements_without_overlap(
                global_pool,
                max_elements=1500
            )
            filter_time = time.time() - filter_start
            print(f"    + PHASE FILTRAGE: {len(filtered_elements)} éléments après anti-overlap en {filter_time:.2f}s")
            
            # Si pas assez d'éléments, baisser le seuil progressivement
            if len(filtered_elements) < 1500 and score_threshold > 0:
                print(f"    ATTENTION - Seulement {len(filtered_elements)} éléments, recherche supplémentaire...")
                
                # Recherche d'éléments supplémentaires avec seuil réduit
                additional_elements = []
                additional_threshold = score_threshold / 2
                
                for article_idx, (article_id, article_score) in enumerate(top_articles_list, 1):
                    # Prendre seulement les articles qui n'ont pas assez d'éléments
                    article_elements = get_elements_for_article(
                        article_id, 
                        query_terms, 
                        element_ranker,
                        score_threshold=additional_threshold
                    )
                    
                    # Filtrer pour ne garder que les éléments non déjà pris
                    taken_paths = {e['xml_path'] for e in filtered_elements}
                    for elem in article_elements:
                        if len(filtered_elements) + len(additional_elements) >= 1500:
                            break
                        
                        conflict = False
                        for taken in taken_paths:
                            if (elem['xml_path'].startswith(taken + '/') or 
                                taken.startswith(elem['xml_path'] + '/')):
                                conflict = True
                                break
                        
                        if not conflict:
                            additional_elements.append(elem)
                            taken_paths.add(elem['xml_path'])
                
                # Ajouter les éléments supplémentaires
                if additional_elements:
                    additional_elements.sort(key=lambda x: -x['score'])
                    filtered_elements.extend(additional_elements[:1500 - len(filtered_elements)])
                    print(f"    SUCCES - {len(additional_elements)} éléments supplémentaires ajoutés")
            
            # 6. GROUPEMENT: Par article pour éviter l'interleaving
            group_start = time.time()
            elements_by_article = group_elements_by_article(filtered_elements[:1500])
            group_time = time.time() - group_start
            
            # 7. ÉCRITURE: Dans le fichier
            write_start = time.time()
            write_results_to_file(f, query_id, elements_by_article, team_name)
            write_time = time.time() - write_start
            
            query_time = time.time() - query_start_time
            
            # Statistiques
            num_articles = len(elements_by_article)
            avg_elements_per_article = len(filtered_elements[:1500]) / num_articles if num_articles > 0 else 0
            
            print(f"    RESULTATS: {len(filtered_elements[:1500])} éléments sur {num_articles} articles")
            print(f"    TEMPS TOTAL REQUETE: {query_time:.2f}s")
            print(f"      - FETCH: {fetch_time:.2f}s")
            print(f"      - BROWSE: {browse_time:.2f}s")
            print(f"      - TRI: {sort_time:.2f}s")
            print(f"      - FILTRAGE: {filter_time:.2f}s")
            print(f"      - GROUPEMENT: {group_time:.2f}s")
            print(f"      - ECRITURE: {write_time:.2f}s")
            print()
    
    total_time = time.time() - total_start_time
    print(f"\nSUCCES - Run sauvegardé: {filename}")
    print(f"TEMPS TOTAL D'EXECUTION: {total_time:.2f} secondes")
    
    return filename

def generate_fetch_browse_pooling_optimized(run_id, article_ranker, element_ranker, queries,
                                           top_articles=500, score_threshold=0.01,
                                           progress_interval=50, weighting_scheme="ltn",
                                           k1=1.2, b=0.75):
    """
    Fetch and Browse avec pooling intelligent - Version optimisée avec choix de pondération
    
    Args:
        run_id: Identifiant du run
        article_ranker: Ranker pour les articles
        element_ranker: Ranker pour les éléments
        queries: Dictionnaire des requêtes
        top_articles: Nombre d'articles à considérer
        score_threshold: Seuil minimal de score
        progress_interval: Intervalle d'affichage de progression
        weighting_scheme: Schéma de pondération (ltn, ltc, bm25)
        k1: Paramètre k1 pour BM25 (si applicable)
        b: Paramètre b pour BM25 (si applicable)
    
    Returns:
        Nom du fichier généré
    """
    team_name = "AlphaAnaClement"
    
    # Construire le nom de fichier avec les paramètres
    weighting_str = weighting_scheme
    if weighting_scheme == "bm25":
        weighting_str = f"bm25_k{k1}_b{b}"
    
    filename = f"{team_name}_{run_id}_testXML_fetch-browse_{weighting_str}.txt"
    runs_dir = "data/runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    print(f"\nGENERATION DU RUN - Fetch and Browse avec {weighting_scheme.upper()}")
    print(f"   Fichier: {filename}")
    print(f"   Paramètres: top_articles={top_articles}, score_threshold={score_threshold}")
    if weighting_scheme == "bm25":
        print(f"   Paramètres BM25: k1={k1}, b={b}")
    print("-" * 70)
    
    total_start_time = time.time()
    
    with open(os.path.join(runs_dir, filename), "w", encoding="utf-8") as f:
        for query_id, query_text in queries.items():
            query_start_time = time.time()
            print(f"  REQUETE {query_id}: '{query_text}'")
            
            # 1. Préparer les termes de la requête
            query_terms = element_ranker.process_query_terms(query_text)
            print(f"    Termes de la requête: {len(query_terms)}")
            
            # 2. FETCH: Recherche des articles pertinents
            fetch_start = time.time()
            top_articles_list = article_ranker.search_query(
                query_text, 
                weighting_scheme=weighting_scheme,
                top_k=top_articles,
                k1=k1,
                b=b
            )
            fetch_time = time.time() - fetch_start
            print(f"    PHASE FETCH: {len(top_articles_list)} articles trouvés en {fetch_time:.2f}s")
            
            # 3. BROWSE: Collecte des éléments pertinents avec seuil
            browse_start = time.time()
            global_pool = []
            elements_collected = 0
            articles_with_elements = 0
            
            for article_idx, (article_id, article_score) in enumerate(top_articles_list, 1):
                # Affichage de progression
                if article_idx % progress_interval == 0:
                    print(f"      Article {article_idx}/{len(top_articles_list)} - "
                          f"{len(global_pool)} éléments collectés")
                
                # Récupérer les éléments de cet article (avec seuil)
                article_elements = get_elements_for_article(
                    article_id, 
                    query_terms, 
                    element_ranker,
                    score_threshold=score_threshold,
                    weighting_scheme=weighting_scheme,
                    k1=k1,
                    b=b
                )
                
                if article_elements:
                    articles_with_elements += 1
                    global_pool.extend(article_elements)
                    elements_collected += len(article_elements)
            
            browse_time = time.time() - browse_start
            print(f"    PHASE BROWSE: {elements_collected} éléments de {articles_with_elements} articles en {browse_time:.2f}s")
            
            # 4. TRI: Trier le pool global par score
            sort_start = time.time()
            global_pool.sort(key=lambda x: -x['score'])
            sort_time = time.time() - sort_start
            print(f"    + PHASE TRI: {len(global_pool)} éléments triés en {sort_time:.2f}s")
            
            # 5. FILTRAGE: Éliminer les overlaps
            filter_start = time.time()
            filtered_elements = select_top_elements_without_overlap(
                global_pool,
                max_elements=1500
            )
            filter_time = time.time() - filter_start
            print(f"    + PHASE FILTRAGE: {len(filtered_elements)} éléments après anti-overlap en {filter_time:.2f}s")
            
            # Si pas assez d'éléments, baisser le seuil progressivement
            if len(filtered_elements) < 1500 and score_threshold > 0:
                print(f"    ATTENTION - Seulement {len(filtered_elements)} éléments, recherche supplémentaire...")
                
                # Recherche d'éléments supplémentaires avec seuil réduit
                additional_elements = []
                additional_threshold = score_threshold / 2
                
                for article_idx, (article_id, article_score) in enumerate(top_articles_list, 1):
                    # Prendre seulement les articles qui n'ont pas assez d'éléments
                    article_elements = get_elements_for_article(
                        article_id, 
                        query_terms, 
                        element_ranker,
                        score_threshold=additional_threshold
                    )
                    
                    # Filtrer pour ne garder que les éléments non déjà pris
                    taken_paths = {e['xml_path'] for e in filtered_elements}
                    for elem in article_elements:
                        if len(filtered_elements) + len(additional_elements) >= 1500:
                            break
                        
                        conflict = False
                        for taken in taken_paths:
                            if (elem['xml_path'].startswith(taken + '/') or 
                                taken.startswith(elem['xml_path'] + '/')):
                                conflict = True
                                break
                        
                        if not conflict:
                            additional_elements.append(elem)
                            taken_paths.add(elem['xml_path'])
                
                # Ajouter les éléments supplémentaires
                if additional_elements:
                    additional_elements.sort(key=lambda x: -x['score'])
                    filtered_elements.extend(additional_elements[:1500 - len(filtered_elements)])
                    print(f"    SUCCES - {len(additional_elements)} éléments supplémentaires ajoutés")
            
            # 6. GROUPEMENT: Par article pour éviter l'interleaving
            group_start = time.time()
            elements_by_article = group_elements_by_article(filtered_elements[:1500])
            group_time = time.time() - group_start
            
            # 7. ÉCRITURE: Dans le fichier
            write_start = time.time()
            write_results_to_file(f, query_id, elements_by_article, team_name)
            write_time = time.time() - write_start
            
            query_time = time.time() - query_start_time
            
            # Statistiques
            num_articles = len(elements_by_article)
            avg_elements_per_article = len(filtered_elements[:1500]) / num_articles if num_articles > 0 else 0
            
            print(f"    RESULTATS: {len(filtered_elements[:1500])} éléments sur {num_articles} articles")
            print(f"    TEMPS TOTAL REQUETE: {query_time:.2f}s")
            print(f"      - FETCH: {fetch_time:.2f}s")
            print(f"      - BROWSE: {browse_time:.2f}s")
            print(f"      - TRI: {sort_time:.2f}s")
            print(f"      - FILTRAGE: {filter_time:.2f}s")
            print(f"      - GROUPEMENT: {group_time:.2f}s")
            print(f"      - ECRITURE: {write_time:.2f}s")
            print()
    
    total_time = time.time() - total_start_time
    print(f"\nSUCCES - Run sauvegardé: {filename}")
    print(f"TEMPS TOTAL D'EXECUTION: {total_time:.2f} secondes")
    
    return filename
