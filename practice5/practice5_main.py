import os
import time
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from collections import Counter, defaultdict
import re


def create_index_with_config(data_file_path, tokenization="basic", stemmer="nostem", stop_words="nostop"):
    """Crée un index avec une configuration spécifique et retourne un dictionnaire complet"""
    print(f"\nCréation de l'index avec configuration: tokenization={tokenization}, stemmer={stemmer}, stop_words={stop_words}")
    print("=" * 60)

    index = WeightedInvertedIndex()
    index.configure_tokenization(tokenization)
    index.configure_stemmer(stemmer)
    index.configure_stop_words(stop_words)
    
    # Mesure du temps d'indexation
    start_time = time.time()
    #indexing_time = index.build_index(data_file_path, False)
    indexing_time = index.build_index_from_xml_collection(data_file_path)
    end_time = time.time()
    
    # Si build_index ne retourne pas le temps, on le calcule
    if indexing_time is None:
        indexing_time = end_time - start_time
    
    # Calcul des statistiques de base
    stats = index.get_collection_statistics(indexing_time)
    
    return {
        'index': index,
        'indexing_time': indexing_time,
        'stats': stats,
        'config': {
            'tokenization': tokenization,
            'stemmer': stemmer,
            'stop_words': stop_words
        }
    }

def create_element_index_with_config(xml_dir, target_tags=['bdy', 'sec', 'p'], 
                                     tokenization="basic", stemmer="nostem", stop_words="nostop"):
    """Crée un index d'éléments XML avec configuration"""
    print(f"\nCréation d'index éléments avec: tags={target_tags}, "
          f"tokenization={tokenization}, stemmer={stemmer}, stop_words={stop_words}")
    print("=" * 70)

    index = WeightedInvertedIndex()
    index.configure_tokenization(tokenization)
    index.configure_stemmer(stemmer)
    index.configure_stop_words(stop_words)
    
    indexing_time = index.build_index_from_xml_elements(xml_dir, target_tags)
    stats = index.get_collection_statistics(indexing_time)
        
    return {
        'index': index,
        'indexing_time': indexing_time,
        'stats': stats,
        'config': {
            'target_tags': target_tags,
            'tokenization': tokenization,
            'stemmer': stemmer,
            'stop_words': stop_words
        }
    }

def compute_statistics(exercise_num, index_data, weighting_scheme="ltn", k1=1.2, b=0.75, target_doc_id="23724", 
                       target_term="ranking", test_query="web ranking scoring algorithm"):
    """Fonction qui utilise un index pré-construit"""
    
    # Extraction des données de configuration
    index = index_data['index']
    indexing_time = index_data['indexing_time']
    stats = index_data['stats']
    config = index_data['config']
    
    # Construction de la description
    config_desc = f"{weighting_scheme.upper()}"
    if config['stop_words'] != "nostop":
        config_desc += f" + stop-words({config['stop_words']})"
    if config['stemmer'] != "nostem":
        config_desc += f" + stemming({config['stemmer']})"
    if config['tokenization'] != "basic":
        config_desc += f" + tokenization({config['tokenization']})"
    
    if weighting_scheme == "bm25":
        config_desc += f" - k1={k1}, b={b}"

    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {config_desc}")
    print("=" * 60)

    start_total_time = time.time()
    
    # Création du ranker
    ranker = RankedRetrieval(index)

    # Nettoyer le cache précédent si nécessaire
    #ranker.clear_cosine_norms_cache()
    if weighting_scheme == "ltc":
        ranker.clear_cosine_norms_cache()
    
    # Calcul du poids du terme cible dans le document cible
    processed_terms = ranker.process_query_terms(target_term)
    target_weight = ranker.get_term_weight(processed_terms[0], target_doc_id, weighting_scheme, k1=k1, b=b) if processed_terms else 0.0
    
    # Calcul du RSV pour la requête test
    query_terms = ranker.process_query_terms(test_query)
    doc_score = sum(ranker.get_term_weight(t, target_doc_id, weighting_scheme, k1=k1, b=b) for t in query_terms)
    
    # Recherche du top-10
    top_docs = ranker.search_query(test_query, weighting_scheme, top_k=10, k1=k1, b=b)
    
    # Temps total (indexation + pondération)
    weighting_time = time.time() - start_total_time
    total_time = indexing_time + weighting_time

    # Affichage des statistiques complètes
    print(f"\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Configuration: {config_desc}")
    print(f"- Temps total d'indexation + pondération: {total_time:.2f} secondes")
    print(f" * Temps d'indexation seul: {indexing_time:.2f} secondes")
    print(f" * Temps de pondération: {weighting_time:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats['avg_term_length']:.2f} caractères")
    print(f"- Poids du terme '{target_term}' dans le document #{target_doc_id}: {target_weight:.6f}")
    print(f"- RSV du document #{target_doc_id} pour '{test_query}': {doc_score:.6f}")
    
    print(f"- TOP-10 DOCUMENTS pour '{test_query}':")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
    
    return ranker

def generate_inex_run(run_id, ranker, queries, weighting_scheme, type_test=None, granularity="article", stemmer="nostem", 
                      stop_words="nostop", tokenization="basic", k1=1.2, b=0.75, print_top10 = False):
    """Génère un run INEX RIC pour les 7 requêtes"""
    
    # Génération du nom de fichier selon le template
    team_name="AlphaAnaClement"
    filename = f"{team_name}_{run_id}_{type_test}_{weighting_scheme}_{granularity}_{stop_words}_{stemmer}"
    if tokenization != "basic":
        filename += f"_{tokenization}"
    if weighting_scheme == "bm25":
        filename += f"_k1_{k1}_b_{b}"
    filename += ".txt"
    
    print(f"\nGénération du run INEX: {filename}")
    print("-" * 60)
        
    # Génération du fichier run
    with open(f"data/runs/{filename}", "w", encoding="utf-8") as f:
        for query_id, query_text in queries.items():
            # Recherche des 1500 meilleurs documents pour chaque requête
            top_docs = ranker.search_query(query_text, weighting_scheme, top_k=1500, k1=k1, b=b)
            
            # Afficher le top10 de docs de la requète
            if print_top10:
                print(f"  - TOP-10 DOCUMENTS : ")
                for i, (doc_id, score) in enumerate(top_docs[:10], 1):
                    print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
            
            # Écriture des résultats au format INEX RIC
            for i, (doc_id, score) in enumerate(top_docs, 1):
                f.write(f"{query_id} Q0 {doc_id} {i} {score:.6f} {team_name} /article[1]\n")
    
    print(f"Run sauvegardé: {filename} ({len(queries)} requêtes, 1500 documents par requête)")
    return filename

def generate_inex_run_with_metadata(run_id, ranker, queries, config_info, print_top10=False):
    """Génère un run INEX RIC en utilisant les métadonnées"""
    
    team_name = "AlphaAnaClement"
    index = ranker.index
    
    # Construire le nom de fichier selon la configuration
    filename_parts = [
        team_name,
        str(run_id),
        config_info.get('type_test', 'test'),
        config_info.get('weighting_scheme', 'ltn'),
        config_info.get('granularity', 'article'),
        config_info.get('stop_words', 'nostop'),
        config_info.get('stemmer', 'nostem')
    ]
    
    if config_info.get('tokenization', 'basic') != 'basic':
        filename_parts.append(config_info['tokenization'])
    
    if config_info.get('weighting_scheme') == 'bm25':
        filename_parts.extend([
            f"k1_{config_info.get('k1', 1.2)}",
            f"b_{config_info.get('b', 0.75)}"
        ])
    
    filename = '_'.join(filename_parts) + ".txt"
    
    print(f"\nGénération du run INEX: {filename}")
    print("-" * 60)
    
    # Déterminer le format selon la granularité
    is_element_index = hasattr(index, 'doc_type') and index.doc_type == "element"
    with open(f"data/runs/{filename}", "w", encoding="utf-8") as f:
        for query_id, query_text in queries.items():
            results = defaultdict(list)

            # Recherche
            top_docs = ranker.search_query(
                query_text, 
                weighting_scheme=config_info.get('weighting_scheme', 'ltn'),
                top_k=1500,
                k1=config_info.get('k1', 1.2),
                b=config_info.get('b', 0.75)
            )
            
            if print_top10 and query_id == 2009011:  # Afficher seulement pour une requête
                print(f"  - TOP-10 DOCUMENTS pour requête {query_id}:")
                for i, (doc_id, score) in enumerate(top_docs[:10], 1):
                    print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
            
            for rank, (doc_id, score) in enumerate(top_docs, 1):
                if is_element_index:
                    # Pour les éléments, utiliser les métadonnées
                    metadata = index.get_metadata(doc_id)
                    parent_doc_id = index.get_parent_article_id(doc_id)
                    xml_path = index.get_xml_path(doc_id)

                    results[parent_doc_id].append({
                        'rank': rank,
                        'score': score,
                        'xml_path': xml_path
                    })
                else:
                    # Pour les articles, format simple
                    parent_doc_id = doc_id
                    xml_path = "/article[1]"

                    # Format INEX RIC
                    f.write(f"{query_id} Q0 {parent_doc_id} {rank} {score:.6f} {team_name} {xml_path}\n")

            #Orddonée en format inex pour les elements
            if is_element_index:
                doc_processed = []
                for doc_id, result in results.items():
                    result_sorted = sorted(result, key=lambda x: -x['score'])

                    for element in result:
                        # Format INEX RIC
                        f.write(f"{query_id} Q0 {doc_id} {element['rank']} {element['score']:.6f} {team_name} {element['xml_path']}\n")

    print(f"Run sauvegardé: {filename}")
    return filename

def exercice_3(queries, run_id):
    """Exercice 3: Indexation éléments XML avec SMART lm"""
    print("\n" + "="*60)
    print("EXERCICE 3: Indexation éléments XML (SMART lm)")
    print("="*60)
    
    xml_dir = "data/Practice_05_data/XML-Coll-withSem"
    
    # Configuration spécifique
    index_data = create_element_index_with_config(
        xml_dir, 
        target_tags=['bdy', 'sec', 'p'],
        tokenization="basic",
        stemmer="nostem",
        stop_words="nostop"
    )
    
    # Créer le ranker
    ranker = RankedRetrieval(index_data['index'])
    
    # Configuration pour le run
    config_info = {
        'type_test': 'testXML',
        'weighting_scheme': 'lnt',
        'granularity': 'element-bdy-sec-p',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'tokenization': 'basic'
    }
    
    # Générer le run avec métadonnées
    filename = generate_inex_run_with_metadata(run_id, ranker, queries, config_info, print_top10=True)
    run_id +=1
    print(f"\nExercice 3 terminé: {filename}")
    return run_id, index_data, ranker

def main():
    #data_file_path = "data/Text_Only_Ascii_Coll_NoSem"
    data_file_path = "data/Practice_05_data/XML-Coll-withSem"
    runs_dir = "data/runs"
    
    # Requêtes 
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    # Construction des index avec toutes les configurations
    index_no_stop_no_stem = create_index_with_config(data_file_path, "basic", "nostem", "nostop")
    index_stop_no_stem = create_index_with_config(data_file_path, "basic", "nostem", "stop671")
    index_no_stop_stem = create_index_with_config(data_file_path, "basic", "porter", "nostop")
    index_stop_stem = create_index_with_config(data_file_path, "basic", "porter", "stop671")
    
    # Créer le répertoire runs s'il n'existe pas
    os.makedirs(runs_dir, exist_ok=True)

    # CALCULER le run_id de départ UNE SEULE FOIS
    #base_run_id = len([f for f in os.listdir(runs_dir) if os.path.isfile(os.path.join(runs_dir, f))])
    #current_run_id = base_run_id
    current_run_id = 1 # on commence à 1 pour l'exercice 1
    
    print(f"Run ID de départ: {current_run_id}")
    
    # --- Exercise 1: Indexing XML documents (SMART ltn) ---    
    # Calcul des statistiques pour LTN
    ranker_ltn = compute_statistics(exercise_num=1, index_data=index_no_stop_no_stem, weighting_scheme="ltn")
    # Génération du run INEX pour LTN
    generate_inex_run(current_run_id, ranker_ltn, queries, "ltn", "article", "nostem", "nostop")
    current_run_id += 1
    
    
    # --- Exercise 2: test runs avec variantes d'index (12 combinaisons) ---
    weighting_schemes = ["ltn", "ltc", "bm25"]
    indexers = [index_no_stop_no_stem, index_stop_no_stem, index_no_stop_stem, index_stop_stem]

    # Structure: pour chaque type d'indexation, pour chaque algorithme, pour chaque requête
    for index_data in indexers:
        for weighting in weighting_schemes:
            
            # Extraire la configuration de l'index
            config = index_data['config']
            stemmer_name = config['stemmer']
            stop_name = config['stop_words']
            
            # Calcul des statistiques pour cette configuration (k1 et b par défaut, si BM25)
            ranker_result = compute_statistics(4, index_data, weighting)   
            
            # Génération du run pour les 7 requêtes 
            generate_inex_run(current_run_id, ranker_result, queries, weighting, "test2", "article", stemmer_name, stop_name)
            
            current_run_id += 1
    
    # --- Exercise 3: Indexing XML elements (SMART ltn) ---
    current_run_id, _, _ = exercice_3(queries, current_run_id)


if __name__ == "__main__":
     # Nettoyer le dossier runs au début
    if os.path.exists("data/runs"):
        for file in os.listdir("data/runs"):
            if file.endswith(".txt"):
                os.remove(os.path.join("data/runs", file))
        print("Dossier 'runs' nettoyé")

    main()