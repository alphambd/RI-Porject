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

def generate_inex_run(run_id, ranker, queries, weighting_scheme, test_type=None, granularity="article", stemmer="nostem", 
                      stop_words="nostop", tokenization="basic", k1=1.2, b=0.75, print_top10 = False):
    """Génère un run INEX RIC pour les 7 requêtes"""
    
    # Génération du nom de fichier selon le template
    team_name="AlphaAnaClement"

    if test_type is None:
        filename = f"{team_name}_{run_id}_{weighting_scheme}_{granularity}_{stop_words}_{stemmer}"
    else:
        filename = f"{team_name}_{run_id}_{test_type}_{weighting_scheme}_{granularity}_{stop_words}_{stemmer}"
    
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
                        f.write(f"{query_id} Q0 {doc_id} {element['rank']} {element['score']:.6f} {team_name} {xml_path}\n")

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

def exercice_4(queries, current_run_id):
    """Exercice 4: Création de runs d'éléments XML avec différentes configurations"""
    print("\n" + "=" * 60)
    print("EXERCICE 4: XML elements runs")
    print("=" * 60)

    xml_dir = "data/Practice_05_data/XML-Coll-withSem"

    configurations = [
        {
            'target_tags': ['bdy', 'sec', 'p'],
            'tokenization': 'basic',
            'stemmer': 'nostem',
            'stop_words': 'nostop',
            'weighting_scheme': 'ltn'
        },
        {
            'target_tags': ['bdy', 'sec', 'p'],
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'nostop',
            'weighting_scheme': 'ltc'
        },
        {
            'target_tags': ['bdy', 'sec'],
            'tokenization': 'basic',
            'stemmer': 'nostem',
            'stop_words': 'stop671',
            'weighting_scheme': 'bm25',
            'k1': 1.2,
            'b': 0.75
        },
        {
            'target_tags': ['bdy', 'sec', 'p'],
            'tokenization': 'extended',
            'stemmer': 'porter',
            'stop_words': 'stop671',
            'weighting_scheme': 'bm25',
            'k1': 1.5,
            'b': 0.8
        }
    ]

    for i, config in enumerate(configurations):
        print(f"\nConfiguration {i + 1}/{len(configurations)}:")
        print(f"  Tags: {config['target_tags']}")
        print(f"  Pondération: {config['weighting_scheme']}")
        print(f"  Stemmer: {config['stemmer']}")
        print(f"  Stop-words: {config['stop_words']}")

        index_data = create_element_index_with_config(
            xml_dir,
            target_tags=config['target_tags'],
            tokenization=config['tokenization'],
            stemmer=config['stemmer'],
            stop_words=config['stop_words']
        )

        ranker = RankedRetrieval(index_data['index'])

        granularity_str = '-'.join(config['target_tags'])
        config_info = {
            'type_test': 'element_run',
            'weighting_scheme': config['weighting_scheme'],
            'granularity': f'element-{granularity_str}',
            'stemmer': config['stemmer'],
            'stop_words': config['stop_words'],
            'tokenization': config['tokenization']
        }

        if 'k1' in config:
            config_info['k1'] = config['k1']
            config_info['b'] = config['b']

        filename = generate_inex_run_with_metadata(current_run_id, ranker, queries, config_info, print_top10=True)
        current_run_id += 1
    return current_run_id

def exercice_5(queries, current_run_id, xml_dir, team_name="AlphaAnaClement", k1=1.2, b=0.75):
    """
    Exercice 5: BM25Fw (late combination, Wilkinson94)
    - BM25 calculé séparément pour chaque champ
    - Scores combinés avec pondération αi
    """

    print("\n" + "=" * 60)
    print("EXERCICE 5: Fields weighting – BM25Fw (late combination)")
    print("=" * 60)

    # --- Définition des champs et pondérations αi ---
    fields = ["title", "abstract", "body"]
    field_weights = {"title": 2.0, "abstract": 1.5, "body": 1.0}

    # --- Étape 1: Construction d’un index pour tous les champs ---
    index = WeightedInvertedIndex()
    index.configure_tokenization("basic")
    index.configure_stemmer("porter")
    index.configure_stop_words("stop671")
    index.build_index_from_xml_collection(xml_dir)

    # --- Initialisation du moteur de recherche BM25 ---
    ranker = RankedRetrieval(index)

    # --- Préparation du fichier de sortie ---
    filename = f"{team_name}_{current_run_id}_BM25Fw_article_fields.txt"
    os.makedirs("data/runs", exist_ok=True)

    with open(f"data/runs/{filename}", "w", encoding="utf-8") as f:
        for qid, query_text in queries.items():
            article_scores = defaultdict(float)  # combinaison des scores par article

            # Calcul BM25 pour chaque champ séparément
            for field in fields:
                ana = field_weights[field]

                # Ici, on  un BM25 sur le champ : on filtre l'index par champ si possible
                field_results = ranker.search_query(
                    query_text,
                    weighting_scheme="bm25",
                    top_k=1500,
                    k1=k1,
                    b=b,
                    #field=field  # <-- ajoute cette option si ton moteur supporte les champs
                )

                # Combinaison tardive avec pondération αi
                for doc_id, score in field_results:
                    article_scores[doc_id] += ana * score

            # Affichage du nombre de documents uniques
            print(f" * Recherche: '{query_text}' -> termes: {ranker.index.process_tokens(ranker.index.apply_tokenization(query_text))}")
            print(f"   - Documents pertinents potentiels: {len(article_scores)}")

            # Tri et sélection des 1500 meilleurs articles
            ranked_docs = sorted(article_scores.items(), key=lambda x: -x[1])[:1500]

            # Écriture du run au format INEX/TREC
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")

    print(f"Run BM25Fw généré : {filename}")
    return current_run_id + 1



def exercice_6(queries, current_run_id, xml_dir, team_name="AlphaAnaClement", k1=1.2, b=0.75):
    """
    Exercice 6: BM25FR (early combination, Robertson2004)
    - Step 1: combinaison précoce des TF pondérés par champ
    - Step 2: BM25 sur article combiné
    """

    print("\n" + "=" * 60)
    print("EXERCICE 6: Fields weighting – BM25FR (early combination)")
    print("=" * 60)

    # --- Définition des champs et leurs pondérations αi ---
    fields = ["title", "abstract", "body"]
    field_weights = {"title": 2.0, "abstract": 1.5, "body": 1.0}

    # --- Étape 1: Construction de l'index combiné ---
    # Chaque terme t d’un article est pondéré par ses champs:
    # tf'_t,article = Σ αi * tf_t,field_i
    combined_tf_index = WeightedInvertedIndex()
    combined_tf_index.configure_tokenization("basic")  # tokenisation simple
    combined_tf_index.configure_stemmer("porter")      # porter stemmer
    combined_tf_index.configure_stop_words("stop671")  # stopwords spécifique
    combined_tf_index.build_index_from_weighted_fields(xml_dir, fields, field_weights)
    # -> l'index résultant contient les termes pondérés pour chaque article

    # --- Étape 2: Initialisation du moteur de recherche BM25 ---
    ranker = RankedRetrieval(combined_tf_index)

    # --- Préparation du fichier de sortie pour le run INEX ---
    filename = f"{team_name}_{current_run_id}_BM25FR_article_fields.txt"
    os.makedirs("data/runs", exist_ok=True)

    # --- Calcul des scores BM25 pour chaque requête ---
    with open(f"data/runs/{filename}", "w", encoding="utf-8") as f:
        for qid, query_text in queries.items():
            # BM25 appliqué sur le document combiné
            scores = ranker.search_query(
                query_text,
                weighting_scheme="bm25",
                top_k=1500,  # récupérer les 1500 meilleurs documents
                k1=k1,
                b=b
            )
            # Écriture du run au format TREC/INEX
            for rank, (doc_id, score) in enumerate(scores, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {team_name} /article[1]\n")

    print(f"Run BM25FR généré : {filename}")

    # Retourner le run_id incrémenté pour la suite
    return current_run_id + 1



def main():
    current_run_id = 1
    data_file_path = "data/Practice_05_data/XML-Coll-withSem"
    runs_dir = "data/runs"

    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }

    current_run_id = 1
    print(f"Run ID de départ: {current_run_id}")

    # --- Exercice 1 ---
    index_no_stop_no_stem = create_index_with_config(data_file_path, "basic", "nostem", "nostop")
    ranker_ltn = compute_statistics(exercise_num=1, index_data=index_no_stop_no_stem, weighting_scheme="ltn")
    generate_inex_run(current_run_id, ranker_ltn, queries, "ltn", None, "article", "nostem", "nostop")
    current_run_id += 1

    # --- Exercice 2 ---
    weighting_schemes = ["ltn", "ltc", "bm25"]
    index_stop_no_stem = create_index_with_config(data_file_path, "basic", "nostem", "stop671")
    index_no_stop_stem = create_index_with_config(data_file_path, "basic", "porter", "nostop")
    index_stop_stem = create_index_with_config(data_file_path, "basic", "porter", "stop671")
    indexers = [index_no_stop_no_stem, index_stop_no_stem, index_no_stop_stem, index_stop_stem]

    for index_data in indexers:
        for weighting in weighting_schemes:
            config = index_data['config']
            stemmer_name = config['stemmer']
            stop_name = config['stop_words']
            ranker_result = compute_statistics(4, index_data, weighting)
            generate_inex_run(current_run_id, ranker_result, queries, weighting, "test2", "article", stemmer_name, stop_name)
            current_run_id += 1

    # --- Exercice 3 ---
    current_run_id, _, _ = exercice_3(queries, current_run_id)


    # --- Exercices 4 ---
    current_run_id = exercice_4(queries, current_run_id)

    # --- Exercice 5 ---
    xml_dir = "data/Practice_05_data/XML-Coll-withSem"
    current_run_id = exercice_5(queries, current_run_id, xml_dir)

    # --- Exercice 6 ---
    xml_dir = "data/Practice_05_data/XML-Coll-withSem"
    current_run_id = exercice_6(queries, current_run_id, xml_dir)


if __name__ == "__main__":
    # Nettoyage du dossier runs avant exécution
    runs_dir = "data/runs"
    if os.path.exists(runs_dir):
        for file in os.listdir(runs_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(runs_dir, file))
        print("Dossier 'runs' nettoyé")

    main()


