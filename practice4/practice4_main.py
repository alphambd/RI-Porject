import os
import time
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

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
    indexing_time = index.build_index(data_file_path, False)
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

def generate_inex_run(run_id, ranker, queries, weighting_scheme, granularity="article", stemmer="nostem", 
                      stop_words="nostop", tokenization="basic", k1=1.2, b=0.75, print_top10 = False):
    """Génère un run INEX RIC pour les 7 requêtes"""
    
    # Génération du nom de fichier selon le template
    team_name="AlphaAnaClement"
    filename = f"{team_name}_{run_id}_test_{weighting_scheme}_{granularity}_{stop_words}_{stemmer}"
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


def exercise5_tokenization(data_file_path, queries, start_run_id):
    """Teste différentes méthodes de tokenization avec compute_statistics"""
    
    tokenizations = ["extended", "hyphen", "apostrophe"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for tokenization in tokenizations:
        print(f"\n--- Testing tokenization: {tokenization} ---")
        
        # Créer l'index avec la configuration de tokenization
        index_data = create_index_with_config(data_file_path, tokenization, "nostem", "nostop")
        
        for weighting in weightings:
            print(f"  - Avec {weighting.upper()}")
            
            # Calcul des statistiques détaillées avec compute_statistics
            ranker_result = compute_statistics(5, index_data, weighting)
            
            # Génération du run pour toutes les requêtes
            generate_inex_run(current_run_id, ranker_result, queries, weighting, "article", "nostem", "nostop", tokenization)
            
            current_run_id += 1
    
    return current_run_id
"""
def exercise5_complete_analysis(data_file_path, queries, start_run_id):
    #Analyse complète des nouvelles configurations de l'exercice 5#
    
    # Définition des configurations à tester (nouvelles seulement)
    configurations = [
        # Nouvelles tokenizations
        {'name': 'Tokenization étendue', 'tokenization': 'extended', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        {'name': 'Tokenization avec traits d\'union', 'tokenization': 'hyphen', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        {'name': 'Tokenization avec apostrophes', 'tokenization': 'apostrophe', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        
        # Nouveaux stemmers
        {'name': 'Stemmer Snowball', 'tokenization': 'basic', 'stemmer': 'snowball', 'stop_words': 'nostop'},
        
        # Nouvelles listes de stop-words
        {'name': 'Stop-words 319', 'tokenization': 'basic', 'stemmer': 'nostem', 'stop_words': 'stop319'},
        {'name': 'Stop-words 733', 'tokenization': 'basic', 'stemmer': 'nostem', 'stop_words': 'stop733'},
    ]
    
    weightings = ["ltn", "ltc", "bm25"]
    current_run_id = start_run_id
    
    print("\n" + "="*70)
    print("EXERCICE 5: ANALYSE DES NOUVELLES CONFIGURATIONS")
    print("="*70)
    
    for config in configurations:
        config_name = config['name']
        tokenization = config['tokenization']
        stemmer = config['stemmer']
        stop_words = config['stop_words']
        
        print(f"\n--- {config_name} ---")
        print(f"Paramètres: tokenization={tokenization}, stemmer={stemmer}, stop_words={stop_words}")
        print("-" * 50)
        
        # Création de l'index avec cette configuration
        index_data = create_index_with_config(data_file_path, tokenization, stemmer, stop_words)
        
        for weighting in weightings:
            print(f"\n  Testing {weighting.upper()}...")
            
            # Calcul des statistiques détaillées
            ranker_result = compute_statistics(5, index_data, weighting)
            
            # Génération du run INEX
            generate_inex_run(current_run_id, ranker_result, queries, weighting, 
                            "article", stemmer, stop_words, tokenization)
            
            current_run_id += 1
    
    return current_run_id
"""

def exercise5_complete_analysis(data_file_path, queries, start_run_id):
    """Version avec stockage des résultats et affichage final"""
    
    configurations = [
        {'name': 'Tokenization étendue', 'tokenization': 'extended', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        {'name': 'Tokenization avec traits d\'union', 'tokenization': 'hyphen', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        {'name': 'Tokenization avec apostrophes', 'tokenization': 'apostrophe', 'stemmer': 'nostem', 'stop_words': 'nostop'},
        {'name': 'Stemmer Snowball', 'tokenization': 'basic', 'stemmer': 'snowball', 'stop_words': 'nostop'},
        {'name': 'Stop-words 319', 'tokenization': 'basic', 'stemmer': 'nostem', 'stop_words': 'stop319'},
        {'name': 'Stop-words 733', 'tokenization': 'basic', 'stemmer': 'nostem', 'stop_words': 'stop733'},
    ]
    
    weightings = ["ltn", "ltc", "bm25"]
    current_run_id = start_run_id
    summary_results = []
    
    print("\n" + "="*70)
    print("EXERCICE 5: ANALYSE AVEC TEMPS D'INDEXATION")
    print("="*70)
    print("Calcul en cours...")
    
    # PHASE 1: Stockage de tous les résultats
    for config in configurations:
        config_name = config['name']
        tokenization = config['tokenization']
        stemmer = config['stemmer']
        stop_words = config['stop_words']
        
        print(f"Traitement de: {config_name}")
        
        # Création de l'index avec mesure du temps
        index_data = create_index_with_config(data_file_path, tokenization, stemmer, stop_words)
        indexing_time = index_data['indexing_time']
        
        for weighting in weightings:
            # Calcul des statistiques et génération du run
            ranker = compute_statistics(5, index_data, weighting)
            generate_inex_run(current_run_id, ranker, queries, weighting, 
                            "article", stemmer, stop_words, tokenization)
            
            # Stockage pour l'affichage final
            summary_results.append({
                'run_id': current_run_id,
                'config': config_name,
                'weighting': weighting,
                'indexing_time': indexing_time,
                'tokenization': tokenization,
                'stemmer': stemmer,
                'stop_words': stop_words
            })
            
            current_run_id += 1
    
    # PHASE 2: Affichage du tableau final
    print(f"\n{'Run ID':<6} {'Configuration':<35} {'Weighting':<8} {'Temps Indexation':<15}")
    print("-" * 70)
    
    for result in summary_results:
        print(f"{result['run_id']:<6} {result['config']:<35} {result['weighting'].upper():<8} {result['indexing_time']:<15.2f}s")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL - EXERCICE 5")
    print("="*70)
    
    # Temps total d'indexation
    total_indexing_time = sum(result['indexing_time'] for result in summary_results)
    print(f"Temps total d'indexation: {total_indexing_time:.2f} secondes")
    print(f"Nombre total de runs générés: {len(summary_results)}")
    
    # Configuration la plus rapide
    fastest_config = min(summary_results, key=lambda x: x['indexing_time'])
    print(f"Configuration la plus rapide: Run {fastest_config['run_id']} - {fastest_config['config']} ({fastest_config['indexing_time']:.2f}s)")
    
    # Configuration la plus lente
    slowest_config = max(summary_results, key=lambda x: x['indexing_time'])
    print(f"Configuration la plus lente: Run {slowest_config['run_id']} - {slowest_config['config']} ({slowest_config['indexing_time']:.2f}s)")
    
    # Temps moyen par configuration
    avg_indexing_time = total_indexing_time / len(configurations)
    print(f"Temps moyen par configuration: {avg_indexing_time:.2f}s")
    
    return current_run_id

def exercice6_bm25_tuning(index_data, queries, start_run_id):
    """Version optimisée du tuning BM25 - génère seulement les runs sans statistiques détaillées"""
    b_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0 à 1.0 step 0.1
    k1_values = [round(i * 0.2, 1) for i in range(21)]  # 0.0 à 4.0 step 0.2

    print(f"BM25 Tuning: testing {len(b_values)} b values and {len(k1_values)} k1 values...")

    run_id = start_run_id
    index = index_data['index']
    config = index_data['config']
    
    stemmer = config['stemmer']
    stopwords = config['stop_words']
    tokenizer = config['tokenization']

    # Créer le ranker une seule fois
    ranker = RankedRetrieval(index)

    # Fix k1=1.2, tester b
    for b in b_values:
        print(f"Testing b={b} with k1=1.2, run_id={run_id}")
        
        # Génération directe du run sans calcul de statistiques
        generate_inex_run(run_id, ranker, queries, "bm25", "article", stemmer, stopwords, tokenizer, 1.2, b)
        
        run_id += 1

    # Fix b=0.75, tester k1
    for k1 in k1_values:
        print(f"Testing k1={k1} with b=0.75, run_id={run_id}")
        
        # Génération directe du run sans calcul de statistiques
        generate_inex_run(run_id, ranker, queries, "bm25", "article", stemmer, stopwords, tokenizer, k1, 0.75)
        
        run_id += 1
    
    return run_id


def main():
    data_file_path = "data/Text_Only_Ascii_Coll_NoSem"
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
    
    # --- Exercise 1: SMART LTN ---    
    # Calcul des statistiques pour LTN
    ranker_ltn = compute_statistics(exercise_num=1, index_data=index_no_stop_no_stem, weighting_scheme="ltn")
    # Génération du run INEX pour LTN
    generate_inex_run(current_run_id, ranker_ltn, queries, "ltn", "article", "nostem", "nostop")
    current_run_id += 1
    
    # --- Exercise 2: SMART LTC ---
    ranker_ltc = compute_statistics(exercise_num=2, index_data=index_no_stop_no_stem, weighting_scheme="ltc")
    generate_inex_run(current_run_id, ranker_ltc, queries, "ltc", "article", "nostem", "nostop")
    current_run_id += 1
    
    # --- Exercise 3: BM25 (k1 et b par défaut, si BM25) ---
    ranker_bm25 = compute_statistics(exercise_num=3, index_data=index_no_stop_no_stem, weighting_scheme="bm25")
    generate_inex_run(current_run_id, ranker_bm25, queries, "bm25", "article", "nostem", "nostop")
    current_run_id += 1
    
    
    # --- Exercise 4: test runs avec variantes d'index (12 combinaisons) ---
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
            generate_inex_run(current_run_id, ranker_result, queries, weighting, "article", stemmer_name, stop_name)
            
            current_run_id += 1

    
    # --- Exercise 5: Test d'autres méthode de Tokenization, stemmer, stop-words, weighting ---
    start_run_id_ex5 = current_run_id
    current_run_id = exercise5_complete_analysis(data_file_path, queries, current_run_id)
    
    # Bilan 
    total_runs_ex5 = current_run_id - start_run_id_ex5
    print(f"\n" + "-"*60)
    print("BILAN EXERCICE 5")
    print("="*60)
    print(f"Runs générés: {total_runs_ex5}")
    print(f"Run ID final: {current_run_id}")
    print("Toutes les configurations ont été testées avec succès.")

        
    # --- Exercise 6: BM25 tuning ---
    print("\n" + "=" * 60)
    print("EXERCICE 6: BM25 TUNING")
    print("=" * 60)

    # Utiliser l'index de base (no stop, no stem) pour le tuning
    print("Starting BM25 tuning with base configuration...")
    start_run_id_ex6 = current_run_id
    current_run_id = exercice6_bm25_tuning(index_stop_stem, queries, current_run_id)
    
    print("BM25 tuning completed!")
    print(f"Total runs générés: {current_run_id - start_run_id_ex6}")
    print(f"Run ID final: {current_run_id}")
    

if __name__ == "__main__":
     # Nettoyer le dossier runs au début
    if os.path.exists("data/runs"):
        for file in os.listdir("data/runs"):
            if file.endswith(".txt"):
                os.remove(os.path.join("data/runs", file))
        print("Dossier 'runs' nettoyé")

    main()