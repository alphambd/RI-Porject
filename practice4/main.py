import time
import os
import shutil
from indexer import WeightedInvertedIndex
from ranker import RankedRetrieval

# Configuration globale
TEAM_NAME = "VictorAlbertJules"
DATA_PATH = "data/Text_Only_Ascii_Coll_NoSem"

def get_official_queries():
    """Retourne les 7 requêtes officielles INEX"""
    return {
        "2009011": "olive oil health benefit",
        "2009036": "nothing hill film actors",
        "2009067": "probabilistic models in information retrieval",
        "2009073": "web link network analysis", 
        "2009074": "web ranking scoring algorithm",
        "2009078": "supervised machine learning algorithm",
        "2009085": "operating system mutual exclusion"
    }

def compute_statistics(index, exercise_num, use_stop_words=False, use_stemmer=False):
    """Fonction générique pour les exercices de statistiques"""
    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET {'AVEC' if use_stemmer else 'SANS'} STEMMING")
    print("=" * 60)
    
    if use_stop_words:
        index.load_stop_words()
    
    indexing_time = index.build_index(DATA_PATH, False)
    
    if indexing_time is None:
        print("Échec de l'indexation...")
        return None, {}
    
    stats = index.get_collection_statistics(indexing_time)
    
    print(f"\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Temps d'indexation: {stats['indexing_time']:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats['avg_term_length']:.2f} caractères")
    
    return index, stats

def run_weighting_experiment(index, exercise_name, weighting_scheme, run_id, 
                           generate_run=True, k1=1.2, b=0.75):
    """Exécute les exercices avec mesure CORRECTE du temps"""
    print(f"\n" + "=" * 60)
    print(f"{exercise_name}: {weighting_scheme.upper()} WEIGHTING")
    print("=" * 60)

    # Initialisation du moteur de pondération
    ranker = RankedRetrieval(index, cache_dir="data/norm_cache")
    
    # Initialisation du temps de pondération
    start_time = time.time()
    
    # Requête pour tous les exercices
    query = "web ranking scoring algorithm"
    query_terms = ranker.process_query_terms(query)
    
    # Calcul du poids pour le terme "ranking" dans le document #23724
    ranking_weight = 0.0
    if "ranking" in query_terms:
        ranking_weight = ranker.get_term_weight("ranking", "23724", weighting_scheme, k1, b)
    else:
        # Chercher le terme le plus proche
        for term in query_terms:
            if "rank" in term:
                ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme, k1, b)
                break
    
    # Calcul du RSV du document #23724
    doc_score = ranker.calculate_document_rsv(query, "23724", weighting_scheme, k1, b)

    # Recherche du Top-10
    top_docs = ranker.search_query(query, weighting_scheme, top_k=10, k1=k1, b=b)
    
    # Génération du run INEX si demandé
    run_file = None
    if generate_run:
        queries_dict = get_official_queries()
        run_file = ranker.generate_inex_run(
            queries_dict, weighting_scheme, run_id, TEAM_NAME,
            "articles", 
            "stop671" if index.stop_word_active else "nostop",
            "porter" if index.stemmer_active else "nostem",
            f"k1_{k1}_b_{b}" if weighting_scheme == "bm25" else "",
            top_k=1500, k1=k1, b=b
        )
    
    # Fin de la mesure du temps
    weighting_time = time.time() - start_time
    
    # Affichage des résultats
    print(f"- Temps de pondération TOTAL: {weighting_time:.2f} secondes")
    print(f"- Poids de 'ranking' dans doc #23724: {ranking_weight:.6f}")
    print(f"- RSV du document #23724: {doc_score:.6f}")
    
    print(f"- TOP-10 DOCUMENTS:")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
    
    return weighting_time, ranking_weight, doc_score, top_docs, run_file

"""
def exercise_1():
    #EXERCICE 1: SMART ltn first run
    print("\n" + "=" * 80)
    print("EXERCICE 1: SMART ltn FIRST RUN")
    print("=" * 80)
    
    # Configuration pour l'exercice 1
    index = WeightedInvertedIndex()
    index.stop_word_active = False
    index.stemmer_active = False
    
    # Calcul des statistiques
    index, stats = compute_statistics(index, "1", False, False)
    
    # Expérience de pondération
    weighting_time, ranking_weight, doc_score, top_docs, run_file = run_weighting_experiment(
        index, "EXERCICE 1", "ltn", "01"
    )
    
    return index, stats, run_file

def exercise_2():
    #EXERCICE 2: SMART ltc first run
    print("\n" + "=" * 80)
    print("EXERCICE 2: SMART ltc FIRST RUN")
    print("=" * 80)
    
    # Configuration pour l'exercice 2
    index = WeightedInvertedIndex()
    index.stop_word_active = False
    index.stemmer_active = False
    
    # Calcul des statistiques
    index, stats = compute_statistics(index, "2", False, False)
    
    # Expérience de pondération
    weighting_time, ranking_weight, doc_score, top_docs, run_file = run_weighting_experiment(
        index, "EXERCICE 2", "ltc", "02"
    )
    
    return index, stats, run_file

def exercise_3():
    #EXERCICE 3: BM25 first run
    print("\n" + "=" * 80)
    print("EXERCICE 3: BM25 FIRST RUN")
    print("=" * 80)
    
    # Configuration pour l'exercice 3
    index = WeightedInvertedIndex()
    index.stop_word_active = False
    index.stemmer_active = False
    
    # Calcul des statistiques
    index, stats = compute_statistics(index, "3", False, False)
    
    # Expérience de pondération avec BM25
    weighting_time, ranking_weight, doc_score, top_docs, run_file = run_weighting_experiment(
        index, "EXERCICE 3", "bm25", "03", k1=1.2, b=0.75
    )
    
    return index, stats, run_file
"""
def first_run_experiment(exercise_num, weighting_scheme, run_id, use_stop_words=False, use_stemmer=False, k1=1.2, b=0.75):
    """
    Fonction générique pour les exercices de pondération
    """
    print(f"\n" + "=" * 80)
    print(f"EXERCICE {exercise_num}: {weighting_scheme.upper()} FIRST RUN")
    print(f"Stop-words: {'AVEC' if use_stop_words else 'SANS'}, Stemming: {'AVEC' if use_stemmer else 'SANS'}")
    print("=" * 80)
    
    # Configuration de l'index
    index = WeightedInvertedIndex()
    index.stop_word_active = use_stop_words
    index.stemmer_active = use_stemmer
    
    if use_stop_words:
        index.load_stop_words()
    
    # Calcul des statistiques
    print("\n" + "=" * 60)
    print(f"CONFIGURATION: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET {'AVEC' if use_stemmer else 'SANS'} STEMMING")
    print("=" * 60)
    
    indexing_time = index.build_index(DATA_PATH, False)
    
    if indexing_time is None:
        print("Échec de l'indexation...")
        return None, {}, None
    
    stats = index.get_collection_statistics(indexing_time)
    
    print(f"\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Temps d'indexation: {stats['indexing_time']:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats['avg_term_length']:.2f} caractères")
    
    # Expérience de pondération
    weighting_time, ranking_weight, doc_score, top_docs, run_file = run_weighting_experiment(
        index, f"EXERCICE {exercise_num}", weighting_scheme, run_id, 
        generate_run=True, k1=k1, b=b
    )
    
    return index, stats, run_file

def exercise_4(base_index):
    """EXERCICE 4: 12 test runs"""
    print("\n" + "=" * 80)
    print("EXERCICE 4: 12 TEST RUNS")
    print("=" * 80)
    
    configurations = [
        # (weighting, stop_words, stemmer, run_id, k1, b)
        ("ltn", False, False, "test_01", 1.2, 0.75),
        ("ltn", False, True,  "test_02", 1.2, 0.75),
        ("ltn", True,  False, "test_03", 1.2, 0.75),
        ("ltn", True,  True,  "test_04", 1.2, 0.75),
        ("ltc", False, False, "test_05", 1.2, 0.75),
        ("ltc", False, True,  "test_06", 1.2, 0.75),
        ("ltc", True,  False, "test_07", 1.2, 0.75),
        ("ltc", True,  True,  "test_08", 1.2, 0.75),
        ("bm25", False, False, "test_09", 1.2, 0.75),
        ("bm25", False, True,  "test_10", 1.2, 0.75),
        ("bm25", True,  False, "test_11", 1.2, 0.75),
        ("bm25", True,  True,  "test_12", 1.2, 0.75)
    ]
    
    generated_runs = []
    
    for config in configurations:
        weighting, stop_words, stemmer, run_id, k1, b = config
        
        # Configurer un nouvel index avec la même configuration de base
        index = WeightedInvertedIndex()
        index.stop_word_active = stop_words
        index.stemmer_active = stemmer
        
        if stop_words:
            index.load_stop_words()
        
        # Reconstruire l'index avec cette configuration
        indexing_time = index.build_index(DATA_PATH, False)
        
        print(f"\n--- Configuration: {weighting}, stop_words={stop_words}, stemmer={stemmer} ---")
        
        # Générer le run
        ranker = RankedRetrieval(index)
        queries_dict = get_official_queries()
        
        run_file = ranker.generate_inex_run(
            queries_dict, weighting, run_id, TEAM_NAME,
            "articles",
            "stop671" if stop_words else "nostop",
            "porter" if stemmer else "nostem",
            f"k1_{k1}_b_{b}" if weighting == "bm25" else "",
            top_k=1500, k1=k1, b=b
        )
        
        generated_runs.append(run_file)
    
    return generated_runs

def exercise_5():
    """EXERCICE 5: Tokenization, stemmer, stop-words, weighting variations"""
    print("\n" + "=" * 80)
    print("EXERCICE 5: VARIATIONS AVANCÉES")
    print("=" * 80)
    
    # Cet exercice étend l'exercice 4 avec plus de variations
    # Pour l'instant, nous réutilisons exercise_4 avec des paramètres supplémentaires
    print("L'exercice 5 étend l'exercice 4 avec des variations supplémentaires.")
    print("Voir les runs générés dans l'exercice 4.")
    
    return []

def exercise_6(base_index):
    """EXERCICE 6: BM25 tuning"""
    print("\n" + "=" * 80)
    print("EXERCICE 6: BM25 PARAMETER TUNING")
    print("=" * 80)
    
    generated_runs = []
    
    # Stratégie 1: k1 fixe à 1.2, b variable de 0.0 à 1.0
    print("\n--- Stratégie 1: k1 fixe (1.2), b variable ---")
    k1_fixed = 1.2
    for b in [i/10 for i in range(0, 11)]:  # 0.0, 0.1, ..., 1.0
        run_id = f"tune_k1_{k1_fixed}_b_{b:.1f}"
        print(f"Génération run: {run_id}")
        
        index = WeightedInvertedIndex()
        index.stop_word_active = False
        index.stemmer_active = False
        index.build_index(DATA_PATH, False)
        
        ranker = RankedRetrieval(index)
        queries_dict = get_official_queries()
        
        run_file = ranker.generate_inex_run(
            queries_dict, "bm25", run_id, TEAM_NAME,
            "articles", "nostop", "nostem", f"k1_{k1_fixed}_b_{b:.1f}",
            top_k=1500, k1=k1_fixed, b=b
        )
        
        generated_runs.append(run_file)
    
    # Stratégie 2: b fixe à 0.75, k1 variable de 0.0 à 4.0
    print("\n--- Stratégie 2: b fixe (0.75), k1 variable ---")
    b_fixed = 0.75
    for k1 in [i/5 for i in range(0, 21)]:  # 0.0, 0.2, ..., 4.0
        run_id = f"tune_k1_{k1:.1f}_b_{b_fixed}"
        print(f"Génération run: {run_id}")
        
        index = WeightedInvertedIndex()
        index.stop_word_active = False
        index.stemmer_active = False
        index.build_index(DATA_PATH, False)
        
        ranker = RankedRetrieval(index)
        queries_dict = get_official_queries()
        
        run_file = ranker.generate_inex_run(
            queries_dict, "bm25", run_id, TEAM_NAME,
            "articles", "nostop", "nostem", f"k1_{k1:.1f}_b_{b_fixed}",
            top_k=1500, k1=k1, b=b_fixed
        )
        
        generated_runs.append(run_file)
    
    return generated_runs

def main():
    """Fonction principale exécutant tous les exercices"""
    all_runs = []
    
    print("DÉBUT DE LA SESSION PRATIQUE 4: ÉVALUATION")
    print("=" * 50)
    
    """
    # Exercice 1: SMART ltn
    index1, stats1, run1 = exercise_1()
    all_runs.append(run1)
    
    # Exercice 2: SMART ltc
    index2, stats2, run2 = exercise_2()
    all_runs.append(run2)
    
    # Exercice 3: BM25
    index3, stats3, run3 = exercise_3()
    all_runs.append(run3)
    """
    # Exercice 1: SMART ltn
    first_run_experiment(exercise_num="1", weighting_scheme="ltn", run_id="01", use_stop_words=False, use_stemmer=False)
    # Exercice 2: SMART ltc
    first_run_experiment(exercise_num="2", weighting_scheme="ltc", run_id="02", use_stop_words=False, use_stemmer=False)
    # Exercice 3: BM25
    first_run_experiment(exercise_num="3", weighting_scheme="bm25", run_id="03", use_stop_words=False, use_stemmer=False, k1=1.2, b=0.75)
    # Exercice 4: 12 test runs (utilise index1 comme base)
    #runs4 = exercise_4(index1)
    #all_runs.extend(runs4)
    
    # Exercice 5:


if __name__ == "__main__":
    main()