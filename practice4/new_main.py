import os
import time
from new_indexer import WeightedInvertedIndex
from new_ranker import RankedRetrieval

def compute_statistics(exercise_num, file_name, use_stop_words=False, use_stemmer=False, stemmer_name="porter", stop_list_name="stop671"):
    """Fonction générique pour les exercices de statistiques"""
    print(f"\nCONFIGURATION {exercise_num}: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET STEMMING")
    print("=" * 60)

    index = WeightedInvertedIndex()
    
    # Configuration simplifiée
    index.configure_stemmer(stemmer_name if use_stemmer else "nostem")
    index.configure_stop_words(stop_list_name if use_stop_words else "nostop")

    indexing_time = index.build_index("data/" + file_name, False)

    if indexing_time is None:
        print("Échec de l'indexation...")
        return None

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

    return index

def generate_run_filename(team_name, run_id, weighting, granularity, stemmer, stop_words, tokenization="article", params=""):
    """Génère un nom de fichier selon le template INEX"""
    filename = f"{team_name}_{run_id}_{weighting}_{granularity}_{stemmer}_{stop_words}"
    if tokenization != "article":
        filename += f"_{tokenization}"
    if params:
        filename += f"_{params}"
    return filename + ".txt"

def run_weighting_experiment(index, query_id, weighting_scheme, query_request, run_id, k1=1.2, b=0.75, is_tuning=False):
    """Exécute les exercices avec mesure du temps et génération de runs - VERSION OPTIMISÉE"""
    print(f"\n{query_id}: {weighting_scheme.upper()} WEIGHTING" + (f" - k1={k1}, b={b}" if weighting_scheme == "bm25" else ""))
    print("-" * 60)

    start_time = time.time()
    ranker = RankedRetrieval(index)

    query_terms = ranker.process_query_terms(query_request)
    term = query_terms[1] if len(query_terms) >= 2 else query_terms[0] if query_terms else "ranking"

    # UNE SEULE RECHERCHE pour le top-1500 (qui inclut le top-10)
    if weighting_scheme == "bm25":
        ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme, k1=k1, b=b)
        doc_score = sum(ranker.get_term_weight(t, "23724", weighting_scheme, k1=k1, b=b) for t in query_terms)
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=1500, k1=k1, b=b)
    else:
        ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme)
        doc_score = sum(ranker.get_term_weight(t, "23724", weighting_scheme) for t in query_terms)
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=1500)

    weighting_time = time.time() - start_time

    print(f"- Temps de pondération: {weighting_time:.2f}s")
    print(f"- Poids de '{term}' dans doc #23724: {ranking_weight:.6f}")
    print(f"- RSV du document #23724: {doc_score:.6f}")

    # Affichage du top-10 (extrait des 1500 premiers)
    print("- TOP-10 DOCUMENTS:")
    for i, (doc_id, score) in enumerate(top_docs[:10], 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")

    # Génération du run
    params = f"k1_{k1}_b_{b}" if weighting_scheme == "bm25" else ""
    run_filename = generate_run_filename(
        team_name="AlphaAnaClement",
        run_id=run_id,
        weighting=weighting_scheme,
        granularity="article",
        stemmer=index.stemmer_name,
        stop_words=index.stop_list_name,
        tokenization=index.tokenization_method,
        params=params
    )

    # Sauvegarde du run complet (top-1500)
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/{run_filename}", "w", encoding="utf-8") as f:
        for i, (doc_id, score) in enumerate(top_docs, 1):
            f.write(f"{query_id} Q0 {doc_id} {i} {score} AlphaAnaClement /article[1]\n")

    print(f"- Run sauvegardé: {run_filename} ({len(top_docs)} documents)")
    
    return weighting_time, ranking_weight, doc_score, top_docs

def create_index_with_config(tokenization="basic", stemmer="nostem", stop_words="nostop"):
    """Crée un index avec une configuration spécifique"""
    index = WeightedInvertedIndex()
    index.configure_tokenization(tokenization)
    index.configure_stemmer(stemmer)
    index.configure_stop_words(stop_words)
    index.build_index("data/Text_Only_Ascii_Coll_NoSem", False)
    return index

"""
def exercise5_stemmers(queries, start_run_id):
    #Teste différents algorithmes de stemming
    
    stemmers = ["nostem", "porter", "snowball", "lovins", "paice"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stemmer in stemmers:
        print(f"\n--- Testing stemmer: {stemmer} ---")
        
        index = create_index_with_config(stemmer=stemmer)
        
        for weighting in weightings:
            for query_id, query_text in queries.items():
                run_weighting_experiment(
                    index, query_id, weighting, query_text, 
                    current_run_id
                )
            current_run_id += 1
    
    return current_run_id
    """
def exercise5_stemmers(queries, start_run_id):
    """Teste différents algorithmes de stemming - VERSION SIMPLIFIÉE"""
    
    # GARDER seulement porter et snowball
    stemmers = ["nostem", "porter", "snowball"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stemmer in stemmers:
        print(f"\n--- Testing stemmer: {stemmer} ---")
        
        try:
            index = create_index_with_config(stemmer=stemmer)
            
            for weighting in weightings:
                for query_id, query_text in queries.items():
                    run_weighting_experiment(
                        index, query_id, weighting, query_text, 
                        current_run_id
                    )
                current_run_id += 1
        except Exception as e:
            print(f"Erreur avec le stemmer {stemmer}: {e}")
            continue
    
    return current_run_id

def exercise5_stemmers(queries, start_run_id):
    """Teste différents algorithmes de stemming"""
    stemmers = ["nostem", "porter", "snowball"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stemmer in stemmers:
        print(f"\n--- Testing stemmer: {stemmer} ---")
        try:
            index = create_index_with_config(stemmer=stemmer)
            for weighting in weightings:
                for query_id, query_text in queries.items():
                    run_weighting_experiment(index, query_id, weighting, query_text, current_run_id)
                current_run_id += 1
        except Exception as e:
            print(f"Erreur avec le stemmer {stemmer}: {e}")
    
    return current_run_id

def exercise5_tokenization(queries, start_run_id):
    """Teste différentes méthodes de tokenization"""
    
    tokenizations = ["basic", "extended", "hyphen", "apostrophe"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for tokenization in tokenizations:
        print(f"\n--- Testing tokenization: {tokenization} ---")
        
        index = create_index_with_config(tokenization=tokenization)
        
        for weighting in weightings:
            for query_id, query_text in queries.items():
                run_weighting_experiment(
                    index, query_id, weighting, query_text,
                    current_run_id
                )
            current_run_id += 1
    
    return current_run_id


def exercise5_stop_words(queries, start_run_id):
    """Teste différentes listes de stop-words"""
    
    stop_lists = ["nostop", "stop344", "stop571", "stop671", "stop759"]
    weightings = ["ltn", "ltc", "bm25"]
    
    current_run_id = start_run_id
    
    for stop_list in stop_lists:
        print(f"\n--- Testing stop-words: {stop_list} ---")
        
        index = create_index_with_config(stop_words=stop_list)
        
        for weighting in weightings:
            for query_id, query_text in queries.items():
                run_weighting_experiment(
                    index, query_id, weighting, query_text,
                    current_run_id
                )
            current_run_id += 1
    
    return current_run_id


def bm25_tuning(index, queries, start_run_id):
    """Teste plusieurs valeurs de k1 et b pour BM25 et génère des runs"""
    b_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0 à 1.0 step 0.1
    k1_values = [round(i * 0.2, 1) for i in range(21)]  # 0.0 à 4.0 step 0.2

    print(f"Testing {len(b_values)} b values and {len(k1_values)} k1 values...")

    run_id = start_run_id

    # Fix k1=1.2, tester b
    for b in b_values:
        print(f"Testing b={b} with k1=1.2, run_id={run_id}")
        for query_id, query_request in queries.items():
            run_weighting_experiment(index, query_id, "bm25", query_request, run_id, k1=1.2, b=b, is_tuning=True)
        run_id += 1

    # Fix b=0.75, tester k1
    for k1 in k1_values:
        print(f"Testing k1={k1} with b=0.75, run_id={run_id}")
        for query_id, query_request in queries.items():
            run_weighting_experiment(index, query_id, "bm25", query_request, run_id, k1=k1, b=0.75, is_tuning=True)
        run_id += 1
    
    return run_id

def test_cache_compatibility():
    """Test pour vérifier que le cache fonctionne avec le nouveau code"""
    index = WeightedInvertedIndex()
    index.configure_stop_words("stop671")
    index.configure_stemmer("porter")
    index.build_index("data/Text_Only_Ascii_Coll_NoSem", False)
    
    print(f"stop_word_active: {index.stop_word_active}")
    print(f"stemmer_active: {index.stemmer_active}")
    print(f"stop_list_name: {index.stop_list_name}")
    print(f"stemmer_name: {index.stemmer_name}")
    
    # Test du ranker
    ranker = RankedRetrieval(index)
    print("Ranker initialisé avec succès!")
    
    # Test LTC (doit charger/calculer les normes)
    weight = ranker.get_term_weight("ranking", "23724", "ltc")
    print(f"Poids LTC: {weight}")

def test_snowball_compatibility():
    """Test spécifique pour vérifier l'intégration de Snowball"""
    print("Test d'intégration Snowball...")
    
    # Test basique
    test_words = ["running", "cats", "happily"]
    for word in test_words:
        stemmed = stem_word(word)
        print(f"  {word} -> {stemmed}")
    
    # Test avec l'index
    index = WeightedInvertedIndex()
    index.configure_stemmer("snowball")
    index.configure_stop_words("nostop")
    
    # Test sur un petit échantillon
    tokens = ["running", "jumping", "happily", "agreement"]
    processed = index.process_tokens(tokens)
    print(f"Tokens traités: {tokens} -> {processed}")


def main():
    queries = {
        2009011: "olive oil health benefit",
        2009036: "nothing hill film actors", 
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm", 
        2009085: "operating system mutual exclusion"
    }
    
    # Créer le répertoire runs s'il n'existe pas
    os.makedirs("runs", exist_ok=True)

    # CALCULER le run_id de départ UNE SEULE FOIS
    base_run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
    current_run_id = base_run_id
    
    print(f"Run ID de départ: {current_run_id}")


    # === EXERCICES 1-3: Configurations de base + SMART LTN ===
    print("\n" + "=" * 60)
    print("EXERCICES 1: SMART LTN - SANS STOP-WORDS ET STEMMING")
    print("=" * 60)
    
    # Construction des index de base pour les exercices 1-3
    index_no_stop_no_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem", use_stop_words=False, use_stemmer=False)
    
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltn", query_request, current_run_id)
    current_run_id += 1
    """
    # Exercice 2: SMART LTC
    print("\n" + "=" * 60)
    print("EXERCICES 2: SMART LTC - SANS STOP-WORDS ET STEMMING")
    print("=" * 60)

    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltc", query_request, current_run_id)
    current_run_id += 1

    
    # Exercice 3: BM25
    print("\n" + "=" * 60)
    print("EXERCICES 3: BM25 - SANS STOP-WORDS ET STEMMING")
    print("=" * 60)

    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "bm25", query_request, current_run_id)
    current_run_id += 1
    """
    """
    # Exercice 4: Test runs avec variantes d'index
    print("\n" + "=" * 60)
    print("EXERCICES 4: TEST RUNS AVEC VARIANTES D'INDEX")
    print("=" * 60)
    
    index_stop_no_stem = compute_statistics(2, "Text_Only_Ascii_Coll_NoSem", use_stop_words=True, use_stemmer=False)
    index_no_stop_stem = compute_statistics(3, "Text_Only_Ascii_Coll_NoSem", use_stop_words=False, use_stemmer=True)
    index_stop_stem = compute_statistics(4, "Text_Only_Ascii_Coll_NoSem", use_stop_words=True, use_stemmer=True)

    algorithms = ["ltn", "ltc", "bm25"]
    indexers = [
        ("no_stop_no_stem", index_no_stop_no_stem),
        ("stop_no_stem", index_stop_no_stem), 
        ("no_stop_stem", index_no_stop_stem),
        ("stop_stem", index_stop_stem)
    ]
    
    for index_name, index in indexers:
        for algorithm in algorithms:
            print(f"\n--- {index_name} avec {algorithm} ---")
            for query_id, query_request in queries.items():
                run_weighting_experiment(index, query_id, algorithm, query_request, current_run_id)
            current_run_id += 1
    """
    
    """
    # === EXERCICE 5: Exploration avancée ===
    print("\n" + "=" * 60)
    print("EXERCICE 5: EXPLORATION AVANCÉE")
    print("=" * 60)
    
    print("\n1. Test des algorithmes de stemming...")
    current_run_id = exercise5_stemmers(queries, current_run_id)
    
    print("\n2. Test des méthodes de tokenization...")
    current_run_id = exercise5_tokenization(queries, current_run_id)
    
    print("\n3. Test des listes de stop-words...")
    current_run_id = exercise5_stop_words(queries, current_run_id)
    """
    #test_snowball_compatibility()

    
    """
    # === EXERCICE 6: BM25 Tuning ===
    print("\n" + "=" * 60)
    print("EXERCICE 6: BM25 TUNING")
    print("=" * 60)

    print("Starting BM25 tuning...")
    current_run_id = bm25_tuning(index_no_stop_no_stem, queries, current_run_id)
    print("BM25 tuning completed!")

    print(f"\n=== TOUS LES EXERCICES TERMINÉS ===")
    print(f"Total des runs générés: {current_run_id - base_run_id}")
    print(f"Run ID final: {current_run_id}")
    """
if __name__ == "__main__":
    main()