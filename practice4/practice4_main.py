import os
import time

from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval_optimized import RankedRetrieval


def compute_statistics(exercise_num, file_name, use_stop_words=False, use_stemmer=False):
    """Fonction générique pour les exercices de statistiques"""
    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET STEMMING")
    print("=" * 60)

    index = WeightedInvertedIndex()
    index.stop_word_active = use_stop_words
    index.stemmer_active = use_stemmer

    if use_stop_words:
        index.load_stop_words()

    indexing_time = index.build_index("data/" + file_name, False)

    if indexing_time is None:
        print("Échec de l'indexation...")
        return None, 0, {}

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


def run_weighting_experiment(index, query_id, weighting_scheme, query_request, run_id, k1=1.2, b=0.75, is_tuning=False):
    """Exécute les exercices avec mesure du temps et génération de runs"""

    if is_tuning:
        print("\n" + "=" * 60)
        print(f"{query_id}: BM25 TUNING - k1={k1}, b={b}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        if weighting_scheme == "bm25":
            print(f"{query_id}: BM25 WEIGHTING - k1={k1}, b={b}")
        else:
            print(f"{query_id}: {weighting_scheme.upper()} WEIGHTING")
        print("=" * 60)

    start_time = time.time()
    ranker = RankedRetrieval(index, cache_dir="data/norm_cache")

    query_terms = ranker.process_query_terms(query_request)

    term = query_terms[1]  # terme après traitement

    # Utilise k1 et b seulement pour BM25
    if weighting_scheme == "bm25":
        ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme, k1=k1, b=b)
        doc_score = sum(ranker.get_term_weight(term, "23724", weighting_scheme, k1=k1, b=b) for term in query_terms)
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=10, k1=k1, b=b)
    else:
        ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme)
        doc_score = sum(ranker.get_term_weight(term, "23724", weighting_scheme) for term in query_terms)
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=10)

    weighting_time = time.time() - start_time

    print(f"- Temps de pondération TOTAL: {weighting_time:.2f} secondes")
    print(f"- Poids de 'ranking' dans doc #23724: {ranking_weight:.6f}")
    print(f"- RSV du document #23724: {doc_score:.6f}")

    print("- TOP-10 DOCUMENTS:")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")

    # Même logique pour top_k=1500
    if weighting_scheme == "bm25":
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=1500, k1=k1, b=b)
    else:
        top_docs = ranker.search_query(query_request, weighting_scheme, top_k=1500)

    file_name = f"AlphaAnaClement_{run_id}_test_{weighting_scheme}_article"
    file_name += "_stop671" if index.stop_word_active else "_nostop"
    file_name += "_porter" if index.stemmer_active else "_nostem"
    if weighting_scheme == "bm25":
        file_name += f"_k1_{k1}_b_{b}"
    file_name += ".txt"

    if not os.path.exists("runs/" + file_name):
        with open("runs/" + file_name, "w", encoding="utf-8") as f:
            f.write("")

    for i, (doc_id, score) in enumerate(top_docs, 1):
        with open("runs/" + file_name, "a", encoding="utf-8") as f:
            f.write(f"{query_id} Q0 {doc_id} {i} {score} AlphaAnaClement /article[1]\n")

    return weighting_time, ranking_weight, doc_score, top_docs


def bm25_tuning(index, queries):
    """Teste plusieurs valeurs de k1 et b pour BM25 et génère des runs"""
    b_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0 à 1.0 step 0.1
    k1_values = [round(i * 0.2, 1) for i in range(21)]  # 0.0 à 4.0 step 0.2

    print(f"Testing {len(b_values)} b values and {len(k1_values)} k1 values...")

    # Fix k1=1.2, tester b
    for b in b_values:
        run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
        print(f"Testing b={b} with k1=1.2, run_id={run_id}")
        for query_id, query_request in queries.items():
            run_weighting_experiment(index, query_id, "bm25", query_request, run_id, k1=1.2, b=b, is_tuning=True)

    # Fix b=0.75, tester k1
    for k1 in k1_values:
        run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
        print(f"Testing k1={k1} with b=0.75, run_id={run_id}")
        for query_id, query_request in queries.items():
            run_weighting_experiment(index, query_id, "bm25", query_request, run_id, k1=k1, b=0.75, is_tuning=True)


def main():
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion",
    }

    # Construction des index
    index_no_stop_no_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem", use_stop_words=False, use_stemmer=False)
    index_stop_no_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem", use_stop_words=True, use_stemmer=False)
    index_stop_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem", use_stop_words=True, use_stemmer=True)
    index_no_stop_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem", use_stop_words=False, use_stemmer=True)

    # --- Exercise 1: SMART LTN ---
    run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltn", query_request, run_id)

    # --- Exercise 2: SMART LTC ---
    run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltc", query_request, run_id)

    # --- Exercise 3: BM25 ---
    run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "bm25", query_request, run_id)

    # --- Exercise 4 & 5: test runs avec variantes d'index ---
    algorithms = ["ltn", "ltc", "bm25"]
    indexers = [index_stop_no_stem, index_stop_stem, index_no_stop_stem]
    for index in indexers:
        for algorithm in algorithms:
            run_id = len([f for f in os.listdir("runs") if os.path.isfile(os.path.join("runs", f))])
            for query_id, query_request in queries.items():
                run_weighting_experiment(index, query_id, algorithm, query_request, run_id)

    # --- Exercise 6: BM25 tuning ---
    print("\n" + "=" * 60)
    print("EXERCICE 6: BM25 TUNING")
    print("=" * 60)

    print("Starting BM25 tuning...")
    bm25_tuning(index_no_stop_no_stem, queries)
    print("BM25 tuning completed!")


if __name__ == "__main__":
    main()