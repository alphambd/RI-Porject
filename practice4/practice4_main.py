import os
import time

from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval_optimized import RankedRetrieval


def compute_statistics(exercise_num, file_name, use_stop_words=False, use_stemmer=False):
    """Fonction générique pour les exercices de statistiques"""
    print("\n" + "=" * 60)
    print(
        f"CONSTRUCTION INDEX: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS, {'AVEC' if use_stemmer else 'SANS'} STEMMING")
    print("=" * 60)

    index = WeightedInvertedIndex()
    index.stop_word_active = use_stop_words
    index.stemmer_active = use_stemmer

    if use_stop_words:
        index.load_stop_words()

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

    # CORRECTION: Mode 'a' pour ajouter les résultats de chaque requête
    with open("runs/" + file_name, "a", encoding="utf-8") as f:
        for i, (doc_id, score) in enumerate(top_docs, 1):
            f.write(f"{query_id} Q0 {doc_id} {i} {score} AlphaAnaClement /article[1]\n")

    print(f"- {len(top_docs)} résultats ajoutés pour {query_id} dans {file_name}")

    return weighting_time, ranking_weight, doc_score, top_docs


def generate_run_for_all_queries(index, queries, run_id, weighting_scheme, k1=1.2, b=0.75, is_tuning=False):
    """Génère un run complet avec les 7 requêtes (nouvelle fonction)"""
    file_name = f"AlphaAnaClement_{run_id}_test_{weighting_scheme}_article"
    file_name += "_stop671" if index.stop_word_active else "_nostop"
    file_name += "_porter" if index.stemmer_active else "_nostem"
    if weighting_scheme == "bm25":
        file_name += f"_k1_{k1}_b_{b}"
    file_name += ".txt"

    print(f"\n⭐ Génération du run {run_id}: {file_name}")

    # Nettoyer le fichier au début
    with open("runs/" + file_name, "w", encoding="utf-8") as f:
        f.write("")  # Fichier vide au début

    total_lines = 0
    for query_id, query_request in queries.items():
        # Appeler run_weighting_experiment qui va ajouter au fichier
        _, _, _, top_docs = run_weighting_experiment(
            index, query_id, weighting_scheme, query_request, run_id, k1, b, is_tuning
        )
        total_lines += len(top_docs)

    print(f"⭐ Run {run_id} complété: {total_lines} lignes totales")
    return run_id + 1


def evaluate_bm25_score(index, queries, k1, b):
    """Évalue le score moyen BM25 pour des paramètres donnés"""
    total_score = 0
    num_queries = 0

    for query_id, query_request in queries.items():
        ranker = RankedRetrieval(index, cache_dir="data/norm_cache")
        top_docs = ranker.search_query(query_request, "bm25", top_k=10, k1=k1, b=b)

        if top_docs:
            total_score += sum(score for _, score in top_docs)
            num_queries += 1

    return total_score / num_queries if num_queries > 0 else 0


def bm25_gradient_descent_optimization(index, queries, start_run_id):
    """Optimisation BM25 par gradient descent simplifié"""
    print("\n" + "=" * 60)
    print("GRADIENT DESCENT OPTIMIZATION")
    print("=" * 60)

    # Point de départ
    k1, b = 1.2, 0.75
    learning_rate = 0.1
    best_score = evaluate_bm25_score(index, queries, k1, b)
    best_params = (k1, b)

    print(f"Démarrage: k1={k1}, b={b}, score={best_score:.6f}")

    for iteration in range(10):
        # Calcul des gradients approximatifs
        grad_k1 = (evaluate_bm25_score(index, queries, k1 + 0.01, b) - best_score) / 0.01
        grad_b = (evaluate_bm25_score(index, queries, k1, b + 0.01) - best_score) / 0.01

        # Mise à jour des paramètres
        new_k1 = k1 - learning_rate * grad_k1
        new_b = b - learning_rate * grad_b

        # Contrainte des bornes
        new_k1 = max(0.1, min(4.0, new_k1))
        new_b = max(0.0, min(1.0, new_b))

        new_score = evaluate_bm25_score(index, queries, new_k1, new_b)

        print(f"Itération {iteration + 1}: k1={new_k1:.3f}, b={new_b:.3f}, score={new_score:.6f}")

        if new_score > best_score:
            best_score = new_score
            k1, b = new_k1, new_b
            best_params = (k1, b)
            print(f"  → Amélioration! Nouveau meilleur score")
        else:
            learning_rate *= 0.5
            print(f"  → Pas d'amélioration, learning rate réduit à {learning_rate:.3f}")

        if learning_rate < 0.01:
            print("  → Learning rate trop petit, arrêt de l'optimisation")
            break

    optimal_k1, optimal_b = best_params
    print(f"\n⭐ OPTIMISATION TERMINÉE!")
    print(f"⭐ Paramètres optimaux: k1={optimal_k1:.3f}, b={optimal_b:.3f}")
    print(f"⭐ Meilleur score: {best_score:.6f}")

    # Génération du run optimal
    run_id = start_run_id
    run_id = generate_run_for_all_queries(
        index, queries, run_id, "bm25", optimal_k1, optimal_b, is_tuning=True
    )

    return optimal_k1, optimal_b, run_id


def bm25_tuning(index, queries, start_run_id):
    """Teste plusieurs valeurs de k1 et b pour BM25 et génère des runs"""
    b_values = [round(i * 0.1, 1) for i in range(11)]  # 0.0 à 1.0 step 0.1
    k1_values = [round(i * 0.2, 1) for i in range(21)]  # 0.0 à 4.0 step 0.2

    print(f"Testing {len(b_values)} b values and {len(k1_values)} k1 values...")

    run_id = start_run_id

    # Fix k1=1.2, tester b
    for b in b_values:
        print(f"Testing b={b} with k1=1.2, run_id={run_id}")
        run_id = generate_run_for_all_queries(
            index, queries, run_id, "bm25", 1.2, b, is_tuning=True
        )

    # Fix b=0.75, tester k1
    for k1 in k1_values:
        print(f"Testing k1={k1} with b=0.75, run_id={run_id}")
        run_id = generate_run_for_all_queries(
            index, queries, run_id, "bm25", k1, 0.75, is_tuning=True
        )

    return run_id


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

    current_run_id = 1

    # --- Exercise 1: SMART LTN ---
    print("\n" + "=" * 60)
    print("EXERCICE 1: SMART LTN")
    print("=" * 60)
    current_run_id = generate_run_for_all_queries(
        index_no_stop_no_stem, queries, current_run_id, "ltn"
    )

    # --- Exercise 2: SMART LTC ---
    print("\n" + "=" * 60)
    print("EXERCICE 2: SMART LTC")
    print("=" * 60)
    current_run_id = generate_run_for_all_queries(
        index_no_stop_no_stem, queries, current_run_id, "ltc"
    )

    # --- Exercise 3: BM25 ---
    print("\n" + "=" * 60)
    print("EXERCICE 3: BM25")
    print("=" * 60)
    current_run_id = generate_run_for_all_queries(
        index_no_stop_no_stem, queries, current_run_id, "bm25"
    )

    # --- Exercise 4 : test runs avec variantes d'index ---
    print("\n" + "=" * 60)
    print("EXERCICE 4: VARIANTS AVEC STOP-WORDS & STEMMING")
    print("=" * 60)
    algorithms = ["ltn", "ltc", "bm25"]
    indexers = [index_stop_no_stem, index_stop_stem, index_no_stop_stem]

    for index in indexers:
        for algorithm in algorithms:
            current_run_id = generate_run_for_all_queries(
                index, queries, current_run_id, algorithm
            )

    # --- Exercise 6: BM25 tuning ---
    print("\n" + "=" * 60)
    print("EXERCICE 6: BM25 TUNING")
    print("=" * 60)

    print("Starting BM25 grid search...")
    current_run_id = bm25_tuning(index_no_stop_no_stem, queries, current_run_id)

    print("\nStarting BM25 gradient descent optimization...")
    optimal_k1, optimal_b, current_run_id = bm25_gradient_descent_optimization(
        index_no_stop_no_stem, queries, current_run_id
    )

    print(f"\n⭐ TOUS LES EXERCICES TERMINÉS!")
    print(f"⭐ Total runs générés: {current_run_id - 1}")
    print(f"⭐ Paramètres BM25 optimaux: k1={optimal_k1:.3f}, b={optimal_b:.3f}")


if __name__ == "__main__":
    # Nettoyer le dossier runs au début
    if os.path.exists("runs"):
        for file in os.listdir("runs"):
            if file.endswith(".txt"):
                os.remove(os.path.join("runs", file))
        print("Dossier 'runs' nettoyé")

    main()