import os
import time

from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval_optimized import RankedRetrieval


def compute_statistics(exercise_num, file_name, use_stop_words=False, use_stemmer=False):
    """Calcule et affiche les statistiques de la collection pour différents réglages."""

    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET STEMMING")
    print("=" * 60)

    index = WeightedInvertedIndex()
    index.stop_word_active = use_stop_words
    index.stemmer_active = use_stemmer

    if use_stop_words:
        index.load_stop_words()

    # Construction de l'index
    indexing_time = index.build_index("data/" + file_name, False)

    if indexing_time is None:
        print("Échec lors de la construction de l'index.")
        return None, 0, {}

    # Récupération des statistiques
    stats = index.get_collection_statistics(indexing_time)

    print("\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Temps d'indexation: {stats['indexing_time']:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats['avg_term_length']:.2f} caractères")

    return index


def run_weighting_experiment(index, query_id, weighting_scheme, query_request, run_id):
    """Lance un test complet de pondération avec mesure précise du temps."""

    print("\n" + "=" * 60)
    print(f"{query_id}: {weighting_scheme.upper()} WEIGHTING")
    print("=" * 60)

    # Démarrage de la mesure de temps
    start_time = time.time()

    # Initialisation du moteur de recherche pondéré
    ranker = RankedRetrieval(index, cache_dir="data/norm_cache")

    # Prétraitement des termes de la requête
    query_terms = ranker.process_query_terms(query_request)

    # Exemple: récupération du poids d'un terme dans un document spécifique
    term = query_terms[1]  # terme après traitement
    ranking_weight = ranker.get_term_weight(term, "23724", weighting_scheme)

    # Calcul du RSV du document #23724
    doc_score = 0.0
    for term in query_terms:
        term_weight = ranker.get_term_weight(term, "23724", weighting_scheme)
        doc_score += term_weight

    # Recherche du Top-10 pour affichage
    top_docs = ranker.search_query(query_request, weighting_scheme, top_k=10)

    # Temps total écoulé
    weighting_time = time.time() - start_time

    print(f"- Temps total de pondération: {weighting_time:.2f} secondes")
    print(f"- Poids du terme 'ranking' dans le document #23724: {ranking_weight:.6f}")
    print(f"- RSV du document #23724: {doc_score:.6f}")

    print("- TOP-10 DOCUMENTS:")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")

    # Recherche étendue pour génération du fichier run
    top_docs = ranker.search_query(query_request, weighting_scheme, top_k=1500)

    # Construction du nom du fichier run
    file_name = f"AlphaAnaClement_{run_id}_test_{weighting_scheme}_article"
    file_name += "_stop671" if index.stop_word_active else "_nostop"
    file_name += "_porter" if index.stemmer_active else "_nostem"
    if weighting_scheme == "bm25":
        file_name += "_k1.2_b0.75"
    file_name += ".txt"

    # Création du fichier si nécessaire
    full_path = "runs/" + file_name
    if not os.path.exists(full_path):
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("")

    # Écriture du run au format TREC
    for i, (doc_id, score) in enumerate(top_docs, 1):
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(f"{query_id} Q0 {doc_id} {i} {score} AlphaAnaClement /article[1]\n")

    return weighting_time, ranking_weight, doc_score, top_docs


def main():
    """Fonction principale orchestrant la construction des index et les expériences."""

    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion",
    }

    # Construction des différents index selon les options
    index_no_stop_no_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem",
                                               use_stop_words=False, use_stemmer=False)
    index_stop_no_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem",
                                            use_stop_words=True, use_stemmer=False)
    index_stop_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem",
                                         use_stop_words=True, use_stemmer=True)
    index_no_stop_stem = compute_statistics(1, "Text_Only_Ascii_Coll_NoSem",
                                            use_stop_words=False, use_stemmer=True)

    # Première série de runs avec LTN
    run_id = len([f for f in os.listdir("runs")
                  if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltn", query_request, run_id)

    # Série LTH
    run_id = len([f for f in os.listdir("runs")
                  if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "ltc", query_request, run_id)

    # Série BM25
    run_id = len([f for f in os.listdir("runs")
                  if os.path.isfile(os.path.join("runs", f))])
    for query_id, query_request in queries.items():
        run_weighting_experiment(index_no_stop_no_stem, query_id, "bm25", query_request, run_id)

    # Séries complémentaires avec combinaisons stopwords/stemming
    algorithms = ["ltn", "ltc", "bm25"]
    indexers = [index_stop_no_stem, index_stop_stem, index_no_stop_stem]

    for index in indexers:
        for algorithm in algorithms:
            run_id = len([f for f in os.listdir("runs")
                          if os.path.isfile(os.path.join("runs", f))])
            for query_id, query_request in queries.items():
                run_weighting_experiment(index, query_id, algorithm, query_request, run_id)


if __name__ == "__main__":
    main()
