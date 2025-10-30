import time
import os
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

def compute_statistics(exercise_num, use_stop_words=False, use_stemmer=False):
    """Fonction générique pour les exercices de statistiques"""
    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {'AVEC' if use_stop_words else 'SANS'} STOP-WORDS ET STEMMING")
    print("=" * 60)
    
    index = WeightedInvertedIndex()
    index.stop_word_active = use_stop_words
    index.stemmer_active = use_stemmer
    
    if use_stop_words:
        index.load_stop_words()
    
    indexing_time = index.build_index("data/practice_03_data/Text_Only_Ascii_Coll_NoSem", False)
    
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

def run_weighting_experiment(index, exercise_name, weighting_scheme):
    """Exécute les exercices 3, 4, 5"""
    print(f"\n" + "=" * 60)
    print(f"{exercise_name}: {weighting_scheme.upper()} WEIGHTING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialisation du moteur de pondération
    ranker = RankedRetrieval(index)
    
    weighting_time = time.time() - start_time
    print(f"- Temps de pondération: {weighting_time:.2f} secondes")
    
    # Requête pour tous les exercices
    query = "web ranking scoring algorithm"
    query_terms = ranker.process_query_terms(query)
    
    # Poids du terme "ranking" dans le document #23724
    ranking_process = query_terms[1] # récupérer le terme après traitement
    try:
        ranking_weight = ranker.get_term_weight(ranking_process, "23724", weighting_scheme)
        print(f"- Poids de 'ranking' dans doc #23724: {ranking_weight:.6f}")
    except KeyError:
        print(f"Document #23724 ou terme 'ranking' non trouvé...")
        ranking_weight = 0.0
    
    # Score RSV du document #23724
    doc_score = 0.0
    try:
        for term in query_terms:
            term_weight = ranker.get_term_weight(term, "23724", weighting_scheme)
            doc_score += term_weight
        print(f"- RSV du document #23724: {doc_score:.6f}")
    except KeyError:
        print(f"Impossible de calculer RSV pour doc #23724...")
        doc_score = 0.0
    
    # Top-10 des documents
    top_docs = ranker.search_query(query, weighting_scheme, top_k=10)
    
    print(f"- TOP-10 DOCUMENTS:")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f" {i:2d}. Doc {doc_id}: {score:.6f}")
    
    return weighting_time, ranking_weight, doc_score, top_docs


def main():
    """Fonction principale"""    
    # Exercice 1: sans traitement de tokens
    index1 = compute_statistics(1, use_stop_words=False, use_stemmer=False)
    
    # Exercice 2: avec traitement  
    index2 = compute_statistics(2, use_stop_words=True, use_stemmer=True)
    
    # Utiliser l'index avec traitement pour les exercices 3-5
    index = index2
    
    # Exercice 3: SMART ltn
    run_weighting_experiment(index, "EXERCICE 3", "ltn")
    
    # Exercice 4: SMART ltc  
    run_weighting_experiment(index, "EXERCICE 4", "ltc")
    
    # Exercice 5: BM25
    run_weighting_experiment(index, "EXERCICE 5", "bm25")
    

if __name__ == "__main__":
    main()