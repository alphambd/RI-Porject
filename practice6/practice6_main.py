import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import time
from typing import Dict, List, Optional
from xml_run_manager import INEXRunGenerator
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from field_weighted_index import generate_field_weighted_run #, generate_field_weighted_run_cached

# ==================== CONSTANTES ET CONFIGURATIONS ====================

TEAM_NAME = "AlphaAnaClement"
XML_DIR = "data/Practice_05_data/XML-Coll-withSem"

# Requêtes INEX standard
INEX_QUERIES = {
    2009011: "olive oil health benefit",
    2009036: "notting hill film actors", 
    2009067: "probabilistic models in information retrieval",
    2009073: "web link network analysis",
    2009074: "web ranking scoring algorithm",
    2009078: "supervised machine learning algorithm",
    2009085: "operating system mutual exclusion"
}

# Paramètres par défaut
TARGET_DOC_ID = "23724"
TARGET_TERM = "ranking"
TEST_QUERY = "web ranking scoring algorithm"

# ==================== FONCTIONS UTILITAIRES ====================

def print_exercise_header(exercise_num: int, title: str):
    """Affiche l'en-tête d'un exercice"""
    print("\n" + "=" * 70)
    print(f"EXERCICE {exercise_num}: {title}")
    print("=" * 70)

def compute_statistics_for_config(index_data: Dict, weighting_scheme: str = "ltn",
                                k1: float = 1.2, b: float = 0.75) -> Dict:
    """
    Calcule les statistiques pour une configuration donnée
    """
    index = index_data['index']
    indexing_time = index_data['indexing_time']
    
    # Initialiser le ranker et mesurer le temps de pondération
    weighting_start = time.time()
    ranker = RankedRetrieval(index)
    
    # Calculer les poids spécifiques
    query_terms = ranker.process_query_terms(TEST_QUERY)
    target_terms = ranker.process_query_terms(TARGET_TERM)
    
    target_weight = 0.0
    if target_terms:
        target_weight = ranker.get_term_weight(
            target_terms[0], TARGET_DOC_ID, weighting_scheme, k1, b
        )
    
    doc_score = sum(
        ranker.get_term_weight(t, TARGET_DOC_ID, weighting_scheme, k1, b)
        for t in query_terms
    )
    
    # Recherche top-10
    top_docs = ranker.search_query(TEST_QUERY, weighting_scheme, top_k=10, k1=k1, b=b)
    weighting_time = time.time() - weighting_start
    
    # Récupérer les statistiques de base
    stats = index.get_collection_statistics(indexing_time)
    
    # Calculer le temps total
    total_time = indexing_time + weighting_time
    
    return {
        'index': index,
        'ranker': ranker,
        'stats': stats,
        'indexing_time': indexing_time,
        'weighting_time': weighting_time,
        'total_time': total_time,
        'target_weight': target_weight,
        'doc_score': doc_score,
        'top_docs': top_docs,
        'weighting_scheme': weighting_scheme,
        'k1': k1,
        'b': b
    }

def display_statistics(stats_data: Dict, config_desc: str):
    """Affiche les statistiques formatées"""
    print(f"\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Configuration: {config_desc}")
    print(f"- Temps total d'indexation + pondération: {stats_data['total_time']:.2f} secondes")
    print(f" * Temps d'indexation seul: {stats_data['indexing_time']:.2f} secondes")
    print(f" * Temps de pondération: {stats_data['weighting_time']:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats_data['stats']['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats_data['stats']['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats_data['stats']['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats_data['stats']['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats_data['stats']['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats_data['stats']['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats_data['stats']['avg_term_length']:.2f} caractères")
    
    print(f"- Poids du terme '{TARGET_TERM}' dans le document #{TARGET_DOC_ID}: {stats_data['target_weight']:.6f}")
    print(f"- RSV du document #{TARGET_DOC_ID} pour '{TEST_QUERY}': {stats_data['doc_score']:.6f}")
    
    # Afficher le nombre de documents pertinents potentiels
    relevant_docs = stats_data['ranker'].search_query(
        TEST_QUERY, 
        stats_data['weighting_scheme'], 
        top_k=None,
        k1=stats_data['k1'],
        b=stats_data['b']
    )
    print(f"- Documents pertinents potentiels: {len(relevant_docs)}")
    
    print(f"- TOP-10 DOCUMENTS pour '{TEST_QUERY}':")
    for i, (doc_id, score) in enumerate(stats_data['top_docs'], 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")


def exercice4():
    """
    Exercice 4 — Articles run exploiting links (BM25 + PageRank)
    """

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    generator = INEXRunGenerator()
    filename = generator.generate_article_run_with_pagerank(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_id="ex4_pagerank_baseline",
        top_k=1500,
        alpha=0.9,      # BM25 dominant
        k1=1.2,
        b=0.75
    )

    return filename

def exercice4_tuning():
    """
    Exercice 4 — Tuning BM25 / PageRank interpolation
    """

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    alpha_values = [0.7, 0.8, 0.85, 0.9, 0.95]
    results = []
    generator = INEXRunGenerator()

    for alpha in alpha_values:
        print("\n" + "=" * 60)
        print(f"EX4 TUNING — alpha(BM25) = {alpha}")
        print("=" * 60)

        run_id = f"ex4_pagerank_a{alpha}"
        
        filename = generator.generate_article_run_with_pagerank(
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_id=run_id,
            top_k=1500,
            alpha=alpha,
            k1=1.2,
            b=0.75
        )

        results.append((alpha, filename))

    return results
    
def exercice5():
    """
    Exercice 5 — Baseline BM25F (content + anchors)
    """

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    generator = INEXRunGenerator()
    filename = generator.generate_article_run_with_anchors(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_id="5_anchor",
        top_k=1500,
        alpha_content=1.0,
        alpha_anchor=0.7,   # valeur raisonnable par défaut
        k1=1.2,
        b=0.6
    )

    return filename

def exercice5_tuning():
    """
    Exercice 5 — Tuning BM25F (alpha_content / alpha_anchor)
    """

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    alpha_content_values = [0.8, 1.0, 1.2]
    alpha_anchor_values = [0.2, 0.5, 0.8, 1.0]

    results = []
    generator = INEXRunGenerator()

    for a_content in alpha_content_values:
        for a_anchor in alpha_anchor_values:
            print("\n" + "=" * 60)
            print(
                f"EX5 TUNING — a_content={a_content}, "
                f"a_anchor={a_anchor}"
            )
            print("=" * 60)

            run_id = (
                f"ex5_anchor_"
                f"ac{a_content}_aa{a_anchor}"
            )

            filename = generator.generate_article_run_with_anchors(
                xml_dir=XML_DIR,
                queries=INEX_QUERIES,
                config=config,
                run_id=run_id,
                top_k=1500,
                alpha_content=a_content,
                alpha_anchor=a_anchor,
                k1=1.2,
                b=0.75
            )

            results.append((a_content, a_anchor, filename))

    return results



def display_links_stats(stats: Dict):
    XML_DIR = "data/Practice_05_data/XML-Coll-withSem"

    print("\n[STATISTIQUES LIENS INEX]")
    print(f"Nombre d'articles: {stats['num_articles']}")
    print(f"Total des liens: {stats['total_links']}")   
    print(f"Liens article->article: {stats['article_to_article_links']}")
    print(f"Liens externes: {stats['external_links']}")
    print(f"Références internes: {stats['internal_refs']}")
    print(f"Erreurs de parsing: {stats['parse_errors']}")
    print(f"Degré de sortie moyen: {stats['avg_out_degree']:.2f}")
    print(f"Degré de sortie max: {stats['max_out_degree']}")
    print(f"Degré de sortie min: {stats['min_out_degree']}")


def main():
    XML_DIR = "data/Practice_05_data/XML-Coll-withSem"

    #exercice4()
    #exercice5()
    
    print("\n=== FIN DES EXÉCUTIONS ===\n")

if __name__ == "__main__":
    main()