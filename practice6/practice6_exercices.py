import os
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

# ==================== EXERCICE 1 ====================

def exercice1_test():
    """Exercice 1: 12 runs"""
    print_exercise_header(1, "XML documents test runs (12 combinaisons)")
    
    generator = INEXRunGenerator()
    
    # Toutes les combinaisons
    
    combinations = [
        # (weighting, stop, stemmer, run_id)
        ("bm25", "nostop", "nostem", "test2"),
        ("bm25", "nostop", "porter", "test2"),
        ("bm25", "stop671", "nostem", "test2"),
        ("bm25", "stop671", "porter", "test2")
    ]
    
    results = []
    
    for i, (weighting, stop, stemmer, run_id) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {i}/12: {weighting.upper()}, stop={stop}, stemmer={stemmer}")
        print('='*60)
        
        # Configuration simple
        config = {
            'tokenization': 'basic',
            'stemmer': stemmer,
            'stop_words': stop,
            #'use_lxml': True
        }
        
        # Appel direct à la fonction
        filename = generator.generate_article_run(
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_id=run_id,
            weighting_scheme=weighting
        )
    
    print(f"\nEXERCICE 1 TERMINÉ: 12 runs générées.")
    print("\n" + "="*70)
    return results


def exercice1_bm25_tuning():
    """
    Exercice 1 — BM25 parameter tuning (k1, b)
    """

    #k1_values = [0.6, 0.9, 1.2, 1.5, 2.0]
    k1 = 1.2
    b_values = [0.4, 0.5, 0.6, 0.65, 0.7, 0.8]

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    runs = []

    #for k1 in k1_values:
    for b in b_values:

        run_id = f"p6e1_bm25_tuning"

        print("\n" + "=" * 70)
        print(f"BM25 TUNING — k1={k1}, b={b}")
        print("=" * 70)

        generator = INEXRunGenerator()
        filename = generator.generate_article_run(
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_id=run_id,
            weighting_scheme='bm25',
            k1=k1,
            b=b
        )

        runs.append((k1, b, filename))

    return runs


# ==================== EXERCICE 2 ====================

# ==================== EXERCICE 3 ====================

def exercice3():
    run_gen = INEXRunGenerator(team_name="AlphaAnaClement")
    
    # Configuration optimisée
    fetch_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop'
    }
    
    browse_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'target_tags': ['bdy', 'sec', 'p']
    }
    
    run_params = {
        'top_articles': 1500,
        'max_elements': 1500,
        'max_elements_per_article': 2,  
        'weighting_scheme': 'ltn',
        'avoid_overlaps': True,
        'min_element_score': 0.1,  
        'fallback_to_article': True
    }

    bonus_tags = {
        'bdy': 1.0,
        'sec': 1.1,
        'p':   1.2
    }
    
    filename = run_gen.generate_fetch_browse(
        run_id="_4_testXML",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        fetch_config=fetch_config,
        browse_config=browse_config,
        run_params=run_params,
        bonus_tags=bonus_tags
    )
    
    return filename
    
def exercice3_test_stop_stem():
    """
    Exercice 3 - Phase 2: Test des stop-words et stemmer
    Pondération + Stop-words + Stemmer pour les éléments XML
    Granularité fixe: [bdy, sec, p]
    """

    run_params = {
        'top_articles': 1500,
        'max_elements': 1500,
        'max_elements_per_article': 5,  
        'weighting_scheme': 'ltn',
        'avoid_overlaps': True,
        'min_element_score': 0.1,  
        'fallback_to_article': True
    }
    
    combinations = [
        # (weighting, stop, stemmer, run_id)
        ("ltn", "nostop", "nostem", "testXMLel"),
        ("ltn", "nostop", "porter", "testXMLel"),
        ("ltn", "stop671", "nostem", "testXMLel"),
        ("ltn", "stop671", "porter", "testXMLel"),
    ]

    generator = INEXRunGenerator()

    for i, (weighting, stop, stemmer, run_id) in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"CONFIGURATION {i}/4: {weighting.upper()}, stop={stop}, stemmer={stemmer}")
        print('='*60)
        
        # Configuration simple
        config = {
            'tokenization': 'basic',
            'stemmer': stemmer,
            'stop_words': stop,
            #'use_lxml': True
        }
        
        # Appel direct à la fonction
        filename = generator.generate_fetch_browse(
            run_id=run_id+f"_{i+1}",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            fetch_config=config,
            browse_config=config,
            run_params=run_params
        )


