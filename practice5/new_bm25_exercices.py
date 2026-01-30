import math
import os
import time
from typing import Dict, Optional
from typing import List, Tuple
from xml_run_manager import INEXRunGenerator
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
#from field_weighted_index import generate_field_weighted_run_cached
from field_weighted_index import FieldWeightedIndex
from field_weighted_index import generate_field_weighted_run_simple, generate_field_weighted_run_with_rest
from new_field_weighted_index import generate_field_weighted_run
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

def run_alpha_sweep_fr(
    base_fields_config: Dict[str, List[str]],
    base_weights: Dict[str, float],
    field_to_add: str,
    field_tags: List[str],
    alpha_values: List[float],
    run_prefix: str,
    xml_dir: str,
    queries: Dict[int, str],
    config: Dict,
    run_params: Dict) -> List[Tuple[float, str]]:
    """
    Ajoute un champ et teste plusieurs valeurs de alpha (BM25FR).
    """
    results = []

    for alpha in alpha_values:
        print("\n" + "="*60)
        print(f"TEST FR — Champ '{field_to_add}', alpha={alpha}")
        print("="*60)

        # Copier les configs de base
        fields_config = dict(base_fields_config)
        field_weights = dict(base_weights)

        # Ajouter le champ testé
        fields_config[field_to_add] = field_tags
        field_weights[field_to_add] = alpha

        #run_id = f"{run_prefix}_{field_to_add}_a{alpha}"
        run_id = f"{run_prefix}_{field_to_add}_a.p{alpha}_a.t2.0_a.bdy1.0_a.sec0.3"
        run_type = "bm25fr"

        filename = generate_field_weighted_run(
            run_id=run_id,
            run_type=run_type,
            xml_dir=xml_dir,
            queries=queries,
            config=config,
            run_params=run_params,
            fields_config=fields_config,
            field_weights=field_weights
        )

        results.append((alpha, filename))

    return results

def run_alpha_sweep(
    method: str,  # "bm25fr" ou "bm25fw"
    base_fields_config: Dict[str, List[str]],
    base_weights: Dict[str, float],
    field_to_add: str,
    field_tags: List[str],
    alpha_values: List[float],
    run_prefix: str,
    xml_dir: str,
    queries: Dict[int, str],
    config: Dict,
    run_params: Dict
) -> List[Tuple[float, str]]:
    """
    Ajoute un champ et teste plusieurs valeurs de alpha
    pour BM25FR ou BM25FW.
    """
    assert method in {"bm25fr", "bm25fw"}

    results = []

    for alpha in alpha_values:
        print("\n" + "="*60)
        print(f"TEST {method.upper()} — Champ '{field_to_add}', alpha={alpha}")
        print("="*60)

        # Copier les configs de base
        fields_config = dict(base_fields_config)
        field_weights = dict(base_weights)

        # Ajouter le champ testé
        fields_config[field_to_add] = field_tags
        field_weights[field_to_add] = alpha

        run_id = f"{run_prefix}_{method}_{field_to_add}_a{alpha}"

        filename = generate_field_weighted_run(
            run_id=run_id,
            run_type=method,
            xml_dir=xml_dir,
            queries=queries,
            config=config,
            run_params=run_params,
            fields_config=fields_config,
            field_weights=field_weights
        )

        results.append((alpha, filename))

    return results

# ==================== EXERCICE 5 ====================
#def exo6_bm25fw_opti_p():


# ==================== EXERCICE 6 ====================


def exercice5_bm25fw():
    """
    Exercice 5 — BM25FW (Wilkinson, late combination)
    avgdl UNWEIGHTED
    """
    print_exercise_header(5, "BM25FW — Late Combination (Wilkinson)")

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        #'sec': ['sec'],
        #'p': ['p'],
        #'rest': ['__REST__']
    }

    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        #'sec': 1.0,
        #'p': 1.0,
        #'rest': 1.0
    }

    run_params = {
        'k1': 1.2,
        'b': 0.6,
        'max_files': None
    }

    return generate_field_weighted_run(
        run_id="5",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )

def exercice6_bm25fr_test():
    """
    Exercice 5 — BM25FR (Robertson, early combination)
    avgdl WEIGHTED
    """
    print_exercise_header(5, "BM25FR — Early Combination (Robertson)")

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    # Baseline réaliste
    base_fields = {
        'title': ['title'],
        'bdy':   ['bdy'],
        'sec':   ['sec']
    }

    base_weights = {
        'title': 2.0,
        'bdy':   1.0,
        'sec':   0.3
    }

    # Valeurs à tester pour sec
    #alpha_values = [0.5, 0.8, 1.3, 1.5, 1.8, 2.0]
    #alpha_values = [0.1, 0.2, 0.3, 0.4]
    alpha_values = [0.8, 0.9, 1.1, 1.2, 1.3]

    results_sec = run_alpha_sweep_fr(
        base_fields_config=base_fields,
        base_weights=base_weights,
        field_to_add='p',
        field_tags=['p'],
        alpha_values=alpha_values,
        run_prefix="FR_add-p",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params={
            'k1': 1.2,
            'b': 0.6
        }
    )

def exercice5_bm25fw_test():
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    fields_config = {
        'title': ['title'],
        'bdy':   ['bdy'],
        'sec':   ['sec'],
        'p':     ['p']
    }

    base_weights = {
        'title': 1.0,
        'bdy':   2.5,
        'sec':   2.5,
        'p':     3.5
    }

    #alpha_p_values = [0.8, 0.9, 1.0, 1.1, 1.2]
    alpha_title_values = [1.0, 1.5, 2.0, 2.5]

    for alpha in alpha_title_values:
        field_weights = dict(base_weights)
        field_weights['title'] = alpha
        #run_id = f"FW_title_a.{alpha}_a.t2.0_a._a.sec0.3_a.p0.2"
        run_id = f"FW_title_a.{alpha}_a.bdy2.5_a.sec2.5_a.p3.5"
        generate_field_weighted_run(
            run_id=run_id,
            run_type="bm25fw",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params={
                'k1': 1.2,
                'b': 0.6
            },
            fields_config=fields_config,
            field_weights=field_weights
        )

def exercice6_bm25fr():
    """
    Exercice 6 — BM25FR (Robertson, early combination)
    avgdl WEIGHTED
    """
    print_exercise_header(6, "BM25FR — Early Combination (Robertson)")

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        #'sec': ['sec'],
        #'p': ['p'],
        #'rest': ['__REST__']
    }

    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        #'sec': 1.0,
        #'p': 1.0,
        #'rest': 1.0
    }

    run_params = {
        'k1': 1.2,
        'b': 0.6,
        'max_files': None
    }

    return generate_field_weighted_run(
        run_id="6",
        run_type="bm25fr",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )        

def exo6_bm25fr_opti_title():
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    fields_config = {
        'title': ['title'],
        'bdy':   ['bdy'],
        'sec':   ['sec'],
        'p':     ['p']
    }

    base_weights = {
        'title': 1.0,
        'bdy':   1.0,
        'sec':   1.0,
        'p':     1.0
    }

    alpha_title_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for alpha in alpha_title_values:
        field_weights = dict(base_weights)
        field_weights['title'] = alpha

        run_id = f"FR_title-only_a{alpha:.1f}"

        generate_field_weighted_run(
            run_id=run_id,
            run_type="bm25fr",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params={
                'k1': 1.2,
                'b': 0.6
            },
            fields_config=fields_config,
            field_weights=field_weights
        )

def exo6_bm25fr_opti_bdy():
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    fields_config = {
        'title': ['title'],
        'bdy':   ['bdy'],
        'sec':   ['sec'],
        'p':     ['p']
    }

    base_weights = {
        'title': 2.0,
        'bdy':   1.0,
        'sec':   0.3,
        'p':     0.2
    }

    #alpha_title_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    alpha_bdy_values = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    
    for alpha in alpha_bdy_values:
        field_weights = dict(base_weights)
        field_weights['bdy'] = alpha
        #run_id = f"FR_bdy-only_a{alpha:.1f}"
        run_id = f"FR_bdy_a.{alpha}_a.t2.0_a._a.sec0.3_a.p0.2"

        generate_field_weighted_run(
            run_id=run_id,
            run_type="bm25fr",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params={
                'k1': 1.2,
                'b': 0.6
            },
            fields_config=fields_config,
            field_weights=field_weights
        )


def exo6_bm25fr_opti_sec():
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    # Baseline réaliste
    base_fields = {
        'title': ['title'],
        'bdy':   ['bdy'],
        'p':     ['p']
    }

    base_weights = {
        'title': 2.0,
        'bdy':   1.0,
        'p':     1.0
    }

    # Valeurs à tester pour sec
    #alpha_values = [0.5, 0.8, 1.3, 1.5, 1.8, 2.0]
    alpha_values = [0.1, 0.2, 0.3, 0.4]

    results_sec = run_alpha_sweep_fr(
        base_fields_config=base_fields,
        base_weights=base_weights,
        field_to_add='sec',
        field_tags=['sec'],
        alpha_values=alpha_values,
        run_prefix="FR_add-sec",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params={
            'k1': 1.2,
            'b': 0.6
        }
    )


def exo6_bm25fr_opti_p():
    # Baseline FR
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    base_fields = {
        'title': ['title'],
        'bdy':   ['bdy'],
        #'sec':   ['sec']
    }

    base_weights = {
        'title': 1.0,
        'bdy':   1.0,
        #'sec':   1.0
    }

    #alpha_values = [0.2, 0.5, 1.0]
    alpha_values = [0.3, 0.5, 1.0, 1.5, 1.8]

    results_p = run_alpha_sweep_fr(
        base_fields_config=base_fields,
        base_weights=base_weights,
        field_to_add='p',
        field_tags=['p'],
        alpha_values=alpha_values,
        run_prefix="FR_step2",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params={
            'k1': 1.2,
            'b': 0.6
        }
    )




