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

# ==================== EXERCICE 5 ====================

def exercice5():
    """Exercice 5: BM25Fw - Pondération par champs (combinaison tardive)"""
    print_exercise_header(5, "BM25Fw - Late combination of fields")
    
    # Configuration simple
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    # Champs avec regroupement automatique
    fields_config = {
        'title': ['title'],      # Titre unique
        'body': ['bdy'],         # Corps unique
        'sections': ['sec'],     # TOUTES les sections regroupées
        'paragraphs': ['p'],     # TOUS les paragraphes regroupés
    }
    
    # Poids à tester (vous pouvez modifier ces valeurs)
    field_weights = {
        'title': 1.0,    # Titre très important
        'body': 1.0,     # Corps important
        'sections': 1.0, # Sections moyennes
        'paragraphs': 1.0# Paragraphes basiques
    }
    
    # Paramètres BM25
    run_params = {
        'k1': 1.2,
        'b': 0.6,
        'max_files': None  # Tous les fichiers
    }
    
    # Générer le run
    filename = generate_field_weighted_run_simple(
        run_id="5",
        run_type=f"bm25fw_{config['stop_words']}_{config['stemmer']}",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    print(f"\n Exercice 5 terminé")
    print(f" Fichier généré: {filename}")
    
    return filename


def exercice5_rest():
    """Exercice 5 avec option 'rest'"""
    print_exercise_header(5, "BM25Fw avec champ 'rest'")
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    # Configuration de base
    base_fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        'p': ['p']
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    
    # Tester avec et sans "rest"
    results = []
    
    print("\n1. Test SANS champ 'rest':")
    filename_no_rest = generate_field_weighted_run_with_rest(
        run_id="5_no_rest",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=base_fields_config,
        field_weights={'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0},
        include_rest=False
    )
    results.append(('no_rest', filename_no_rest))
    
    print("\n2. Test AVEC champ 'rest' (poids=1.0):")
    filename_with_rest = generate_field_weighted_run_with_rest(
        run_id="5_with_rest",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=base_fields_config,
        field_weights={'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0, 'rest': 1.0},
        include_rest=True,
        rest_weight=1.0
    )
    results.append(('with_rest_1.0', filename_with_rest))
    
    # Tester différents poids pour "rest"
    for rest_weight in [0.5, 1.5]:
        print(f"\n3. Test AVEC champ 'rest' (poids={rest_weight}):")
        filename = generate_field_weighted_run_with_rest(
            run_id=f"5_rest_{rest_weight}",
            run_type="bm25fw",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params=run_params,
            fields_config=base_fields_config,
            field_weights={'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0, 'rest': rest_weight},
            include_rest=True,
            rest_weight=rest_weight
        )
        results.append((f'with_rest_{rest_weight}', filename))
    
    return results

def exercice6_rest():
    """Exercice 6 avec option 'rest'"""
    print_exercise_header(6, "BM25Fr avec champ 'rest'")
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    base_fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        'p': ['p']
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    
    results = []
    
    # Tester différentes configurations
    test_configs = [
        {'name': 'no_rest', 'include_rest': False, 'weights': {'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0}},
        {'name': 'rest_0.5', 'include_rest': True, 'rest_weight': 0.5, 'weights': {'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0, 'rest': 0.5}},
        {'name': 'rest_1.0', 'include_rest': True, 'rest_weight': 1.0, 'weights': {'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0, 'rest': 1.0}},
        {'name': 'rest_1.5', 'include_rest': True, 'rest_weight': 1.5, 'weights': {'title': 3.0, 'bdy': 2.0, 'sec': 1.5, 'p': 1.0, 'rest': 1.5}},
    ]
    
    for test in test_configs:
        print(f"\nTest: {test['name']}")
        
        filename = generate_field_weighted_run_with_rest(
            run_id=f"6_{test['name']}",
            run_type="bm25fr",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params=run_params,
            fields_config=base_fields_config,
            field_weights=test['weights'],
            include_rest=test['include_rest'],
            rest_weight=test.get('rest_weight', 1.0)
        )
        
        results.append((test['name'], filename))
    
    return results

def test_complete():
    """Test complet de toutes les configurations"""
    print("="*70)
    print("TEST COMPLET AVEC/SANS 'rest'")
    print("="*70)
    
    print("\n=== EXERCICE 5 (BM25Fw) ===")
    results_5 = exercice5_rest()
    
    print("\n=== EXERCICE 6 (BM25Fr) ===")
    results_6 = exercice6_rest()
    
    print("\n" + "="*70)
    print("RÉSUMÉ DES RUNS GÉNÉRÉS")
    print("="*70)
    
    print("\nBM25Fw (Exercice 5):")
    for name, filename in results_5:
        print(f"  {name}: {os.path.basename(filename)}")
    
    print("\nBM25Fr (Exercice 6):")
    for name, filename in results_6:
        print(f"  {name}: {os.path.basename(filename)}")
    
    return results_5, results_6

def exercice5_simple_with_rest():
    """Version simple avec 'rest'"""
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        'rest': ['__REST__']  # Champ pour tout le reste
    }
    
    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        'rest': 1.0  # Poids égal à 1 pour commencer
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.75
    }
    
    filename = generate_field_weighted_run_simple(
        run_id="5_simple_rest",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    return filename

def exercice6_simple_with_rest():
    """Version simple avec 'rest'"""
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        'rest': ['__REST__']
    }
    
    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        'rest': 1.0
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.75
    }
    
    filename = generate_field_weighted_run_simple(
        run_id="6_simple_rest",
        run_type="bm25fr",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    return filename

# ==================== EXERCICE 6 ====================

def exercice6():
    """Exercice 6: BM25Fr - Pondération par champs (combinaison précoce)"""
    print_exercise_header(6, "BM25Fr - Early combination of fields")
    
    # Même configuration que l'exercice 5
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    
    # Mêmes champs pour comparaison équitable
    fields_config = {
        'title': ['title'],
        'body': ['bdy'],
        'sections': ['sec'],
        'paragraphs': ['p'],
    }
    
    # Mêmes poids ou différents pour tester
    field_weights = {
        'title': 1.0,
        'body': 1.0,
        'sections': 1.0,
        'paragraphs': 1.0
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.6,
        'max_files': None
    }
    
    # Seule différence: run_type="bm25fr"
    filename = generate_field_weighted_run_simple(
        run_id="6",
        run_type=f"bm25fr_{config['stop_words']}_{config['stemmer']}",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    print(f"\n Exercice 6 terminé")
    print(f" Fichier généré: {filename}")
    
    return filename

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
        'sec': ['sec'],
        'p': ['p'],
        'rest': ['__REST__']
    }

    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        'p': 1.0,
        'rest': 1.0
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
        'sec': ['sec'],
        'p': ['p'],
        'rest': ['__REST__']
    }

    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        'p': 1.0,
        'rest': 1.0
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






