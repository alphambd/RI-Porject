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
from field_weighted_index import generate_field_weighted_run_simple, generate_field_weighted_run
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
"""
def exercice5_optimized():
    #Exercice 5 optimisé avec champs multiples et regroupés
    print_exercise_header(5, "BM25Fw optimisé - Champs multiples")
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop'
    }
    
    # Configuration flexible des champs
    fields_config = {
        'title': ['title'],      # Champ unique
        'body': ['bdy'],         # Champ unique  
        'sections': ['sec'],     # TOUTES les sections regroupées en un seul champ
        #'paragraphs': ['p'],     # TOUS les paragraphes regroupés en un seul champ
        # Vous pouvez ajouter d'autres champs :
        # 'captions': ['caption'],
        # 'links': ['link']
    }
    
    # Poids à optimiser (vous pouvez tester différentes valeurs)
    field_weights = {
        'title': 1.0,      # Titre très important
        'body': 1.0,       # Corps du document
        'sections': 1.0   # Sections regroupées
        #'paragraphs': 1.0  # Paragraphes regroupés
    }
    
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    
    generator = INEXRunGenerator()
    
    filename = generate_field_weighted_run_cached(
        generator=generator,
        run_id="test5_optimized",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    return filename
"""

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

def exercice5_run():
    """Génère un run BM25Fw pondéré par champs"""
    print_exercise_header(5, "BM25Fw - Pondération par champs")
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        #'p': ['p'],
    }
    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        #'p': 1.0
    }
    run_params = {
        'k1': 1.2,
        'b': 0.6,
        'max_files': None
    }
    filename = generate_field_weighted_run(
        run_id="5",
        run_type="bm25fw",
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

def exercice6_run():
    """Génère un run BM25Fr pondéré par champs"""
    print_exercise_header(6, "BM25Fr - Pondération par champs")
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }
    fields_config = {
        'title': ['title'],
        'bdy': ['bdy'],
        'sec': ['sec'],
        #'p': ['p'],
    }
    field_weights = {
        'title': 1.0,
        'bdy': 1.0,
        'sec': 1.0,
        #'p': 1.0
    }
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    filename = generate_field_weighted_run(
        run_id="6",
        run_type="bm25fr",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    return filename






