import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import time
from typing import Dict, List, Optional
from p5_bm25_exercices import exercice5_bm25fw, exercice5_bm25fw_test, exercice6_bm25fr
from xml_run_manager import INEXRunGenerator
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from utiles import clean_runs_directory, print_exercise_header, compute_statistics, compute_statistics_for_config, display_statistics, create_index_with_config, create_element_index_with_config, extract_inex_link_graph

from field_weighted_index import generate_field_weighted_run #, generate_field_weighted_run_cached
from practice6_exercices import exercice1_test, exercice1_bm25_tuning

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
        run_id="ex4_PR",
        top_k=1500,
        pagerank_alpha=0.9,      # BM25 dominant
        k1=1.2,
        b=0.65
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

        run_id = f"ex4_PR_a{alpha}"
        
        filename = generator.generate_article_run_with_pagerank(
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_id=run_id,
            top_k=1500,
            pagerank_alpha=alpha,
            k1=1.2,
            b=0.65
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
        b=0.65
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
                b=0.65
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

    clean_runs_directory()

    #exercice1_test()
    #exercice1_bm25_tuning()

    # Exercice 3 tests
    #exercice5_bm25fw()
    #exercice6_bm25fr()
    #exercice5_bm25fw_test()

    #gragraph, stats = extract_inex_link_graph()
    #display_links_stats(stats)

    #exercice4()
    #exercice4_tuning()
    #exercice5()
    exercice5_tuning()


    
    print("\n=== FIN DES EXÉCUTIONS ===\n")

if __name__ == "__main__":
    main()