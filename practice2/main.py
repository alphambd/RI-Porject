import os
import matplotlib.pyplot as plt
import time
import gzip
import json
from stats_analyzer import StatsAnalyzer

def main():
    """Fonction principale conforme à l'énoncé"""
    # Initialisation
    os.makedirs('graphs', exist_ok=True)
    analyzer = StatsAnalyzer(data_path="practice2_data")

    # ===========================================
    # EXERCICE 1: PERFORMANCE D'INDEXATION
    # ===========================================
    print("=" * 60)
    print("EXERCICE 1: PERFORMANCE D'INDEXATION")
    print("=" * 60)
    
    # Indexation de toutes les collections (baseline)
    analyzer.run_indexation_experiment('base', use_all_files=True)
    base_results = analyzer.all_results['base']
    
    # Données pour les graphiques
    sizes = [r['total_tokens'] for r in base_results]
    times_base = [r['time_seconds'] for r in base_results]
    
    # Graphique du temps d'indexation (Exercice 1)
    analyzer.plot_single_metric(
        sizes, times_base, 'Base', 'bo-',
        '#mots', 'time (s)',
        '1. indexing time vs size of the coll', 'ex1_indexing_time.png'
    )
    
    # ===========================================
    # EXERCICE 2: STATISTIQUES DE LA COLLECTION
    # ===========================================
    print("\n" + "=" * 60)
    print("EXERCICE 2: STATISTIQUES DE LA COLLECTION")
    print("=" * 60)
    
    # Statistiques pour le fichier 9 (baseline)
    analyzer.compute_statistics(analyzer.all_results['base'], "Exercice 2")
    
    # Données pour les graphiques de l'exercice 2
    doclen_base = [r['statistics']['avg_document_length'] for r in base_results]
    termlen_base = [r['statistics']['avg_term_length'] for r in base_results]
    vocab_base = [r['statistics']['vocabulary_size'] for r in base_results]
    
    # Graphiques de l'exercice 2
    analyzer.plot_single_metric(
        sizes, doclen_base, 'Base', 'bo-',
        '#mots', '#terms',
        '2.1. avg doc length vs size of the coll', 'ex2_doc_length.png'
    )
    
    analyzer.plot_single_metric(
        sizes, termlen_base, 'Base', 'bo-',
        '#mots', '#chars',
        '2.2. avg terms length vs size of the coll', 'ex2_term_length.png'
    )
    
    analyzer.plot_single_metric(
        sizes, vocab_base, 'Base', 'bo-',
        '#mots', 'terms',
        '2.3. vocabulary size vs size of the coll', 'ex2_vocabulary.png'
    )
    
    # ===========================================
    # EXERCICE 3: ANALYSE AVEC STOP WORDS
    # ===========================================
    print("\n" + "=" * 60)
    print("EXERCICE 3: ANALYSE AVEC STOP WORDS")
    print("=" * 60)
    
    # Indexation avec stop words (fichier 9 seulement)
    analyzer.run_indexation_experiment('stopwords', stop_words=True, use_all_files=False)
    stop_results = analyzer.all_results['stopwords']
    
    # Statistiques pour le fichier 9 avec stop words
    analyzer.compute_statistics(analyzer.all_results['stopwords'], "Exercice 3")
    
    # Données pour les graphiques de l'exercice 3
    doclen_stop = [stop_results[0]['statistics']['avg_document_length']] if stop_results else []
    termlen_stop = [stop_results[0]['statistics']['avg_term_length']] if stop_results else []
    vocab_stop = [stop_results[0]['statistics']['vocabulary_size']] if stop_results else []
    
    # Graphiques de l'exercice 3 (superposés aux graphiques de l'exercice 2)
    analyzer.plot_comparison_metric(
        sizes, [doclen_base, doclen_stop], 
        ['Base', 'Stop words'], ['bo-', 'red'],  # 'ro' → 'red'
        '#mots', '#terms',
        '3.1. avg doc length vs size of the coll', 'ex3_doc_length.png'
    )

    analyzer.plot_comparison_metric(
        sizes, [termlen_base, termlen_stop], 
        ['Base', 'Stop words'], ['bo-', 'red'],  # 'ro' → 'red'
        '#mots', '#chars',
        '3.2. avg terms length vs size of the coll', 'ex3_term_length.png'
    )

    analyzer.plot_comparison_metric(
        sizes, [vocab_base, vocab_stop], 
        ['Base', 'Stop words'], ['bo-', 'red'],  # 'ro' → 'red'
        '#mots', 'terms',
        '3.3. vocabulary size vs size of the coll', 'ex3_vocabulary.png'
    )

    # ===========================================
    # EXERCICE 4: ANALYSE AVEC STOP WORDS + STEMMING
    # ===========================================
    print("\n" + "=" * 60)
    print("EXERCICE 4: ANALYSE AVEC STOP WORDS + STEMMING")
    print("=" * 60)

    # Indexation avec stop words + stemming (fichier 9 seulement)
    analyzer.run_indexation_experiment('stemming', stop_words=True, stemming=True, use_all_files=False)
    stem_results = analyzer.all_results['stemming']

    # Statistiques pour le fichier 9 avec stop words + stemming
    analyzer.compute_statistics(analyzer.all_results['stemming'], "Exercice 4")
    
    # Données pour les graphiques de l'exercice 4
    doclen_stem = [stem_results[0]['statistics']['avg_document_length']] if stem_results else []
    termlen_stem = [stem_results[0]['statistics']['avg_term_length']] if stem_results else []
    vocab_stem = [stem_results[0]['statistics']['vocabulary_size']] if stem_results else []

    # Graphiques de l'exercice 4 (superposés aux graphiques des exercices 2 et 3)
    analyzer.plot_comparison_metric(
        sizes, [doclen_base, doclen_stop, doclen_stem], 
        ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
        '#mots', '#terms',
        '4.1. avg doc length vs size of the coll', 'ex4_doc_length.png'
    )

    analyzer.plot_comparison_metric(
        sizes, [termlen_base, termlen_stop, termlen_stem], 
        ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
        '#mots', '#chars',
        '4.2. avg terms length vs size of the coll', 'ex4_term_length.png'
    )

    analyzer.plot_comparison_metric(
        sizes, [vocab_base, vocab_stop, vocab_stem], 
        ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
        '#mots', 'terms',
        '4.3. vocabulary size vs size of the coll', 'ex4_vocabulary.png'
    )
    
    # ===========================================
    # SYNTHESE DES RESULTATS
    # ===========================================
    
    print("\n" + "=" * 60)
    print("SYNTHESE DES RESULTATS")
    print("=" * 60)

    for config in ['base', 'stopwords', 'stemming']:
        if analyzer.all_results[config]:
            last_result = analyzer.all_results[config][-1]
            stats = last_result['statistics']
            
            config_name = {
                'base': 'Exo 2',
                'stopwords': 'Exo 3', 
                'stemming': 'Exo 4'
            }[config]
            
            print(f"- {config_name} : stats = ({stats['avg_document_length']:.0f} terms, "
                f"{stats['avg_term_length']:.2f} char, {stats['vocabulary_size']} distinct terms), "
                f"{last_result['time_seconds']:.2f}sec")

if __name__ == "__main__":
    main()