import os
import time
from typing import Dict, List, Optional
from xml_run_manager import INEXRunGenerator
from indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from field_weighted_index import generate_field_weighted_run_cached

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


def print_exercise_header(exercise_num: int, title: str):
    """Affiche l'en-tête d'un exercice"""
    print("\n" + "=" * 70)
    print(f"EXERCICE {exercise_num}: {title}")
    print("=" * 70)

# ==================== EXERCICE 1 ====================

def exercice1():
    """Exercice 1: Indexation XML documents (SMART ltn)"""
    print_exercise_header(1, "Indexation XML documents (SMART ltn)")
    
    # Configuration exercice 1
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        #'use_lxml': True
    }
    
    # Créer le générateur
    generator = INEXRunGenerator()
    
    print("\n[Étape 1/3] Création de l'index...")
    
    # Charger/créer l'index
    index_data = generator.create_or_load_index(
        xml_dir=XML_DIR,
        index_type='article',
        config=config
    )
    
    print("\n[Étape 2/3] Calcul des statistiques...")
    
    # Calculer les statistiques avec ltn (SMART ltn)
    stats_data = compute_statistics_for_config(
        index_data=index_data,
        weighting_scheme="ltn"
    )
    
    # Afficher les statistiques
    display_statistics(stats_data, "ltn (SMART)")
    
    print("\n[Étape 3/3] Génération du run INEX...")
    
    # Générer le run
    filename = generator.generate_article_run(
        run_id="1",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        weighting_scheme="ltn"
    )
    
    print(f"\nExercice 1 terminé")
    print(f"Run généré: {filename}")
    
    return {
        'stats_data': stats_data,
        'filename': filename
    }

# ==================== EXERCICE 2 ====================

def exercice2():
    """Exercice 2: 12 runs - VERSION ULTRA SIMPLE"""
    print_exercise_header(2, "XML documents test runs (12 combinaisons)")
    
    generator = INEXRunGenerator()
    
    # Toutes les combinaisons
    combinations = [
        # (weighting, stop, stemmer, run_id)
        ("ltn", "nostop", "nostem", "test2"),
        ("ltn", "nostop", "porter", "test2"),
        ("ltn", "stop671", "nostem", "test2"),
        ("ltn", "stop671", "porter", "test2"),
        
        ("ltc", "nostop", "nostem", "test2"),
        ("ltc", "nostop", "porter", "test2"),
        ("ltc", "stop671", "nostem", "test2"),
        ("ltc", "stop671", "porter", "test2"),
        
        ("bm25", "nostop", "nostem", "test2"),
        ("bm25", "nostop", "porter", "test2"),
        ("bm25", "stop671", "nostem", "test2"),
        ("bm25", "stop671", "porter", "test2"),
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
        
        results.append({
            'num': i,
            'weighting': weighting,
            'stop': stop,
            'stemmer': stemmer,
            'filename': filename
        })
    
    # Afficher le résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES 12 RUNS")
    print("="*70)
    
    for result in results:
        line_count = 0
        try:
            with open(result['filename'], 'r') as f:
                line_count = sum(1 for _ in f)
        except:
            pass
        
        status = " OK" if line_count == 10500 else f"  {line_count}/10500"
        
        print(f"{result['num']:2d}. {result['weighting']:4s} | "
              f"{result['stemmer']:7s} | stop={result['stop']:8s} | "
              f"{status:15s} | {os.path.basename(result['filename'])}")
    
    return results

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
        'top_articles': 1500,  # Augmenter pour plus de couverture
        'max_elements': 1500,
        'max_elements_per_article': 5,  # Prendre jusqu'à 2 éléments par article
        'weighting_scheme': 'ltn',
        'selection_strategy': 'optimal',  # Nouvelle stratégie
        'avoid_overlaps': True,
        'min_element_score': 0.00001,  # Très bas pour inclure plus d'éléments
        'fallback_to_article': True
    }
    
    filename = run_gen.generate_fetch_browse(
        run_id="testXML_optimized",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        fetch_config=fetch_config,
        browse_config=browse_config,
        run_params=run_params
    )
    
    return filename


# ==================== EXERCICE 4 ====================

def exercice4_test1():
    """
    Exercice 4 - Phase 1: Test des combinaisons de prétraitement
    Pondération + Stop-words + Stemmer pour les éléments XML
    Granularité fixe: [bdy, sec, p]
    Pas de BM25 (renvoie des score très faibles)
    """
    
    print("\n" + "="*70)
    print("EXERCICE 4 - PHASE 1: TEST PRÉTRAITEMENT")
    print("Granularité fixe: bdy, sec, p")
    print("Pondérations: ltn, ltc")
    print("Stop-words: nostop, stop671, stop319")
    print("Stemmer: nostem, porter")
    print("="*70)
    
    generator = INEXRunGenerator(team_name="AlphaAnaClement")
    
    # Paramètres à tester
    weighting_schemes = ['ltn', 'ltc']
    stop_options = ['nostop', 'stop671']
    stemmer_options = ['nostem', 'porter']
    
    # Configuration de base commune
    base_run_params = {
        'top_articles': 2000,
        'max_elements': 1500,
        'max_elements_per_article': 1,
        'selection_strategy': 'hierarchical',
        'avoid_overlaps': True,
        'fallback_to_article': True,
        'min_element_score': 0.01
    }
    
    results = []
    run_counter = 1
    total_runs = len(weighting_schemes) * len(stop_options) * len(stemmer_options)
    
    print(f"Nombre total de runs à générer: {total_runs}")
    
    for weighting in weighting_schemes:
        for stop in stop_options:
            for stemmer in stemmer_options:
                print(f"\n{'='*60}")
                print(f"RUN {run_counter}/{total_runs}")
                print(f"Configuration: {weighting.upper()}, stop={stop}, stemmer={stemmer}")
                print('='*60)
                
                # Configuration fetch (articles)
                fetch_config = {
                    'tokenization': 'basic',
                    'stemmer': stemmer,
                    'stop_words': stop
                }
                
                # Configuration browse (éléments)
                browse_config = {
                    'tokenization': 'basic',
                    'stemmer': stemmer,
                    'stop_words': stop,
                    'target_tags': ['bdy', 'sec', 'p']
                }
                
                # Paramètres de run
                run_params = base_run_params.copy()
                run_params['weighting_scheme'] = weighting
                
                # Identifiant unique du run
                run_id = f"ex4_pretrait_{run_counter}_{weighting}_{stop}_{stemmer}"
                
                # Génération du run
                print(f"Génération en cours...")
                start_time = time.time()
                
                filename = generator.generate_fetch_browse_run_optimized(
                    run_id=run_id,
                    xml_dir=XML_DIR,
                    queries=INEX_QUERIES,
                    fetch_config=fetch_config,
                    browse_config=browse_config,
                    run_params=run_params
                )
                
                generation_time = time.time() - start_time
                
                # Vérification du fichier
                line_count = 0
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                except Exception as e:
                    print(f"Erreur lecture fichier: {e}")
                
                # Stockage des résultats
                result_entry = {
                    'run_number': run_counter,
                    'weighting': weighting,
                    'stop': stop,
                    'stemmer': stemmer,
                    'filename': filename,
                    'basename': os.path.basename(filename),
                    'line_count': line_count,
                    'generation_time': generation_time,
                    'config_summary': f"{weighting}_{stop}_{stemmer}"
                }
                
                results.append(result_entry)
                
                print(f"Run généré: {os.path.basename(filename)}")
                print(f"Lignes: {line_count} (attendu: 10500)")
                print(f"Temps génération: {generation_time:.2f}s")
                
                run_counter += 1
    
    # Affichage du résumé
    print("\n" + "="*70)
    print("RÉSUMÉ PHASE 1 - PRÉTRAITEMENT")
    print("="*70)
    
    print("\nListe des runs générés:")
    print("-" * 80)
    print(f"{'No':<4} {'Pond':<6} {'Stop':<10} {'Stem':<8} {'Lignes':<8} {'Fichier'}")
    print("-" * 80)
    
    for result in results:
        status = "OK" if result['line_count'] == 10500 else "PROBLÈME"
        print(f"{result['run_number']:<4} "
              f"{result['weighting']:<6} "
              f"{result['stop']:<10} "
              f"{result['stemmer']:<8} "
              f"{result['line_count']:<8} "
              f"{result['basename'][:40]}...")
    
    return results

# ==================== EXERCICES 5 et 6 ====================

def exercice5():
    """Exercice 5: BM25Fw - Late combination"""
    print_exercise_header(5, "BM25Fw - Late combination of fields")
           
    # Configuration
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        #'use_lxml': True
    }
    
    # champs uniques et distincts
    fields_config = {
        'title': ['title'],      # Balise <title> dans header
        'body': ['bdy'],         # Balise <bdy> principale
    }
    
    # Poids: titre plus important que corps
    field_weights = {
        'title': 1.0,    
        'body': 1.0
    }
    
    # Paramètres BM25
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    
    generator = INEXRunGenerator()
    
    filename = generate_field_weighted_run_cached(
        generator=generator,
        run_id="test5",
        run_type="bm25fw",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    print(f"\nExercice 5 terminé")
    print(f"Run généré: {filename}")
    
    return filename
    
def exercice6():
    """Exercice 6: BM25Fr - Early combination"""
    print_exercise_header(6, "BM25Fr - Early combination of fields")

    # Configuration DIFFÉRENTE pour montrer la variation
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',  
        #'use_lxml': True
    }
    
    # champs uniques et distincts
    # Ajout d'un troisième champ "first_section" pour montrer la différence
    fields_config = {
        'title': ['title'],          # Titre
        'body': ['bdy'],             # Corps principal
        #'first_section': ['sec'],    # Première section
    }
    
    # Poids différents pour montrer l'impact
    field_weights = {
        'title': 1.0, #3.0,           # Très important
        'body': 1.0,            # Standard
        #'first_section': 1.5    # Un peu plus important que le corps
    }
    
    # Paramètres BM25 différents
    run_params = {
        'k1': 1.2,      
        'b': 0.75,      
        'max_files': None
    }
    
    generator = INEXRunGenerator()
    
    filename = generate_field_weighted_run_cached(
        generator=generator,
        run_id="test6",
        run_type="bm25fr",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        run_params=run_params,
        fields_config=fields_config,
        field_weights=field_weights
    )
    
    print(f"\nExercice 6 terminé")
    print(f"Run généré: {filename}")
    
    return filename
    
    
def exercice5_6_test1(algorithme="bm25fr"):
    """
    Exercice 5-6 - Phase 1: Test des combinaisons de prétraitement
    Objectif: Trouver la meilleure combinaison tokenization/stopwords/stemmer
    Paramètres fixés: k1=1.2, b=0.75, α_title=3, α_body=1
    """
    
    print("\n" + "="*70)
    print(f"EXERCICES 5-6 - PHASE 1: OPTIMISATION PRÉTRAITEMENT")
    print(f"Algorithme: {algorithme.upper()}")
    print(f"Paramètres fixes: k1=1.2, b=0.75, α_title=3, α_body=1")
    print("="*70)
        
    generator = INEXRunGenerator(team_name="AlphaAnaClement")
    
    # Paramètres à tester
    stop_options = ['nostop', 'stop671']
    stemmer_options = ['nostem', 'porter']
    
    # Configuration de base commune
    base_config = {
        'tokenization': 'basic',
        #'use_lxml': True
    }
    
    # Configuration des champs (fixe pour cette phase)
    fields_config = {
        'title': ['title'],
        'body': ['bdy']
    }
    
    # Poids fixés pour cette phase
    field_weights = {
        'title': 3.0,
        'body': 1.0
    }
    
    # Paramètres BM25 fixés
    run_params = {
        'k1': 1.2,
        'b': 0.75,
        'max_files': None
    }
    
    results = []
    run_counter = 1
    total_runs = len(stop_options) * len(stemmer_options)
    
    print(f"Nombre total de runs à générer: {total_runs}")
    
    run_id = 0
    for stop in stop_options:
        for stemmer in stemmer_options:
            print(f"\n{'='*60}")
            print(f"RUN {run_counter}/{total_runs}")
            print(f"Configuration: stop={stop}, stemmer={stemmer}")
            print('='*60)
            
            # Configuration complète
            config = base_config.copy()
            config['stemmer'] = stemmer
            config['stop_words'] = stop
            
            # Identifiant du run
            run_id +=1 
            
            # Génération du run
            print(f"Génération en cours avec {algorithme.upper()}...")
            start_time = time.time()
            
            try:
                filename = generate_field_weighted_run_cached(
                    generator=generator,
                    run_id=run_id,
                    run_type=algorithme,  # 'bm25fw' ou 'bm25fr'
                    xml_dir=XML_DIR,
                    queries=INEX_QUERIES,
                    config=config,
                    run_params=run_params,
                    fields_config=fields_config,
                    field_weights=field_weights
                )
                
                generation_time = time.time() - start_time
                
                # Vérification du fichier
                line_count = 0
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                except Exception as e:
                    print(f"Erreur lecture fichier: {e}")
                    filename = None
                
                if filename:
                    # Stockage des résultats
                    result_entry = {
                        'run_number': run_counter,
                        'algorithme': algorithme,
                        'stop': stop,
                        'stemmer': stemmer,
                        'filename': filename,
                        'basename': os.path.basename(filename),
                        'line_count': line_count,
                        'generation_time': generation_time,
                        'config_summary': f"{algorithme}_{stop}_{stemmer}",
                        'full_config': {
                            'config': config,
                            'fields_config': fields_config,
                            'field_weights': field_weights,
                            'run_params': run_params
                        }
                    }
                    
                    results.append(result_entry)
                    
                    print(f"Run généré: {os.path.basename(filename)}")
                    print(f"Lignes: {line_count} (attendu: 10500)")
                    print(f"Temps génération: {generation_time:.2f}s")
                else:
                    print("Échec de la génération du run")
                    
            except Exception as e:
                print(f"Erreur lors de la génération: {e}")
                import traceback
                traceback.print_exc()
            
            run_counter += 1
    
    # Affichage du résumé
    print("\n" + "="*70)
    print(f"RÉSUMÉ PHASE 1 - PRÉTRAITEMENT ({algorithme.upper()})")
    print("="*70)
    
    if not results:
        print("Aucun run n'a été généré avec succès.")
        return None
    
    print("\nListe des runs générés:")
    print("-" * 80)
    print(f"{'No':<4} {'Algo':<8} {'Stop':<10} {'Stem':<8} {'Lignes':<8} {'Fichier'}")
    print("-" * 80)
    
    for result in results:
        status = "OK" if result['line_count'] == 10500 else f"{result['line_count']}"
        print(f"{result['run_number']:<4} "
              f"{result['algorithme']:<8} "
              f"{result['stop']:<10} "
              f"{result['stemmer']:<8} "
              f"{status:<8} "
              f"{result['basename'][:40]}...")
    
    print("\n" + "="*70)
    
    return results

def exercice5_phase1():
    """Wrapper pour la phase 1 de l'exercice 5 (BM25Fw)"""
    print_exercise_header(5, "BM25Fw - Phase 1: Optimisation prétraitement")
    return exercice5_6_test1(algorithme="bm25fw")

def exercice6_phase1():
    """Wrapper pour la phase 1 de l'exercice 6 (BM25Fr)"""
    print_exercise_header(6, "BM25Fr - Phase 1: Optimisation prétraitement")
    return exercice5_6_test1(algorithme="bm25fr")
