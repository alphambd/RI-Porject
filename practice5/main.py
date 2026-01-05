import os
import time
from typing import Dict, List, Optional
from xmlrm import INEXRunGenerator
from indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

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

# ==================== FONCTIONS UTILITAIRES COMMUNES ====================

def clean_runs_directory():
    """Nettoie le dossier des runs"""
    if os.path.exists("data/runs"):
        response = input("\nNettoyer le dossier 'data/runs' ? (o/n): ")
        if response.lower() == 'o':
            for file in os.listdir("data/runs"):
                if file.endswith(".txt"):
                    os.remove(os.path.join("data/runs", file))
            print("Dossier 'runs' nettoyé")

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
    # Initialisation
    run_gen = INEXRunGenerator(team_name="AlphaAnaClement")

    run_gen.generate_exercise3_fetch_browse(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        with_article=True
    )

# ==================== EXERCICE 4 ====================
"""
def exercice4_phase1_ponderation():
    #Exercice 4 - Phase 1: Tester les pondérations
    print("\n" + "="*70)
    print("EXERCICE 4 - PHASE 1: TEST DES PONDÉRATIONS")
    print("Granularité: bdy, sec, p")
    print("Stop-words: nostop, Stemmer: nostem")
    print("="*70)
    
    generator = INEXRunGenerator(team_name="AlphaAnaClement")
    
    # Configuration fixe pour cette phase
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
    
    # Pondérations à tester
    weighting_schemes = ['ltn', 'ltc', 'bm25']
    
    results = []
    
    for weighting in weighting_schemes:
        print(f"\n{'='*50}")
        print(f"TEST PONDÉRATION: {weighting.upper()}")
        print('='*50)
        
        # Configuration des paramètres selon la pondération
        run_params = {
            'top_articles': 2000,
            'max_elements': 1500,
            'max_elements_per_article': 1,
            'weighting_scheme': weighting,
            'selection_strategy': 'hierarchical',
            'avoid_overlaps': True,
            'fallback_to_article': False
        }
        
        # Ajouter paramètres BM25 si nécessaire
        if weighting == 'bm25':
            run_params['bm25_k1'] = 1.2
            run_params['bm25_b'] = 0.75
        
        # Générer le run
        run_id = f"testXML2_{weighting}"
        
        filename = generator.generate_fetch_browse_run_optimized(
            run_id=run_id,
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            fetch_config=fetch_config,
            browse_config=browse_config,
            run_params=run_params
        )
        
        # Vérifier le fichier généré
        line_count = 0
        try:
            with open(filename, 'r') as f:
                line_count = sum(1 for _ in f)
        except:
            pass
        
        results.append({
            'weighting': weighting,
            'filename': filename,
            'line_count': line_count,
            'config': {
                'weighting_scheme': weighting,
                'run_params': run_params
            }
        })
        
        print(f"Run généré: {os.path.basename(filename)}")
        print(f"Lignes: {line_count}")
    
    # Résumé de la phase X
    print("\n" + "="*70)
    print("RÉSUMÉ PHASE X - PONDÉRATIONS")
    print("="*70)
    
    for result in results:
        print(f"{result['weighting'].upper():5s} : {os.path.basename(result['filename'])}")
        print(f"       Lignes: {result['line_count']}")
    
    return results
"""
def exercice4_phase_pretraitement():
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
    
    print("\n" + "="*70)
    print("INSTRUCTIONS POUR LE TEST:")
    print("="*70)
    print("1. Tous les runs sont dans le dossier 'data/runs/'")
    print("2. Uploader chaque run sur http://ri.gery.fr")
    print("3. Noter pour chaque run:")
    print("   - MAP (Mean Average Precision)")
    print("   - P@10 (Precision at 10)")
    print("4. Identifier la meilleure combinaison")
    print("5. Utiliser cette combinaison pour la phase suivante")
    
    # Sauvegarde des métadonnées pour référence
    metadata_file = "data/runs/ex4_phase1_metadata.txt"
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("MÉTADONNÉES EXERCICE 4 - PHASE 1\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre de runs: {len(results)}\n\n")
        
        for result in results:
            f.write(f"\nRUN {result['run_number']}:\n")
            f.write(f"  Pondération: {result['weighting']}\n")
            f.write(f"  Stop-words: {result['stop']}\n")
            f.write(f"  Stemmer: {result['stemmer']}\n")
            f.write(f"  Fichier: {result['basename']}\n")
            f.write(f"  Lignes: {result['line_count']}\n")
            f.write(f"  Temps génération: {result['generation_time']:.2f}s\n")
            f.write("  Résultats (à compléter après test):\n")
            f.write("    MAP: ______\n")
            f.write("    P@10: ______\n")
            f.write("    Observations: ______________________\n")
    
    print(f"\nMétadonnées sauvegardées dans: {metadata_file}")
    
    return results

def exercice5_6_phase1_pretraitement(algorithme="bm25fr"):
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
    
    try:
        from exercices_5_6_with_cache import generate_field_weighted_run_cached
    except ImportError as e:
        print(f"Module exercices_5_6_with_cache non disponible: {e}")
        print("Exécutez d'abord le script pour générer les runs")
        return None
    
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
    print("INSTRUCTIONS POUR LE TEST:")
    print("="*70)
    print("1. Tous les runs sont dans le dossier 'data/runs/'")
    print("2. Uploader chaque run sur http://ri.gery.fr")
    print("3. Noter pour chaque run:")
    print("   - MAP (Mean Average Precision)")
    print("   - P@10 (Precision at 10)")
    print("4. Identifier la meilleure combinaison stop/stemmer")
    print("5. Utiliser cette combinaison pour la phase 2 (optimisation des poids)")
    
    # Sauvegarde des métadonnées
    metadata_file = f"data/runs/ex5-6_phase1_{algorithme}_metadata.txt"
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"MÉTADONNÉES EXERCICES 5-6 - PHASE 1 ({algorithme.upper()})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithme: {algorithme}\n")
        f.write(f"Nombre de runs: {len(results)}\n\n")
        f.write(f"Paramètres fixes:\n")
        f.write(f"  k1: {run_params['k1']}\n")
        f.write(f"  b: {run_params['b']}\n")
        f.write(f"  α_title: {field_weights['title']}\n")
        f.write(f"  α_body: {field_weights['body']}\n\n")
        
        for result in results:
            f.write(f"\nRUN {result['run_number']}:\n")
            f.write(f"  Algorithme: {result['algorithme']}\n")
            f.write(f"  Stop-words: {result['stop']}\n")
            f.write(f"  Stemmer: {result['stemmer']}\n")
            f.write(f"  Fichier: {result['basename']}\n")
            f.write(f"  Lignes: {result['line_count']}\n")
            f.write(f"  Temps génération: {result['generation_time']:.2f}s\n")
            f.write("  Résultats (à compléter après test):\n")
            f.write("    MAP: ______\n")
            f.write("    P@10: ______\n")
            f.write("    Observations: ______________________\n")
    
    print(f"\nMétadonnées sauvegardées dans: {metadata_file}")
    
    return results


def exercice5_phase1():
    """Wrapper pour la phase 1 de l'exercice 5 (BM25Fw)"""
    print_exercise_header(5, "BM25Fw - Phase 1: Optimisation prétraitement")
    return exercice5_6_phase1_pretraitement(algorithme="bm25fw")


def exercice6_phase1():
    """Wrapper pour la phase 1 de l'exercice 6 (BM25Fr)"""
    print_exercise_header(6, "BM25Fr - Phase 1: Optimisation prétraitement")
    return exercice5_6_phase1_pretraitement(algorithme="bm25fr")


# Fonctions pour exécuter les deux algorithmes en parallèle
def exercices5_6_phase1_complet():
    """
    Exécute la phase 1 pour les deux algorithmes (BM25Fw et BM25Fr)
    """
    print("\n" + "="*70)
    print("EXERCICES 5 & 6 - PHASE 1 COMPLÈTE")
    print("Optimisation prétraitement pour les deux algorithmes")
    print("="*70)
    
    results_bm25fw = exercice5_phase1()
    print("\n" + "="*70)
    print("PHASE 1 BM25Fw TERMINÉE")
    print("="*70)
    
    input("\nAppuyez sur Entrée pour passer à BM25Fr...")
    
    results_bm25fr = exercice6_phase1()
    print("\n" + "="*70)
    print("PHASE 1 BM25Fr TERMINÉE")
    print("="*70)
    
    return {
        'bm25fw': results_bm25fw,
        'bm25fr': results_bm25fr
    }



# ==================== EXERCICES 5 et 6 ====================

def exercice5():
    """Exercice 5: BM25Fw - Late combination"""
    print_exercise_header(5, "BM25Fw - Late combination of fields")
    
    try:
        from exercices_5_6_with_cache import generate_field_weighted_run_cached
        
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
            'title': 3.0,    
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
        
    except ImportError as e:
        print(f"Module exercices_5_6_with_cache non disponible: {e}")
        print("Exécutez d'abord le script pour générer les runs")
        return None

def exercice6():
    """Exercice 6: BM25Fr - Early combination"""
    print_exercise_header(6, "BM25Fr - Early combination of fields")
    
    try:
        from exercices_5_6_with_cache import generate_field_weighted_run_cached
        
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
            'title': 3.0,           # Très important
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
        
    except ImportError as e:
        print(f"Module exercices_5_6_with_cache non disponible: {e}")
        return None

# ==================== FONCTION PRINCIPALE ====================

def main():
    
    # Nettoyage initial
    clean_runs_directory()
    
    # Créer dossier runs si nécessaire
    os.makedirs("data/runs", exist_ok=True)
    
    # Vérifier données
    if not os.path.exists(XML_DIR):
        print(f"ERREUR: Dossier de données non trouvé: {XML_DIR}")
        return
    

    #exercice1()
    #exercice2()
    #exercice3()
    #exercice4_phase_pretraitement()
    #exercice5()
    #exercice6()
    exercices5_6_phase1_complet()
    #exercice6_phase1()

    
    

# ==================== EXÉCUTION ====================

if __name__ == "__main__":
        
    main()
