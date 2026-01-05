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
        'use_lxml': True
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
    """Exercice 2: 12 runs avec différentes combinaisons"""
    print_exercise_header(2, "XML documents test runs (12 combinaisons)")
    
    generator = INEXRunGenerator()
    all_results = []
    
    weighting_schemes = ["ltn", "ltc", "bm25"]
    stop_options = ["nostop", "stop671"]
    stemmer_options = ["nostem", "porter"]
    
    run_counter = 1
    
    for weighting in weighting_schemes:
        for stop in stop_options:
            for stemmer in stemmer_options:
                print(f"\n{'='*50}")
                print(f"CONFIGURATION {run_counter}/12: {weighting.upper()}, {stop}, {stemmer}")
                print('='*50)
                
                # Configuration
                config = {
                    'tokenization': 'basic',
                    'stemmer': stemmer,
                    'stop_words': stop,
                    'test_type': 'test2',
                    'use_lxml': True
                }
                
                # Charger/créer l'index
                index_data = generator.create_or_load_index(
                    xml_dir=XML_DIR,
                    index_type='article',
                    config=config
                )
                
                # Paramètres BM25 si nécessaire
                k1, b = (1.2, 0.75) if weighting == "bm25" else (None, None)
                
                # Calculer statistiques
                stats_data = compute_statistics_for_config(
                    index_data=index_data,
                    weighting_scheme=weighting,
                    k1=k1,
                    b=b
                )
                
                # Afficher statistiques résumées
                print(f"\nSTATISTIQUES ({weighting.upper()}, {stop}, {stemmer}):")
                print(f"- Temps total: {stats_data['total_time']:.2f}s")
                print(f"- Tokens distincts: {stats_data['stats']['distinct_tokens']}")
                print(f"- Terms distincts: {stats_data['stats']['distinct_terms']}")
                print(f"- Longueur moyenne doc: {stats_data['stats']['avg_doc_length']:.2f}")
                print(f"- Poids '{TARGET_TERM}': {stats_data['target_weight']:.6f}")
                print(f"- RSV doc #{TARGET_DOC_ID}: {stats_data['doc_score']:.6f}")
                
                # Génération run
                filename = generator.generate_article_run(
                    run_id=f"{run_counter}",
                    xml_dir=XML_DIR,
                    queries=INEX_QUERIES,
                    config=config,
                    weighting_scheme=weighting,
                    k1=k1,
                    b=b
                )
                
                all_results.append({
                    'config_num': run_counter,
                    'config': config,
                    'weighting': weighting,
                    'filename': filename,
                    'stats': stats_data
                })
                
                run_counter += 1
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ EXERCICE 2")
    print("="*70)
    for result in all_results:
        cfg = result['config']
        basename = os.path.basename(result['filename'])
        print(f"{result['config_num']:2d}. {cfg['stemmer']:7s} | "
              f"stop={cfg['stop_words']:8s} | "
              f"{result['weighting']:4s} | {basename}")
    
    return all_results

# ==================== EXERCICE 3 ====================
"""
def exercice3():
    #Exercice 3: Indexation XML éléments (SMART ltn
    print_exercise_header(3, "Indexation XML éléments (bdy, sec, p) - SMART ltn")
    
    generator = INEXRunGenerator()
    
    # Générer le run spécifique exercice 3
    filename = generator.generate_element_run_exercise3(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES
    )
    
    print(f"\nExercice 3 terminé")
    print(f"Run généré: {filename}")
    
    return filename
"""
def exo3():
    # Initialisation
    run_gen = INEXRunGenerator(team_name="AlphaAnaClement")

    # 1. Charger les queries INEX (à adapter selon ton format)
    
    """
    # 2. Exercice 3 - Version simple
    run_gen.generate_element_run_simple(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES
    )
    """

    # 3. Exercices 4-6 - Fetch & Browse
    run_gen.generate_fetch_browse_run_optimized(
        run_id="ex4_exp1",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        fetch_config={
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'stop671'
        },
        browse_config={
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'stop671',
            'target_tags': ['bdy', 'sec', 'p', 'article']
        },
        run_params = {
            'top_articles': 1500,
            'max_elements': 1500,
            'max_elements_per_article': 5,
            'weighting_scheme': 'ltn',
            'min_element_score': 0.01,
            'selection_strategy': 'hierarchical',
            'avoid_overlaps': True,
            'fallback_to_article': True
        }
    )
# ==================== EXERCICE 4 ====================

def exercice4():
    """Exercice 4: Expérimentation avec éléments XML"""
    print_exercise_header(4, "Expérimentation avec éléments XML")
    
    generator = INEXRunGenerator()
    results = []
    
    # Configurations à tester
    configurations = [
        {
            'name': 'Baseline éléments',
            'target_tags': ['bdy', 'sec', 'p'],
            'weighting': 'ltn',
            'stemmer': 'nostem',
            'stop_words': 'nostop'
        },
        {
            'name': 'Sections seulement (ltc)',
            'target_tags': ['sec'],
            'weighting': 'ltc',
            'stemmer': 'porter',
            'stop_words': 'stop671'
        },
        {
            'name': 'Paragraphes seulement (BM25)',
            'target_tags': ['p'],
            'weighting': 'bm25',
            'k1': 1.5,
            'b': 0.8,
            'stemmer': 'porter',
            'stop_words': 'stop319'
        }
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\nConfiguration {i}: {config['name']}")
        print(f"  Tags: {config['target_tags']}")
        print(f"  Pondération: {config.get('weighting', 'ltn')}")
        
        # Préparer la configuration d'index
        index_config = {
            'tokenization': 'basic',
            'stemmer': config.get('stemmer', 'nostem'),
            'stop_words': config.get('stop_words', 'nostop'),
            'target_tags': config['target_tags'],
            'use_lxml': True
        }
        
        # Charger/créer l'index d'éléments
        index_data = generator.create_or_load_index(
            xml_dir=XML_DIR,
            index_type='element',
            config=index_config
        )
        
        # Paramètres de pondération
        weighting = config.get('weighting', 'ltn')
        k1 = config.get('k1', 1.2) if weighting == 'bm25' else None
        b = config.get('b', 0.75) if weighting == 'bm25' else None
        
        # Calculer statistiques
        stats_data = compute_statistics_for_config(
            index_data=index_data,
            weighting_scheme=weighting,
            k1=k1,
            b=b
        )
        
        # Afficher statistiques résumées
        print(f"  Statistiques:")
        print(f"  - Éléments indexés: {stats_data['stats'].get('doc_count', stats_data['index'].doc_count)}")
        print(f"  - Temps indexation: {stats_data['indexing_time']:.2f}s")
        print(f"  - Longueur moyenne élément: {stats_data['stats']['avg_doc_length']:.2f} terms")
        
        # Générer le run
        filename = generator.generate_element_run(
            run_id=f"{i}_test4",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=index_config,
            weighting_scheme=weighting,
            k1=k1,
            b=b
        )
        
        results.append({
            'name': config['name'],
            'config': index_config,
            'filename': filename,
            'stats': stats_data
        })
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ EXERCICE 4")
    print("="*70)
    for result in results:
        print(f"{result['name']}:")
        print(f"  Run: {os.path.basename(result['filename'])}")
    
    return results

# ==================== EXERCICES 5 et 6 ====================

def exercice5():
    """Exercice 5: BM25Fw - Late combination"""
    print_exercise_header(5, "BM25Fw - Late combination of fields")
    
    try:
        from exercices_5_6_with_cache import generate_field_weighted_run_cached
        
        # Configuration
        config = {
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'stop671',
            'use_lxml': True
        }
        
        fields_config = {
            'title': ['title'],
            'body': ['bdy'],
        }
        
        field_weights = {
            'title': 2.5,
            'body': 1.0
        }
        
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
        
        # Configuration
        config = {
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'nostop',
            'use_lxml': True
        }
        
        fields_config = {
            'title': ['title'],
            'abstract': ['bdy'],
            'body': ['bdy']
        }
        
        field_weights = {
            'title': 3.0,
            'abstract': 1.5,
            'body': 1.0
        }
        
        run_params = {
            'k1': 1.5,
            'b': 0.8,
            'max_files': None
        }
        
        generator = INEXRunGenerator()
        
        filename = generate_field_weighted_run_cached(
            generator=generator,
            run_id=f"{GROUP_NUMBER}_test6",
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

def main(selected_exercises: List[int] = None):
    """Exécute tous les exercices ou seulement ceux spécifiés"""
    print("=" * 70)
    print("PRACTICAL SESSION 5: Structured IR at INEX")
    print("=" * 70)
    
    # Nettoyage initial
    clean_runs_directory()
    
    # Créer dossier runs si nécessaire
    os.makedirs("data/runs", exist_ok=True)
    
    # Vérifier données
    if not os.path.exists(XML_DIR):
        print(f"ERREUR: Dossier de données non trouvé: {XML_DIR}")
        return
    
    # Mapping des exercices
    exercises = {
        1: ("Indexation XML documents (SMART ltn)", exercice1),
        2: ("Test runs documents (12 combinaisons)", exercice2),
        3: ("Indexation XML éléments (SMART ltn)", exercice3),
        4: ("Expérimentation éléments", exercice4),
        5: ("BM25Fw - Late combination", exercice5),
        6: ("BM25Fr - Early combination", exercice6)
    }
    
    # Exécuter tous ou sélection
    if selected_exercises is None:
        selected_exercises = list(exercises.keys())
    
    results = {}
    
    for ex_num in selected_exercises:
        if ex_num in exercises:
            name, func = exercises[ex_num]
            print(f"\n{'='*70}")
            print(f"LANCEMENT EXERCICE {ex_num}: {name}")
            print('='*70)
            
            try:
                results[f'ex{ex_num}'] = func()
                print(f"✅ Exercice {ex_num} terminé")
            except Exception as e:
                print(f"❌ Erreur exercice {ex_num}: {e}")
                import traceback
                traceback.print_exc()
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DE LA SESSION")
    print("="*70)
    
    for ex_num in selected_exercises:
        if ex_num in exercises:
            print(f"Exercice {ex_num}: {exercises[ex_num][0]}")
            if f'ex{ex_num}' in results:
                result = results[f'ex{ex_num}']
                if isinstance(result, dict) and 'filename' in result:
                    print(f"  Run: {os.path.basename(result['filename'])}")
                elif isinstance(result, str):
                    print(f"  Run: {os.path.basename(result)}")
                elif isinstance(result, list):
                    print(f"  {len(result)} configurations testées")
    
    print(f"\nTous les exercices terminés")
    print(f"Les fichiers sont dans: data/runs/")
    
    return results

# ==================== EXÉCUTION ====================

if __name__ == "__main__":
    import sys
    
    # Permettre de spécifier des exercices
    if len(sys.argv) > 1:
        selected = []
        for arg in sys.argv[1:]:
            try:
                ex_num = int(arg)
                if 1 <= ex_num <= 6:
                    selected.append(ex_num)
                else:
                    print(f"Exercice {ex_num} invalide (doit être 1-6)")
            except ValueError:
                print(f"Argument invalide: {arg}")
        
        if selected:
            main(selected_exercises=selected)
        else:
            print("Usage: python main.py [exercice1] [exercice2] ...")
            print("Exemple: python main.py 1 3 5")
            main()  # Tous par défaut
    else:
        # Exécuter tous les exercices
        #main()
        print("hello")

    #clean_runs_directory()
    #exercice1()
    #exercice2()
    #exercice3()
    exo3()

