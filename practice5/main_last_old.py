import os
import time
import hashlib
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Import de tes modules
from indexer import WeightedInvertedIndex, INEXDocument
from ranked_retrieval import RankedRetrieval
from xml_run_manager import INEXRunGenerator
# Pour exercices 5-6
try:
    from exercices_5_6 import FieldWeightedIndex, generate_field_weighted_run_cached
except ImportError:
    pass  # Géré plus tard

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

def generate_run_filename(run_id: str, test_type: str, 
                         weighting_scheme: str, granularity: str,
                         stemmer: str = "nostem", stop_words: str = "nostop",
                         tokenization: str = "basic", k1: float = None, 
                         b: float = None, fields: List[str] = None,
                         method: str = None) -> str:
    """
    Génère un nom de fichier INEX conforme
    """
    parts = [TEAM_NAME, run_id, test_type]
    
    # Méthode spécifique (pour exercices 5-6)
    if method:
        parts.append(method)
    
    # Champs (pour exercices 5-6)
    if fields:
        parts.append(f"fields-{'-'.join(fields)}")
    
    parts.extend([weighting_scheme, granularity, stop_words, stemmer])
    
    if tokenization != "basic":
        parts.append(tokenization)
    
    if weighting_scheme == "bm25" and k1 is not None and b is not None:
        parts.extend([f"k{k1}", f"b{b}"])
    
    filename = "_".join(parts) + ".txt"
    return os.path.join("data/runs", filename)

def print_statistics_header(exercise_num: int, config_desc: str):
    """Affiche l'en-tête des statistiques"""
    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {config_desc}")
    print("=" * 60)

def compute_statistics(index_data: Dict, weighting_scheme: str = "ltn",
                      k1: float = 1.2, b: float = 0.75,
                      target_doc_id: str = TARGET_DOC_ID,
                      target_term: str = TARGET_TERM,
                      test_query: str = TEST_QUERY) -> RankedRetrieval:
    """
    Calcule et affiche les statistiques d'une configuration
    """
    index = index_data['index']
    indexing_time = index_data['indexing_time']
    stats = index_data['stats']
    config = index_data['config']
    
    # Construction de la description
    config_desc = f"{weighting_scheme.upper()}"
    if config.get('stop_words', 'nostop') != "nostop":
        config_desc += f" + stop-words({config['stop_words']})"
    if config.get('stemmer', 'nostem') != "nostem":
        config_desc += f" + stemming({config['stemmer']})"
    if config.get('tokenization', 'basic') != "basic":
        config_desc += f" + tokenization({config['tokenization']})"
    
    if weighting_scheme == "bm25":
        config_desc += f" - k1={k1}, b={b}"
    
    print_statistics_header("", config_desc)
    
    start_total_time = time.time()
    ranker = RankedRetrieval(index)
    
    # Nettoyer cache si nécessaire
    if weighting_scheme == "ltc":
        ranker.clear_cosine_norms_cache()
    
    # Calcul des poids
    query_terms = ranker.process_query_terms(test_query)
    target_terms = ranker.process_query_terms(target_term)
    
    target_weight = 0.0
    if target_terms:
        target_weight = ranker.get_term_weight(
            target_terms[0], target_doc_id, weighting_scheme, k1=k1, b=b
        )
    
    doc_score = sum(
        ranker.get_term_weight(t, target_doc_id, weighting_scheme, k1=k1, b=b)
        for t in query_terms
    )
    
    # Recherche top-10
    top_docs = ranker.search_query(test_query, weighting_scheme, top_k=10, k1=k1, b=b)
    
    weighting_time = time.time() - start_total_time
    total_time = indexing_time + weighting_time
    
    # Affichage des statistiques
    print(f"\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Configuration: {config_desc}")
    print(f"- Temps total d'indexation + pondération: {total_time:.2f} secondes")
    print(f" * Temps d'indexation seul: {indexing_time:.2f} secondes")
    print(f" * Temps de pondération: {weighting_time:.2f} secondes")
    print(f"- Nombre total d'occurrences de tokens: {stats['total_tokens']}")
    print(f"- Nombre de tokens distincts: {stats['distinct_tokens']}")
    print(f"- Longueur moyenne des tokens: {stats['avg_token_length']:.2f} caractères")
    print(f"- Nombre total d'occurrences de terms: {stats['total_terms']}")
    print(f"- Taille du vocabulaire (terms distincts): {stats['distinct_terms']}")
    print(f"- Longueur moyenne des documents: {stats['avg_doc_length']:.2f} terms")
    print(f"- Longueur moyenne des terms: {stats['avg_term_length']:.2f} caractères")
    print(f"- Poids du terme '{target_term}' dans le document #{target_doc_id}: {target_weight:.6f}")
    print(f"- RSV du document #{target_doc_id} pour '{test_query}': {doc_score:.6f}")
    
    print(f"- TOP-10 DOCUMENTS pour '{test_query}':")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
    
    return ranker

def clean_runs_directory(force: bool = False):
    """Nettoie le dossier des runs"""
    if os.path.exists("data/runs"):
        if force:
            for file in os.listdir("data/runs"):
                if file.endswith(".txt"):
                    os.remove(os.path.join("data/runs", file))
            print("✅ Dossier 'runs' nettoyé")
        else:
            response = input("\nNettoyer le dossier 'data/runs' ? (o/n): ")
            if response.lower() == 'o':
                clean_runs_directory(force=True)

# ==================== EXERCICE 1 ====================

def exercice1():
    """Exercice 1: Indexation XML documents (SMART lm)"""
    print("=" * 70)
    print("EXERCICE 1: Indexation XML documents (SMART lm)")
    print("=" * 70)
    
    # Configuration exercice 1
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'use_lxml': True
    }
    
    # 1. Création de l'index
    print("\n[Étape 1/3] Création de l'index...")
    index = WeightedInvertedIndex()
    index.configure(**config)
    
    index_data = index.build_index_with_stats(
        xml_dir=XML_DIR,
        max_files=None
    )
    
    # 2. Calcul des statistiques
    print("\n[Étape 2/3] Calcul des statistiques...")
    ranker = compute_statistics(
        index_data=index_data,
        weighting_scheme="ltn"  # SMART lm = ltn
    )
    
    # 3. Génération du run
    print("\n[Étape 3/3] Génération du run INEX...")
    generator = INEXRunGenerator()
    
    filename = generator.generate_article_run(
        run_id="1",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config={**config, 'test_type': 'test1'},
        weighting_scheme="lm"  # SMART lm
    )
    
    print(f"\n✅ Exercice 1 terminé")
    print(f"📁 Run généré: {filename}")
    
    return {
        'index_data': index_data,
        'ranker': ranker,
        'filename': filename
    }

# ==================== EXERCICE 2 ====================

def exercice2():
    """Exercice 2: 12 runs avec différentes combinaisons"""
    print("\n" + "=" * 70)
    print("EXERCICE 2: XML documents test runs (12 combinaisons)")
    print("=" * 70)
    
    generator = INEXRunGenerator()
    all_results = []
    
    weighting_schemes = ["lm", "ltc", "bm25"]
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
                    'use_lxml': True,
                    'test_type': 'test2'
                }
                
                # Création index
                index = WeightedInvertedIndex()
                index.configure(**config)
                
                index_data = index.build_index_with_stats(
                    xml_dir=XML_DIR,
                    max_files=None
                )
                
                # Calcul statistiques
                k1, b = (1.2, 0.75) if weighting == "bm25" else (None, None)
                
                ranker = compute_statistics(
                    index_data=index_data,
                    weighting_scheme=weighting,
                    k1=k1,
                    b=b
                )
                
                # Génération run
                filename = generator.generate_article_run(
                    run_id=str(run_counter),
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
                    'filename': filename,
                    'ranker': ranker
                })
                
                run_counter += 1
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ EXERCICE 2")
    print("="*70)
    for result in all_results:
        cfg = result['config']
        print(f"{result['config_num']:2d}. {cfg['stemmer']:7s} | "
              f"stop={cfg['stop_words']:8s} | "
              f"{os.path.basename(result['filename'])}")
    
    return all_results

# ==================== EXERCICE 3 ====================

def exercice3():
    """Exercice 3: Indexation XML éléments (SMART lm)"""
    print("\n" + "=" * 70)
    print("EXERCICE 3: Indexation XML éléments (bdy, sec, p) - SMART lm")
    print("=" * 70)
    
    generator = INEXRunGenerator()
    
    # Générer le run spécifique exercice 3
    filename = generator.generate_element_run_exercise3(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES
    )
    
    print(f"\n✅ Exercice 3 terminé")
    print(f"📁 Run généré: {filename}")
    
    return filename

# ==================== EXERCICE 4 ====================

def exercice4(configs: List[Dict] = None):
    """
    Exercice 4: Expérimentation avec éléments XML
    configs: Liste de configurations à tester
    """
    print("\n" + "=" * 70)
    print("EXERCICE 4: Expérimentation avec éléments XML")
    print("=" * 70)
    
    if configs is None:
        # Configurations par défaut pour exploration
        configs = [
            {
                'name': 'Baseline améliorée',
                'target_tags': ['bdy', 'sec', 'p'],
                'weighting': 'bm25',
                'k1': 1.5,
                'b': 0.8,
                'stemmer': 'porter',
                'stop_words': 'stop671'
            },
            {
                'name': 'Précision maximale',
                'target_tags': ['p'],
                'weighting': 'ltc',
                'stemmer': 'snowball',
                'stop_words': 'stop671'
            },
            {
                'name': 'Sections seulement',
                'target_tags': ['sec'],
                'weighting': 'bm25',
                'k1': 1.2,
                'b': 0.75,
                'stemmer': 'porter',
                'stop_words': 'stop319'
            }
        ]
    
    generator = INEXRunGenerator()
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config['name']}")
        print(f"  Tags: {config['target_tags']}")
        print(f"  Pondération: {config.get('weighting', 'lm')}")
        
        run_config = {
            'tokenization': 'basic',
            'stemmer': config.get('stemmer', 'nostem'),
            'stop_words': config.get('stop_words', 'nostop'),
            'target_tags': config['target_tags'],
            'test_type': 'exp4'
        }
        
        # Indexation éléments
        index_data = generator.create_or_load_index(
            xml_dir=XML_DIR,
            index_type='element',
            config=run_config
        )
        
        # Calcul statistiques
        weighting = config.get('weighting', 'lm')
        k1 = config.get('k1', 1.2)
        b = config.get('b', 0.75)
        
        ranker = compute_statistics(
            index_data=index_data,
            weighting_scheme=weighting,
            k1=k1 if weighting == 'bm25' else None,
            b=b if weighting == 'bm25' else None
        )
        
        # Génération run avec xml_run_manager (à adapter selon ton code)
        # Utilise ta méthode existante ou adapte generate_fetch_browse_run
        
        results.append({
            'name': config['name'],
            'config': run_config,
            'ranker': ranker
        })
    
    return results

# ==================== EXERCICE 5 ====================

def exercice5():
    """Exercice 5: BM25Fw - Late combination (Wilkinson94)"""
    print("=" * 70)
    print("EXERCICE 5: BM25Fw - Late combination of fields")
    print("=" * 70)
    
    try:
        from exercices_5_6 import generate_field_weighted_run_cached
        
        config = {
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'stop671',
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
        
        print(f"\n✅ Exercice 5 terminé")
        print(f"📁 Run généré: {filename}")
        
        return filename
        
    except ImportError as e:
        print(f"❌ Module exercices_5_6 non disponible: {e}")
        print("Exécute d'abord exercices_5_6.py pour générer les runs")
        return None

# ==================== EXERCICE 6 ====================

def exercice6():
    """Exercice 6: BM25Fr - Early combination (Robertson94)"""
    print("\n" + "=" * 70)
    print("EXERCICE 6: BM25Fr - Early combination of fields")
    print("=" * 70)
    
    try:
        from exercices_5_6 import generate_field_weighted_run_cached
        
        config = {
            'tokenization': 'basic',
            'stemmer': 'porter',
            'stop_words': 'nostop',
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
            run_id="test6",
            run_type="bm25fr",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params=run_params,
            fields_config=fields_config,
            field_weights=field_weights
        )
        
        print(f"\n✅ Exercice 6 terminé")
        print(f"📁 Run généré: {filename}")
        
        return filename
        
    except ImportError as e:
        print(f"❌ Module exercices_5_6 non disponible: {e}")
        return None

# ==================== FONCTION PRINCIPALE ====================

def main(selected_exercises: List[int] = None):
    """
    Exécute tous les exercices ou seulement ceux spécifiés
    """
    print("=" * 70)
    print("PRACTICAL SESSION 5: Structured IR at INEX")
    print("=" * 70)
    
    # Nettoyage initial
    clean_runs_directory()
    
    # Créer dossier runs si nécessaire
    os.makedirs("data/runs", exist_ok=True)
    
    # Vérifier données
    if not os.path.exists(XML_DIR):
        print("❌ ERREUR: Dossier de données non trouvé")
        print(f"Placez les données dans: {XML_DIR}")
        return
    
    # Mapping des exercices
    exercises = {
        1: ("Indexation XML documents", exercice1),
        2: ("Test runs documents", exercice2),
        3: ("Indexation XML éléments", exercice3),
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
                    print(f"  📁 {os.path.basename(result['filename'])}")
                elif isinstance(result, str):
                    print(f"  📁 {os.path.basename(result)}")
                elif isinstance(result, list):
                    print(f"  📊 {len(result)} configurations testées")
    
    print(f"\n✅ Tous les exercices terminés")
    print(f"📁 Tous les fichiers sont dans: data/runs/")
    
    return results

# ==================== EXÉCUTION ====================

if __name__ == "__main__":
    import sys
    
    # Permettre de spécifier des exercices: python nouveau_main.py 1 3 5
    if len(sys.argv) > 1:
        selected = []
        for arg in sys.argv[1:]:
            try:
                ex_num = int(arg)
                if 1 <= ex_num <= 6:
                    selected.append(ex_num)
                else:
                    print(f"⚠️  Exercice {ex_num} invalide (doit être 1-6)")
            except ValueError:
                print(f"⚠️  Argument invalide: {arg}")
        
        if selected:
            main(selected_exercises=selected)
        else:
            print("Usage: python nouveau_main.py [exercice1] [exercice2] ...")
            print("Exemple: python nouveau_main.py 1 3 5")
            main()  # Tous par défaut
    else:
        # Exécuter tous les exercices
        main()