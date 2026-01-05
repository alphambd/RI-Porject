import os
import time
from typing import Dict, List, Optional
from xml_run_manager import INEXRunGenerator

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

def print_statistics_header(config_desc: str):
    """Affiche l'en-tête des statistiques"""
    print("\n" + "=" * 60)
    print(f"EXERCICE : {config_desc}")
    print("=" * 60)

def clean_runs_directory(force: bool = False):
    """Nettoie le dossier des runs"""
    if os.path.exists("data/runs"):
        if force:
            for file in os.listdir("data/runs"):
                if file.endswith(".txt"):
                    os.remove(os.path.join("data/runs", file))
            print("Dossier 'runs' nettoyé")
        else:
            response = input("\nNettoyer le dossier 'data/runs' ? (o/n): ")
            if response.lower() == 'o':
                clean_runs_directory(force=True)

# ==================== EXERCICE 1 ====================

def exercice1():
    """Exercice 1: Indexation XML documents (SMART ltn)"""
    print("=" * 70)
    print("EXERCICE 1: Indexation XML documents (SMART ltn)")
    print("=" * 70)
    
    # Configuration exercice 1 - SMART ltn au lieu de lm
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'use_lxml': True
    }
    
    # Créer le générateur
    generator = INEXRunGenerator()
    
    # Étape 1: Création et statistiques de l'index
    print("\n[Étape 1/3] Création de l'index...")
    
    # Charger l'index depuis le cache ou le créer
    index_data = generator.create_or_load_index(
        xml_dir=XML_DIR,
        index_type='article',
        config=config
    )
    
    index = index_data['index']
    
    # Calculer les statistiques de base
    stats = index.get_collection_statistics(index_data['indexing_time'])
    
    # Pour avoir le temps de pondération, on doit faire la recherche
    from ranked_retrieval import RankedRetrieval
    ranker = RankedRetrieval(index)
    
    # Mesurer le temps de pondération
    weighting_start = time.time()
    query_terms = ranker.process_query_terms(TEST_QUERY)
    target_terms = ranker.process_query_terms(TARGET_TERM)
    
    target_weight = 0.0
    if target_terms:
        target_weight = ranker.get_term_weight(
            target_terms[0], TARGET_DOC_ID, "ltn"
        )
    
    doc_score = sum(
        ranker.get_term_weight(t, TARGET_DOC_ID, "ltn")
        for t in query_terms
    )
    
    # Recherche top-10 pour mesurer le temps
    top_docs = ranker.search_query(TEST_QUERY, "ltn", top_k=10)
    weighting_time = time.time() - weighting_start
    
    # Calculer le temps total
    indexing_time = index_data['indexing_time']
    total_time = indexing_time + weighting_time
    
    # Afficher les statistiques
    print_statistics_header("ltn")
    print("\nSTATISTIQUES DE LA COLLECTION:")
    print(f"- Configuration: ltn (SMART)")
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
    
    print(f"- Poids du terme '{TARGET_TERM}' dans le document #{TARGET_DOC_ID}: {target_weight:.6f}")
    print(f"- RSV du document #{TARGET_DOC_ID} pour '{TEST_QUERY}': {doc_score:.6f}")
    
    # Afficher le nombre de documents pertinents potentiels
    relevant_docs = ranker.search_query(TEST_QUERY, "ltn", top_k=None)
    print(f"  - Documents pertinents potentiels: {len(relevant_docs)}")
    
    print(f"- TOP-10 DOCUMENTS pour '{TEST_QUERY}':")
    for i, (doc_id, score) in enumerate(top_docs, 1):
        print(f"  {i:2d}. Doc {doc_id}: {score:.6f}")
    
    # Étape 2: Génération du run (sans réindexation)
    print("\n[Étape 2/3] Génération du run INEX...")
    
    filename = generator.generate_article_run(
        run_id="1_test1",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        config=config,
        weighting_scheme="ltn"  # Utiliser ltn au lieu de lm
    )
    
    print(f"\nExercice 1 terminé")
    print(f"Run généré: {filename}")
    
    return {
        'index_data': index_data,
        'ranker': ranker,
        'filename': filename,
        'stats': stats
    }

# ==================== EXERCICE 2 ====================

def exercice2():
    """Exercice 2: 12 runs avec différentes combinaisons"""
    print("\n" + "=" * 70)
    print("EXERCICE 2: XML documents test runs (12 combinaisons)")
    print("=" * 70)
    
    generator = INEXRunGenerator()
    all_results = []
    
    weighting_schemes = ["ltn", "ltc", "bm25"]  # ltn au lieu de lm
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
                    'use_lxml': True
                }
                
                # Charger/créer l'index
                index_data = generator.create_or_load_index(
                    xml_dir=XML_DIR,
                    index_type='article',
                    config=config
                )
                
                index = index_data['index']
                
                # Calculer et afficher statistiques
                stats = index.get_collection_statistics(index_data['indexing_time'])
                
                print(f"\nSTATISTIQUES ({weighting.upper()}, {stop}, {stemmer}):")
                print(f"- Temps d'indexation: {stats['indexing_time']:.2f}s")
                print(f"- Tokens distincts: {stats['distinct_tokens']}")
                print(f"- Terms distincts: {stats['distinct_terms']}")
                print(f"- Longueur moyenne doc: {stats['avg_doc_length']:.2f}")
                
                # Génération run
                filename = generator.generate_article_run(
                    run_id=str(run_counter),
                    xml_dir=XML_DIR,
                    queries=INEX_QUERIES,
                    config=config,
                    weighting_scheme=weighting,
                    k1=1.2 if weighting == "bm25" else None,
                    b=0.75 if weighting == "bm25" else None
                )
                
                all_results.append({
                    'config_num': run_counter,
                    'config': config,
                    'filename': filename
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
    """Exercice 3: Indexation XML éléments (SMART ltn)"""
    print("\n" + "=" * 70)
    print("EXERCICE 3: Indexation XML éléments (bdy, sec, p) - SMART ltn")
    print("=" * 70)
    
    generator = INEXRunGenerator()
    
    # Générer le run spécifique exercice 3
    filename = generator.generate_element_run_exercise3(
        xml_dir=XML_DIR,
        queries=INEX_QUERIES
    )
    
    print(f"\nExercice 3 terminé")
    print(f"Run généré: {filename}")
    
    return filename

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
        print("ERREUR: Dossier de données non trouvé")
        print(f"Placez les données dans: {XML_DIR}")
        return
    
    # Mapping des exercices
    exercises = {
        1: ("Indexation XML documents (SMART ltn)", exercice1),
        2: ("Test runs documents", exercice2),
        3: ("Indexation XML éléments", exercice3)
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
                print(f"Erreur exercice {ex_num}: {e}")
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
                    print(f"  {os.path.basename(result['filename'])}")
                elif isinstance(result, str):
                    print(f"  {os.path.basename(result)}")
                elif isinstance(result, list):
                    print(f"  {len(result)} configurations testées")
    
    print(f"\nTous les exercices terminés")
    print(f"Tous les fichiers sont dans: data/runs/")
    
    return results

# ==================== EXÉCUTION ====================

if __name__ == "__main__":
    import sys
    
    # Permettre de spécifier des exercices: python main.py 1 3
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
            print("Exemple: python main.py 1 3")
            main()  # Tous par défaut
    else:
        # Exécuter tous les exercices
        #main()
        print("hello")

    clean_runs_directory()
    exercice1()