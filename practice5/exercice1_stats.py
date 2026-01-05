import os
import time
from xml_run_manager import INEXRunGenerator
from indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

def compute_statistics(exercise_num, index_data, weighting_scheme="ltn", 
                      k1=1.2, b=0.75, target_doc_id="23724", 
                      target_term="ranking", 
                      test_query="web ranking scoring algorithm"):
    """Fonction qui utilise un index pré-construit - VOTRE VERSION"""
    
    # Extraction des données de configuration
    index = index_data['index']
    indexing_time = index_data['indexing_time']
    stats = index_data['stats']
    config = index_data['config']
    
    # Construction de la description
    config_desc = f"{weighting_scheme.upper()}"
    if config['stop_words'] != "nostop":
        config_desc += f" + stop-words({config['stop_words']})"
    if config['stemmer'] != "nostem":
        config_desc += f" + stemming({config['stemmer']})"
    if config['tokenization'] != "basic":
        config_desc += f" + tokenization({config['tokenization']})"
    
    if weighting_scheme == "bm25":
        config_desc += f" - k1={k1}, b={b}"

    print("\n" + "=" * 60)
    print(f"EXERCICE {exercise_num}: {config_desc}")
    print("=" * 60)

    start_total_time = time.time()
    
    # Création du ranker
    ranker = RankedRetrieval(index)

    # Nettoyer le cache précédent si nécessaire
    if weighting_scheme == "ltc":
        ranker.clear_cosine_norms_cache()
    
    # Calcul du poids du terme cible dans le document cible
    processed_terms = ranker.process_query_terms(target_term)
    target_weight = ranker.get_term_weight(
        processed_terms[0], target_doc_id, weighting_scheme, k1=k1, b=b
    ) if processed_terms else 0.0
    
    # Calcul du RSV pour la requête test
    query_terms = ranker.process_query_terms(test_query)
    doc_score = sum(
        ranker.get_term_weight(t, target_doc_id, weighting_scheme, k1=k1, b=b) 
        for t in query_terms
    )
    
    # Recherche du top-10
    top_docs = ranker.search_query(
        test_query, weighting_scheme, top_k=10, k1=k1, b=b
    )
    
    # Temps total (indexation + pondération)
    weighting_time = time.time() - start_total_time
    total_time = indexing_time + weighting_time

    # Affichage des statistiques complètes
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

def exercice1_complete():
    """Exercice 1 complet avec compute_statistics"""
    print("=" * 70)
    print("EXERCICE 1: Indexation XML documents (SMART lm)")
    print("=" * 70)
    
    # 1. Créer l'index avec la même configuration que Practice 4
    print("\n[Étape 1/3] Création de l'index...")
    index = WeightedInvertedIndex()
    index.configure(
        tokenization="basic",
        stemmer="nostem",
        stop_words="nostop",
        use_lxml=True
    )
    
    # Indexer et obtenir les données pour compute_statistics
    index_data = index.build_index_with_stats(
        xml_dir="data/Practice_05_data/XML-Coll-withSem",
        max_files=None
    )
    
    # 2. Calculer et afficher les statistiques avec LTN
    print("\n[Étape 2/3] Calcul des statistiques LTN...")
    ranker_ltn = compute_statistics(
        exercise_num=1,
        index_data=index_data,
        weighting_scheme="ltn",  # Ou "lm" selon spécification
        target_doc_id="23724",
        target_term="ranking",
        test_query="web ranking scoring algorithm"
    )
    
    # 3. Générer le run INEX
    print("\n[Étape 3/3] Génération du run INEX...")
    generator = INEXRunGenerator()
    
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'test_type': 'test1'
    }
    
    filename = generator.generate_article_run(
        run_id="1",
        xml_dir="data/Practice_05_data/XML-Coll-withSem",
        queries=queries,
        config=config,
        weighting_scheme="lm"  # SMART lm pour l'exercice
    )
    
    # 4. Validation
    generator.validate_run_file(filename)
    
    print(f"\n✅ Exercice 1 terminé")
    print(f"📊 Statistiques calculées")
    print(f"📁 Run généré: {filename}")
    
    return {
        'index_data': index_data,
        'ranker': ranker_ltn,
        'filename': filename
    }

def exercice2_complete():
    """Exercice 2 avec compute_statistics pour chaque configuration"""
    print("\n" + "=" * 70)
    print("EXERCICE 2: XML documents test runs (12 combinaisons)")
    print("=" * 70)
    
    generator = INEXRunGenerator()
    
    # Requêtes INEX
    queries = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    # Configurations pour 12 runs
    weighting_schemes = ["lm", "ltc", "bm25"]  # 3 weighting schemes
    stop_options = ["nostop", "stop671"]        # 2 options stop-words
    stemmer_options = ["nostem", "porter"]      # 2 options stemming
    
    all_results = []
    run_counter = 1
    
    for weighting in weighting_schemes:
        for stop in stop_options:
            for stemmer in stemmer_options:
                print(f"\n{'='*50}")
                print(f"CONFIGURATION {run_counter}/12: {weighting.upper()}, {stop}, {stemmer}")
                print('='*50)
                
                # 1. Créer l'index pour cette configuration
                index = WeightedInvertedIndex()
                index.configure(
                    tokenization="basic",
                    stemmer=stemmer,
                    stop_words=stop,
                    use_lxml=True
                )
                
                index_data = index.build_index_with_stats(
                    xml_dir="data/Practice_05_data/XML-Coll-withSem",
                    max_files=None
                )
                
                # 2. Calculer statistiques
                ranker = compute_statistics(
                    exercise_num=2,
                    index_data=index_data,
                    weighting_scheme=weighting,
                    k1=1.2 if weighting == "bm25" else None,
                    b=0.75 if weighting == "bm25" else None,
                    target_doc_id="23724",
                    target_term="ranking",
                    test_query="web ranking scoring algorithm"
                )
                
                # 3. Générer run
                config = {
                    'tokenization': 'basic',
                    'stemmer': stemmer,
                    'stop_words': stop,
                    'test_type': 'test2'
                }
                
                k1, b = (1.2, 0.75) if weighting == "bm25" else (None, None)
                
                filename = generator.generate_article_run(
                    run_id=str(run_counter),
                    xml_dir="data/Practice_05_data/XML-Coll-withSem",
                    queries=queries,
                    config=config,
                    weighting_scheme=weighting,
                    k1=k1,
                    b=b
                )
                
                all_results.append({
                    'config_num': run_counter,
                    'weighting': weighting,
                    'stop_words': stop,
                    'stemmer': stemmer,
                    'filename': filename,
                    'ranker': ranker
                })
                
                run_counter += 1
    
    print(f"\n✅ {len(all_results)} configurations traitées pour l'exercice 2")
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ EXERCICE 2")
    print("="*70)
    for result in all_results:
        print(f"{result['config_num']:2d}. {result['weighting'].upper():4s} | "
              f"stop={result['stop_words']:8s} | "
              f"stem={result['stemmer']:7s} | "
              f"{os.path.basename(result['filename'])}")
    
    return all_results

def main():
    """Fonction principale avec tous les exercices"""
    
    # Nettoyer les anciens runs
    if os.path.exists("data/runs"):
        response = input("\nNettoyer le dossier 'data/runs' ? (o/n): ")
        if response.lower() == 'o':
            for file in os.listdir("data/runs"):
                if file.endswith(".txt"):
                    os.remove(os.path.join("data/runs", file))
            print("✅ Dossier 'runs' nettoyé")
    
    print("=" * 70)
    print("PRACTICAL SESSION 5: Structured IR at INEX")
    print("=" * 70)
    
    all_results = {}
    
    # Exercice 1
    print("\n" + "="*70)
    print("LANCEMENT EXERCICE 1")
    print("="*70)
    result1 = exercice1_complete()
    all_results['ex1'] = result1
    
    # Exercice 2  
    print("\n" + "="*70)
    print("LANCEMENT EXERCICE 2")
    print("="*70)
    result2 = exercice2_complete()
    all_results['ex2'] = result2
    
    # Pour les exercices 3-6, utiliser les méthodes du xml_run_manager
    # (garder votre code existant pour exercice3, exercice4, etc.)
    
    return all_results

if __name__ == "__main__":
    # Vérifier que le dossier data existe
    if not os.path.exists("data/Practice_05_data"):
        print("❌ ERREUR: Dossier 'data/Practice_05_data' non trouvé")
        print("Téléchargez le fichier Practice_05_data.zip depuis le site du cours")
    else:
        main()