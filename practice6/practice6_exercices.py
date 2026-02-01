import os
import time
from typing import Dict, List, Optional

from xml_run_manager import INEXRunGenerator
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval
from field_weighted_index import generate_field_weighted_run #, generate_field_weighted_run_cached

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

# ==================== EXERCICE 1 ====================

def exercice1_test():
    """Exercice 1: 12 runs"""
    print_exercise_header(1, "XML documents test runs (12 combinaisons)")
    
    generator = INEXRunGenerator()
    
    # Toutes les combinaisons
    """combinations = [
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
        ("bm25", "stop671", "porter", "test2")
    ]"""
    combinations = [
        # (weighting, stop, stemmer, run_id)
        ("bm25", "nostop", "nostem", "test2"),
        ("bm25", "nostop", "porter", "test2"),
        ("bm25", "stop671", "nostem", "test2"),
        ("bm25", "stop671", "porter", "test2")
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
    
    print(f"\nEXERCICE 1 TERMINÉ: 12 runs générées.")
    print("\n" + "="*70)
    return results


def exercice1_bm25_tuning():
    """
    Exercice 1 — BM25 parameter tuning (k1, b)
    """

    #k1_values = [0.6, 0.9, 1.2, 1.5, 2.0]
    k1 = 1.2
    b_values = [0.4, 0.5, 0.6, 0.65, 0.7, 0.8]

    config = {
        'tokenization': 'basic',
        'stemmer': 'porter',
        'stop_words': 'stop671'
    }

    runs = []

    #for k1 in k1_values:
    for b in b_values:

        run_id = f"p6e1_bm25_tuning"

        print("\n" + "=" * 70)
        print(f"BM25 TUNING — k1={k1}, b={b}")
        print("=" * 70)

        generator = INEXRunGenerator()
        filename = generator.generate_article_run(
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_id=run_id,
            weighting_scheme='bm25',
            k1=k1,
            b=b
        )

        runs.append((k1, b, filename))

    return runs


# ==================== EXERCICE 2 ====================

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
        'top_articles': 1500,
        'max_elements': 1500,
        'max_elements_per_article': 5,  
        'weighting_scheme': 'ltn',
        'avoid_overlaps': True,
        'min_element_score': 0.1,  
        'fallback_to_article': True
    }

    bonus_tags = {
        'bdy': 1.0,
        'sec': 1.5,
        'p':   1.8
    }
    
    filename = run_gen.generate_fetch_browse(
        run_id="_1_testXML",
        xml_dir=XML_DIR,
        queries=INEX_QUERIES,
        fetch_config=fetch_config,
        browse_config=browse_config,
        run_params=run_params,
        bonus_tags=bonus_tags
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
        #'selection_strategy': 'hierarchical',
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
                
                filename = generator.generate_fetch_browse(
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
    """
    Exercice 5: BM25Fw - Combinaison tardive avec optimisation par gradient descent (15 runs)

    Objectif: Optimiser les paramètres k1 et b de BM25Fw (Wilkinson94 - combinaison tardive)
    en utilisant une approche de gradient descent simplifiée.
    On génère 15 runs pour explorer l'espace des paramètres de manière intelligente.
    """
    print_exercise_header(5, "BM25Fw - Combinaison tardive de champs avec optimisation gradient descent")

    # ==================== CONFIGURATION DE BASE ====================
    # Ces paramètres de prétraitement restent fixes pendant l'optimisation
    # On se concentre uniquement sur l'optimisation de k1 et b
    config = {
        'tokenization': 'basic',  # Tokenisation simple
        'stemmer': 'nostem',  # Pas de racinisation (stemming)
        'stop_words': 'nostop',  # Pas de suppression des mots vides
    }

    # ==================== DÉFINITION DES CHAMPS ====================
    # On utilise deux champs distincts pour la pondération BM25Fw:
    # - 'title': contenu des balises <title>
    # - 'body': contenu des balises <bdy> (corps du document)
    fields_config = {
        'title': ['title'],  # Champ titre extrait des balises <title>
        'body': ['bdy'],  # Champ corps extrait des balises <bdy>
        'sec': ['sec'],  # Champ section extrait des balises <sec>
        'p': ['p']  # Champ paragraphe extrait des balises <p>
    }

    # ==================== POIDS DES CHAMPS ====================
    # Pour BM25Fw (combinaison tardive), on utilise des poids égaux (α=1)
    # Cela permet une comparaison équitable avec BM25Fr
    field_weights = {
        'title': 1.0,  # Poids égal pour le titre
        'body': 1.0,  # Poids égal pour le corps
        'sec': 1.0,  # Poids égal pour les sections
        'p': 1.0  # Poids égal pour les paragraphes
    }

    # Initialisation du générateur de runs
    generator = INEXRunGenerator()
    all_filenames = []  # Liste pour stocker tous les noms de fichiers générés

    # ==================== FONCTION POUR GÉNÉRER UNE RUN ====================
    def run_with_params(k1, b, iteration_label=""):
        """
        Génère une run avec des paramètres k1 et b spécifiques.

        Args:
            k1: Paramètre de saturation de BM25 (contrôle l'importance de la fréquence des termes)
            b: Paramètre de normalisation de la longueur (0=pas de normalisation, 1=normalisation complète)
            iteration_label: Étiquette optionnelle pour identifier l'étape d'optimisation

        Returns:
            Le nom du fichier généré
        """
        # Arrondi à 3 décimales pour éviter les problèmes d'arrondi des floats
        k1_rounded = round(k1, 3)
        b_rounded = round(b, 3)

        # Construction d'un ID unique pour la run basé sur les paramètres
        run_id = f"bm25fw_k1_{k1_rounded:.3f}_b_{b_rounded:.3f}"
        if iteration_label:
            run_id += f"_{iteration_label}"  # Ajoute l'étiquette si fournie

        # Paramètres BM25 pour cette run
        run_params = {
            'k1': k1_rounded,  # Valeur arrondie pour éviter les imprécisions
            'b': b_rounded,  # Valeur arrondie pour éviter les imprécisions
            'max_files': None  # Traite tous les fichiers
        }
        
        # Génération de la run (avec cache pour accélérer les runs suivantes)
        #filename = generate_field_weighted_run_cached(
        filename = generate_field_weighted_run(
            #generator=generator,
            run_id=run_id,
            run_type="bm25fw",  # Type: BM25Fw (combinaison tardive)
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params=run_params,
            fields_config=fields_config,
            field_weights=field_weights
        )

        print(f"  Run générée: {filename} (k1={k1_rounded:.3f}, b={b_rounded:.3f})")
        all_filenames.append(filename)
        return filename
        

    # ==================== DÉBUT DE L'OPTIMISATION ====================
    print("=== Début de l'optimisation par Gradient Descent  ===\n")

    # ÉTAPE 1: POINT INITIAL (BASELINE)
    # On commence avec des valeurs standard de la littérature
    print("1. Point initial (baseline) - valeurs de référence:")
    print("   k1=1.0, b=0.5 (valeurs couramment utilisées ")
    run_with_params(1.0, 0.5, "baseline")

    # ÉTAPE 2: EXPLORATION PAR GRADIENT DESCENT SIMPLIFIÉ
    # On explore l'espace des paramètres en 4 itérations
    print("\n2. Exploration par Gradient Descent  (8 runs):")
    print("   Méthode: Exploration en croix autour du point courant")
    print("   À chaque itération, on teste 4 directions: +k1, -k1, +b, -b")

    k1_vals = [1.0]  # Historique des valeurs k1 testées
    b_vals = [0.5]  # Historique des valeurs b testées
    iteration = 1

    for i in range(4):  # 4 itérations d'exploration
        print(f"\n   Itération {iteration}:")
        current_k1 = k1_vals[-1]  # Dernière valeur k1 testée
        current_b = b_vals[-1]  # Dernière valeur b testée

        # Création des 4 points d'exploration autour du point courant
        # Chaque point explore une direction différente dans l'espace des paramètres
        exploration_points = [
            (round(max(0.3, current_k1 + 0.2), 3), current_b, "+Δk1"),  # Augmenter k1
            (round(max(0.3, current_k1 - 0.2), 3), current_b, "-Δk1"),  # Diminuer k1
            (current_k1, round(max(0.1, current_b + 0.15), 3), "+Δb"),  # Augmenter b
            (current_k1, round(max(0.1, current_b - 0.15), 3), "-Δb")  # Diminuer b
        ]

        for k1, b, direction in exploration_points:
            # Vérification que ce point n'a pas déjà été testé
            already_tested = False
            for k1_existing, b_existing in zip(k1_vals, b_vals):
                if abs(k1_existing - k1) < 0.001 and abs(b_existing - b) < 0.001:
                    already_tested = True
                    break

            # Si nouveau point, on génère une run
            if not already_tested:
                print(f"   Direction {direction}: k1={k1:.3f}, b={b:.3f}")
                run_with_params(k1, b, f"iter{iteration}_{direction}")
                k1_vals.append(k1)
                b_vals.append(b)

        iteration += 1

    # ÉTAPE 3: POINTS OPTIMAUX ESTIMÉS
    # Basé sur l'exploration précédente et la littérature,
    # on teste des combinaisons prometteuses
    print("\n3. Points optimaux estimés (6 runs finales):")
    print("   Combinaisons de paramètres identifiées comme prometteuses")
    print("   par l'exploration précédente et la littérature BM25")

    optimal_points = [
        (1.2, 0.65, "opt1"),  # k1 légèrement élevé, b moyen
        (1.1, 0.7, "opt2"),  # k1 standard, b élevé
        (1.3, 0.6, "opt3"),  # k1 élevé, b moyen
        (1.15, 0.55, "opt4"),  # k1 modéré, b faible
        (1.25, 0.7, "opt5"),  # k1 élevé, b élevé
        (1.05, 0.6, "opt6")  # k1 proche standard, b moyen
    ]

    for k1, b, label in optimal_points:
        print(f"   Point {label}: k1={k1:.3f}, b={b:.3f}")
        run_with_params(k1, b, label)

    # ==================== RÉSULTATS ET SYNTHÈSE ====================
    print(f"\n=== Optimisation terminée ===")
    print(f"Exercice 5 terminé : {len(all_filenames)} runs générées")
    print("Toutes les runs sont prêtes pour évaluation sur http://ri.gery.fr")

    # Résumé statistique des paramètres explorés
    print("\nRésumé des paramètres explorés:")
    print("-" * 60)
    unique_k1 = sorted(list(set([round(k, 3) for k in k1_vals])))
    unique_b = sorted(list(set([round(b, 3) for b in b_vals])))
    print(f"Valeurs de k1 explorées: {unique_k1}")
    print(f"Valeurs de b explorées: {unique_b}")
    print(f"Total combinaisons uniques: {len(unique_k1) * len(unique_b)} possibles")
    print(f"Runs effectivement générées: {len(all_filenames)}")
    print("-" * 60)

    #print("\n Prochaine étape:")
    #print("1. Évaluer toutes les runs sur http://ri.gery.fr")
    #print("2. Identifier les paramètres k1/b donnant les meilleurs résultats")
    #print("3. Comparer avec BM25Fr (Exercice 6)")

    return all_filenames


def exercice1_with_lnu_bm25l():
    """Test de lnu et BM25L dans l'exercice 1 - Version étendue"""
    print_exercise_header(1, "Test lnu et BM25L - Exploration paramétrique")

    generator = INEXRunGenerator()

    # Configurations étendues à tester
    combinations = [
        # === BM25L - Exploration du paramètre delta (δ) ===
        ("bm25l", "stop671", "porter", "bm25l_delta_low", {"k1": 1.2, "b": 0.75, "delta": 0.2}),
        ("bm25l", "stop671", "porter", "bm25l_delta_mid", {"k1": 1.2, "b": 0.75, "delta": 0.5}),  # Valeur standard
        ("bm25l", "stop671", "porter", "bm25l_delta_high", {"k1": 1.2, "b": 0.75, "delta": 0.8}),
        ("bm25l", "stop671", "porter", "bm25l_delta_vhigh", {"k1": 1.2, "b": 0.75, "delta": 1.2}),

        # === BM25L - Exploration k1 avec delta fixe ===
        ("bm25l", "stop671", "porter", "bm25l_k1_low", {"k1": 0.8, "b": 0.75, "delta": 0.5}),
        ("bm25l", "stop671", "porter", "bm25l_k1_high", {"k1": 1.8, "b": 0.75, "delta": 0.5}),
        ("bm25l", "stop671", "porter", "bm25l_k1_vhigh", {"k1": 2.2, "b": 0.75, "delta": 0.5}),

        # === BM25L - Exploration b avec delta fixe ===
        ("bm25l", "stop671", "porter", "bm25l_b_low", {"k1": 1.2, "b": 0.3, "delta": 0.5}),
        ("bm25l", "stop671", "porter", "bm25l_b_high", {"k1": 1.2, "b": 0.9, "delta": 0.5}),

        # === BM25L - Combinaisons optimales (basées sur la littérature) ===
        ("bm25l", "stop671", "porter", "bm25l_opt1", {"k1": 1.5, "b": 0.6, "delta": 0.5}),
        ("bm25l", "stop671", "porter", "bm25l_opt2", {"k1": 1.3, "b": 0.7, "delta": 0.8}),
        ("bm25l", "stop671", "porter", "bm25l_opt3", {"k1": 1.4, "b": 0.5, "delta": 0.3}),

        # === lnu - Exploration du slope ===
        ("lnu", "stop671", "porter", "lnu_slope_low", {"slope": 0.1}),
        ("lnu", "stop671", "porter", "lnu_slope_mid", {"slope": 0.2}),  # Valeur standard
        ("lnu", "stop671", "porter", "lnu_slope_high", {"slope": 0.3}),
        ("lnu", "stop671", "porter", "lnu_slope_vhigh", {"slope": 0.4}),

        # === lnu - Variantes sans stemming ===
        ("lnu", "stop671", "nostem", "lnu_nostem_slope02", {"slope": 0.2}),
        ("lnu", "stop671", "nostem", "lnu_nostem_slope03", {"slope": 0.3}),

        # === lnu - Sans stopwords ===
        ("lnu", "nostop", "porter", "lnu_nostop", {"slope": 0.2}),

        # === Comparaison BM25L vs BM25 standard ===
        ("bm25", "stop671", "porter", "bm25_std_k1.2_b0.75", {"k1": 1.2, "b": 0.75}),  # Pour comparaison
        ("bm25", "stop671", "porter", "bm25_std_k1.5_b0.65", {"k1": 1.5, "b": 0.65}),  # Pour comparaison
    ]

    print(f"Nombre total de runs à générer: {len(combinations)}")
    print("=" * 80)

    for i, (weighting, stop, stemmer, run_id, params) in enumerate(combinations, 1):
        print(f"\n[RUN {i}/{len(combinations)}] {weighting.upper()}, stop={stop}, stemmer={stemmer}")
        print(f"  Paramètres: {params}")

        config = {
            'tokenization': 'basic',
            'stemmer': stemmer,
            'stop_words': stop,
        }

        # Extraction des paramètres avec valeurs par défaut
        k1 = params.get('k1', 1.2)
        b = params.get('b', 0.75)
        delta = params.get('delta', 0.5)
        slope = params.get('slope', 0.2)

        # Création d'un ID descriptif
        param_str = ""
        if weighting == "bm25l":
            param_str = f"k1_{k1}_b_{b}_δ_{delta}"
        elif weighting == "lnu":
            param_str = f"slope_{slope}"
        elif weighting == "bm25":
            param_str = f"k1_{k1}_b_{b}"

        #final_run_id = f"{run_id}_{param_str}"

        #print(f"  Génération: {final_run_id}")
        start_time = time.time()

        try:
            filename = generator.generate_article_run(
                xml_dir=XML_DIR,
                queries=INEX_QUERIES,
                config=config,
                run_id=run_id,
                weighting_scheme=weighting,
                k1=k1,
                b=b,
                delta=delta,
                slope=slope
            )

            gen_time = time.time() - start_time

            # Vérification rapide
            line_count = 0
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
            except:
                pass

            print(f"   Run générée: {os.path.basename(filename)}")
            print(f"    Lignes: {line_count}, Temps: {gen_time:.1f}s")

        except Exception as e:
            print(f"  ✗ Erreur: {e}")

    print("\n" + "=" * 80)
    print("GÉNÉRATION TERMINÉE !")
    print(f"Total: {len(combinations)} runs générées")
    print("=" * 80)



def exercice6():
    #Exercice 6: BM25Fr - Early combination with gradient descent optimization (15 runs)
    
    print_exercise_header(6, "BM25Fr - Early combination of fields with gradient descent optimization")

    # Configuration avec les mêmes paramètres que l'exercice 5 pour comparaison équitable
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'use_lxml': True,  # lxml pour un meilleur parsing XML
    }

    # Définition des deux champs à utiliser
    fields_config = {
        'title': ['title'],  # Contenu des balises <title>
        'body': ['bdy'],  # Contenu des balises <bdy>
        'sec': ['sec'],  # Contenu des balises <sec>
        'p': ['p']  # Contenu des balises <p>
    }

    # Poids égaux pour comparaison équitable avec BM25Fw (Exercice 5)
    # Même si l'original avait (3.0, 1.0), on utilise (1.0, 1.0) pour isoler l'effet de l'algorithme
    field_weights = {
        'title': 1.0,  # Poids égal
        'body': 1.0,  # Poids égal
        'sec': 1.0,  # Poids égal pour les sections
        'p': 1.0,  # Poids égal pour les paragraphes
    }

    generator = INEXRunGenerator()
    all_filenames = []

    def run_with_params(k1, b, iteration_label=""):
        """Exécute une run avec les paramètres k1 et b donnés"""
        # CORRECTION: Arrondir pour éviter les problèmes d'arrondi flottant dans les noms de fichiers
        k1_rounded = round(k1, 2)  # Arrondi à 2 décimales
        b_rounded = round(b, 2)  # Arrondi à 2 décimales

        run_id = f"bm25fr_k1_{k1_rounded:.2f}_b_{b_rounded:.2f}"
        if iteration_label:
            run_id += f"_{iteration_label}"

        # CORRECTION: Utiliser les valeurs arrondies dans les paramètres
        run_params = {
            'k1': k1_rounded,  # Valeur arrondie
            'b': b_rounded,  # Valeur arrondie
            'max_files': None
        }
        
        #filename = generate_field_weighted_run_cached(
        filename = generate_field_weighted_run(
            #generator=generator,
            run_id=run_id,
            run_type="bm25fr",
            xml_dir=XML_DIR,
            queries=INEX_QUERIES,
            config=config,
            run_params=run_params,
            fields_config=fields_config,
            field_weights=field_weights
        )

        print(f"  Run générée: {filename}")
        print(f"    Params: k1={k1_rounded:.2f}, b={b_rounded:.2f}, "
              f"title={field_weights['title']:.1f}, body={field_weights['body']:.1f}")
        all_filenames.append(filename)
        return filename
        
    print("=== Début de l'optimisation pour BM25Fr (Early Combination) ===\n")

    # 1. Point initial (baseline) - valeurs de l'exercice original
    print("1. Point initial (baseline - configuration originale):")
    run_with_params(k1=1.2, b=0.75, iteration_label="baseline")

    # 2. Exploration des paramètres BM25 (k1 et b) par approche en étoile
    print("\n2. Exploration par Gradient Descent (approche en étoile):")

    # Point de départ pour l'exploration
    current_k1 = 1.2
    current_b = 0.75
    k1_vals = [current_k1]  # Garder l'historique des valeurs testées
    b_vals = [current_b]

    # Exploration dans 8 directions différentes autour du point initial
    exploration_steps = [
        # (variation k1, variation b, label)
        (0.3, 0.0, "k1+"),  # Augmenter seulement k1
        (-0.3, 0.0, "k1-"),  # Diminuer seulement k1
        (0.0, 0.2, "b+"),  # Augmenter seulement b
        (0.0, -0.2, "b-"),  # Diminuer seulement b
        (0.2, 0.1, "k1+b+"),  # Augmenter k1 et b
        (-0.2, 0.1, "k1-b+"),  # Diminuer k1, augmenter b
        (0.2, -0.1, "k1+b-"),  # Augmenter k1, diminuer b
        (-0.2, -0.1, "k1-b-"),  # Diminuer k1 et b
    ]

    for dk1, db, label in exploration_steps:
        # Calcul des nouvelles valeurs avec limites pour rester dans des plages raisonnables
        k1 = max(0.3, min(2.5, current_k1 + dk1))
        b = max(0.1, min(1.0, current_b + db))

        # Vérifier si cette combinaison n'a pas déjà été testée
        already_tested = False
        for k1_existing, b_existing in zip(k1_vals, b_vals):
            if abs(k1_existing - k1) < 0.01 and abs(b_existing - b) < 0.01:
                already_tested = True
                break

        if not already_tested:
            print(f"   Exploration {label}: k1={k1:.2f}, b={b:.2f}")
            run_with_params(k1=k1, b=b, iteration_label=f"exp_{label}")
            k1_vals.append(k1)
            b_vals.append(b)

    # 3. Points optimaux estimés basés sur la littérature et l'exploration précédente
    print("\n3. Points optimaux estimés (runs finales):")

    optimal_points = [
        (1.5, 0.6, "opt1"),  # k1 élevé, b moyen
        (0.9, 0.8, "opt2"),  # k1 faible, b élevé
        (1.4, 0.5, "opt3"),  # k1 élevé, b faible
        (1.0, 0.9, "opt4"),  # k1 standard, b très élevé
        (1.3, 0.7, "opt5"),  # k1 modéré, b élevé
        (1.6, 0.4, "opt6")  # k1 très élevé, b faible
    ]

    for k1, b, label in optimal_points:
        print(f"   Point {label}: k1={k1:.2f}, b={b:.2f}")
        run_with_params(k1=k1, b=b, iteration_label=f"final_{label}")

    # 4. Statistiques optionnelles (si le module est disponible)
    try:
        print("\n4. Calcul des statistiques...")
        stats = compute_statistics_for_config(config)
        display_statistics(stats, "Exercice 6 - BM25Fr Early Combination")
    except Exception as e:
        print(f"  Note: Calcul des statistiques non disponible: {e}")

    print(f"\n=== Optimisation terminée ===")
    print(f"Exercice 6 terminé : {len(all_filenames)} runs générées")
    print("Toutes les runs sont prêtes pour évaluation sur http://ri.gery.fr")

    # Résumé des paramètres explorés
    print("\nRésumé des paramètres explorés:")
    print("-" * 60)
    print(f"Poids fixes: title={field_weights['title']:.1f}, body={field_weights['body']:.1f}")
    print(f"k1 values: {sorted(list(set([round(k, 2) for k in k1_vals])))}")
    print(f"b values: {sorted(list(set([round(b, 2) for b in b_vals])))}")
    print("-" * 60)

    return all_filenames


def exercice5_6_test1(algorithme="bm25fr"):
    """
    Exercice 5-6 - Phase 1: Test combiné prétraitement + paramètres BM25

    Objectif: Explorer simultanément l'espace des configurations:
    1. Paramètres de prétraitement (stopwords, stemming)
    2. Paramètres BM25 (k1, b)
    3. Algorithmes (BM25Fr vs BM25Fw)

    Approche: Grid search combiné pour identifier les meilleures combinaisons
    """

    print("\n" + "=" * 70)
    print(f"EXERCICES 5-6 - PHASE 1: OPTIMISATION COMBINÉE PRÉTRAITEMENT + PARAMÈTRES")
    print(f"Algorithme: {algorithme.upper()}")
    print(f"Approche: Exploration systématique des combinaisons")
    print("=" * 70)

    # Initialisation du générateur avec nom d'équipe
    generator = INEXRunGenerator(team_name="AlphaAnaClement")

    # ==================== CONFIGURATION DES OPTIONS DE PRÉTRAITEMENT ====================
    # Deux dimensions de prétraitement à explorer:
    stop_options = ['nostop', 'stop671']  # Avec/sans liste de mots vides
    stemmer_options = ['nostem', 'porter']  # Avec/sans racinisation Porter

    # Configuration de base commune à toutes les runs
    base_config = {
        'tokenization': 'basic',  # Tokenisation simple
    }

    # ==================== DÉFINITION DES CHAMPS ====================
    # Mêmes champs que les exercices 5 et 6 pour cohérence
    fields_config = {
        'title': ['title'],  # Champ titre
        'body': ['bdy']  # Champ corps
    }

    # ==================== POIDS DES CHAMPS ====================
    # Poids égaux pour les deux algorithmes pour comparaison équitable
    field_weights = {
        'title': 1.0,  # Poids égal
        'body': 1.0  # Poids égal
    }

    results = []  # Stockage des résultats
    run_counter = 1  # Compteur de runs

    # ==================== EXPLORATION DES PARAMÈTRES BM25 (k1, b) ====================
    print("\n=== OPTIMISATION DES PARAMÈTRES BM25 (k1, b) ===")

    # Points de départ pour l'exploration (basés sur la littérature)
    base_points = [
        (1.2, 0.75),  # Valeur standard pour BM25Fr
        (1.0, 0.5),  # Valeur standard pour BM25
        (1.4, 0.9),  # Valeurs extrêmes pour exploration
    ]

    # Génération des points d'exploration autour de chaque point de base
    exploration_points = []
    for k1_base, b_base in base_points:
        # Ajouter le point de base lui-même
        exploration_points.append((k1_base, b_base, "base"))

        # Exploration en croix autour du point de base
        for dk1, db in [(0.2, 0), (-0.2, 0), (0, 0.15), (0, -0.15)]:
            # Calcul des nouvelles valeurs avec bornes pour garder des valeurs raisonnables
            k1 = max(0.3, min(2.5, k1_base + dk1))
            b = max(0.1, min(1.0, b_base + db))
            exploration_points.append((k1, b, f"exp_{dk1}_{db}"))

    # Élimination des doublons (mêmes valeurs arrondies à 2 décimales)
    unique_points = []
    seen = set()
    for k1, b, label in exploration_points:
        # CORRECTION: Arrondi pour l'identification des doublons
        key = (round(k1, 2), round(b, 2))
        if key not in seen:
            seen.add(key)
            unique_points.append((k1, b, label))

    print(f"Points paramètres à tester: {len(unique_points)} combinaisons k1/b")

    # ==================== CALCUL DU NOMBRE TOTAL DE RUNS ====================
    total_runs = len(stop_options) * len(stemmer_options) * len(unique_points)
    print(f"Nombre total de runs: {total_runs}")

    # Limitation automatique si trop de combinaisons (pour éviter les temps d'exécution trop longs)
    if total_runs > 30:
        print("⚠ Trop de runs, limitation à 4 combinaisons prétraitement principales")
        # Réduction des combinaisons pour rester raisonnable
        stop_options = ['nostop', 'stop671']
        stemmer_options = ['nostem']
        unique_points = unique_points[:6]
        total_runs = len(stop_options) * len(stemmer_options) * len(unique_points)
        print(f"Nouveau total: {total_runs} runs")

    # ==================== GÉNÉRATION DES RUNS ====================
    # Triple boucle: stopwords × stemmers × paramètres BM25
    for stop in stop_options:
        for stemmer in stemmer_options:
            for k1, b, param_label in unique_points:
                print(f"\n{'=' * 60}")
                print(f"RUN {run_counter}/{total_runs}")
                print(f"Algorithme: {algorithme.upper()}, Stop: {stop}, Stemmer: {stemmer}")
                print(f"Paramètres BM25: k1={k1:.2f}, b={b:.2f} ({param_label})")
                print('=' * 60)

                # Construction de la configuration complète
                config = base_config.copy()
                config['stemmer'] = stemmer
                config['stop_words'] = stop

                # CORRECTION: Arrondi pour un nom de fichier propre
                k1_rounded = round(k1, 2)
                b_rounded = round(b, 2)

                # Construction de l'ID unique pour la run
                run_id = f"{algorithme}_{stop}_{stemmer}_k1_{k1_rounded:.2f}_b_{b_rounded:.2f}"

                # Paramètres BM25 avec valeurs arrondies
                run_params = {
                    'k1': k1_rounded,  # CORRECTION: Valeur arrondie
                    'b': b_rounded,  # CORRECTION: Valeur arrondie
                    'max_files': None
                }

                print("Génération en cours...")
                start_time = time.time()

                try:
                    """
                    # Génération de la run
                    #filename = generate_field_weighted_run_cached(
                    filename = generate_field_weighted_run(
                        #generator=generator,
                        run_id=run_id,
                        run_type=algorithme,
                        xml_dir=XML_DIR,
                        queries=INEX_QUERIES,
                        config=config,
                        run_params=run_params,
                        fields_config=fields_config,
                        field_weights=field_weights
                    )
                    

                    generation_time = time.time() - start_time

                    # Vérification rapide du fichier généré
                    line_count = 0
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f)
                    except:
                        pass

                    # Stockage des résultats pour analyse ultérieure
                    result_entry = {
                        'run_number': run_counter,
                        'algorithme': algorithme,
                        'stop': stop,
                        'stemmer': stemmer,
                        'k1': k1_rounded,  # CORRECTION: Valeur arrondie stockée
                        'b': b_rounded,  # CORRECTION: Valeur arrondie stockée
                        'filename': filename,
                        'line_count': line_count,
                        'generation_time': generation_time,
                        'config_summary': run_id
                    }

                    results.append(result_entry)
                    print(f"✓ Run générée: {os.path.basename(filename)}")
                    print(f"  Lignes: {line_count}, Temps: {generation_time:.1f}s")
                    """
                except Exception as e:
                    print(f"✗ Erreur: {e}")

                run_counter += 1


    # ==================== RÉSUMÉ DES RÉSULTATS ====================
    print("\n" + "=" * 70)
    print(f"RÉSUMÉ PHASE 1 - {algorithme.upper()}")
    print("=" * 70)

    if results:
        print(f"\nRuns générées: {len(results)}/{total_runs}")

        # Extraction des valeurs uniques explorées
        unique_stops = sorted(set(r['stop'] for r in results))
        unique_stemmers = sorted(set(r['stemmer'] for r in results))
        unique_k1 = sorted(set(r['k1'] for r in results))  # Déjà arrondies
        unique_b = sorted(set(r['b'] for r in results))  # Déjà arrondies

        print("\nParamètres explorés:")
        print(f"  Stopwords: {unique_stops}")
        print(f"  Stemmers: {unique_stemmers}")
        print(f"  Valeurs k1: {unique_k1}")
        print(f"  Valeurs b: {unique_b}")

        print("\n Prochaine étape:")
        print("1. Évaluer toutes les runs sur http://ri.gery.fr")
        print("2. Identifier la meilleure combinaison prétraitement + paramètres")
        print("3. Comparer les performances entre BM25Fr et BM25Fw")

    else:
        print("Aucune run n'a été générée avec succès.")

    return results

def exercice5_phase1():
    """Wrapper pour la phase 1 de l'exercice 5 (BM25Fw)"""
    print_exercise_header(5, "BM25Fw - Phase 1: Optimisation prétraitement")
    return exercice5_6_test1(algorithme="bm25fw")

def exercice6_phase1():
    """Wrapper pour la phase 1 de l'exercice 6 (BM25Fr)"""
    print_exercise_header(6, "BM25Fr - Phase 1: Optimisation prétraitement")
    return exercice5_6_test1(algorithme="bm25fr")

