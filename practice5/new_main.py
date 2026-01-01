import os
import time
from new_xml_rm import INEXRunGenerator

def generate_exercise3():
    """Génère le run spécifique pour l'exercice 3"""
    
    # Configuration
    XML_DIR = "data/Practice_05_data/XML-Coll-withSem"
    QUERIES = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors", 
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm", 
        2009085: "operating system mutual exclusion"
    }
    
    generator = INEXRunGenerator()
    
    # Configuration POUR L'EXERCICE 3 (éléments seulement)
    browse_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'target_tags': ['bdy', 'sec', 'p']
    }
    
    # Paramètres
    run_params = {
        'top_articles': 1500,
        'score_threshold': 0.00001,
        'max_elements': 1500,
        'weighting_scheme': 'ltn'  # SMART lm
    }
    
    print("=" * 70)
    print("EXERCICE 3: Indexation des éléments XML")
    print("=" * 70)
    
    # Générer le run avec run_id spécial pour exercice 3
    filename = generator.generate_fetch_browse_run(
        run_id="testXML",  # Important: contient "testXML"
        xml_dir=XML_DIR,
        queries=QUERIES,
        fetch_config=browse_config,  # Même config pour fetch et browse
        browse_config=browse_config,
        run_params=run_params
    )
    
    print(f"\nRun généré pour l'exercice 3: {filename}")
    
    # Vérifier le fichier
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Nombre de lignes: {len(lines)}")
        print(f"Première ligne: {lines[0].strip()}")
        print(f"Dernière ligne: {lines[-1].strip()}")
    
    return filename


def main():
    # Configuration
    XML_DIR = "data/Practice_05_data/XML-Coll-withSem"
    QUERIES = {
        2009011: "olive oil health benefit",
        2009036: "notting hill film actors",
        2009067: "probabilistic models in information retrieval",
        2009073: "web link network analysis",
        2009074: "web ranking scoring algorithm",
        2009078: "supervised machine learning algorithm",
        2009085: "operating system mutual exclusion"
    }
    
    # Initialiser le générateur
    generator = INEXRunGenerator()
    
    # Configuration FETCH (articles)
    fetch_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop'
    }
    
    # Configuration BROWSE (éléments)
    browse_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'target_tags': ['article', 'sec', 'bdy', 'p']
    }
    
    # Paramètres du run
    run_params = {
        'top_articles': 1500,
        'score_threshold': 0.1,
        'max_elements': 1500,
        'weighting_scheme': 'ltn',
        'min_element_score': 0.1
    }
    
    print("=" * 70)
    print("SYSTÈME INEX - Fetch and Browse")
    print("=" * 70)
    
    # Générer plusieurs runs avec différentes configurations
    
    runs = [
        ('FB_LTN_BASIC', {'weighting_scheme': 'ltn'}),
        ('FB_LTC', {'weighting_scheme': 'ltc'}),
        #('FB_BM25_STD', {'weighting_scheme': 'bm25', 'k1': 1.2, 'b': 0.75, 'score_threshold': 0.0}),
        #('FB_BM25_OPT', {'weighting_scheme': 'bm25', 'k1': 2.0, 'b': 0.9, 'score_threshold': 0.0})
    ]
    
    for run_id, params in runs:
        print(f"\n\n{'#'*70}")
        print(f"GÉNÉRATION DU RUN: {run_id}")
        print(f"{'#'*70}")
        
        # Fusionner les paramètres
        current_params = run_params.copy()
        current_params.update(params)
        
        # Générer le run
        filename = generator.generate_fetch_browse_run(
            run_id=run_id,
            xml_dir=XML_DIR,
            queries=QUERIES,
            fetch_config=fetch_config,
            browse_config=browse_config,
            run_params=current_params
        )
        
        print(f"Run généré: {filename}")
    
    print("\n" + "="*70)
    print("TOUS LES RUNS ONT ÉTÉ GÉNÉRÉS AVEC SUCCÈS")
    print("="*70)
    
    #generator.test_overlap_logic()


if __name__ == "__main__":
        
    main()