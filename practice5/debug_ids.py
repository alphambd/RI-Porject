# generate_exercise3_final.py
import os
import time
from new_xml_rm import INEXRunGenerator

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
    
    generator = INEXRunGenerator()
    
    # Configuration POUR L'EXERCICE 3
    # IMPORTANT: Inclure 'article' dans les target_tags
    browse_config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'target_tags': ['article', 'bdy', 'sec', 'p']  # AJOUT DE 'article'
    }
    
    # Même config pour fetch (mais on utilisera seulement browse)
    fetch_config = browse_config.copy()
    
    # Paramètres optimisés pour 1500 résultats
    run_params = {
        'top_articles': 1600,  # Un peu plus pour compenser
        'score_threshold': 0.0,  # Prendre tout
        'max_elements': 1500,
        'weighting_scheme': 'ltn',
        'min_element_score': 0.00001  # Très bas pour inclure plus d'éléments
    }
    
    print("=" * 70)
    print("EXERCICE 3 - Version finale avec articles entiers")
    print("=" * 70)
    
    # Générer le run
    filename = generator.generate_fetch_browse_run(
        run_id="testXML_FINAL",
        xml_dir=XML_DIR,
        queries=QUERIES,
        fetch_config=fetch_config,
        browse_config=browse_config,
        run_params=run_params
    )
    
    print(f"\n✅ Run généré: {filename}")
    
    # Vérification rapide
    print("\n[Vérification du fichier]")
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"Lignes totales: {len(lines)}")
        print(f"Lignes attendues: {7 * 1500} = 10,500")
        
        # Compter les /article[1]
        article_only = sum(1 for line in lines if '/article[1]\n' in line)
        print(f"Articles entiers (/article[1]): {article_only}")
        print(f"Sous-éléments: {len(lines) - article_only}")
    
    return filename

if __name__ == "__main__":
    main()