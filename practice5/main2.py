import os
import time
from xml_run_manager2 import INEXRunGenerator
from practice5_exercices import exercice3_specific_run

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
    
    # Configuration unique (fetch et browse ont la même config)
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',
        'stop_words': 'nostop',
        'target_tags': ['article', 'bdy', 'sec', 'p']  # Inclure 'article' est CRUCIAL
    }
    
    # Paramètres optimisés
    run_params = {
        'top_articles': 1600,
        'max_elements': 1500,
        'weighting_scheme': 'ltn',
        'min_element_score': 0.00001
    }
    
    print("=" * 70)
    print("GÉNÉRATION RUN INEX - Version optimisée")
    print("=" * 70)
    
    # Générer le run
    filename = generator.generate_fetch_browse_run(
        run_id="testXML_FINAL",
        xml_dir=XML_DIR,
        queries=QUERIES,
        fetch_config=config,
        browse_config=config,
        run_params=run_params
    )
    
    print(f"\n✅ Run généré: {filename}")
    
    # Vérification finale
    print("\n" + "=" * 70)
    print("VÉRIFICATION FINALE")
    print("=" * 70)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"Nombre total de lignes: {len(lines)}")
        print(f"Attendu (7 requêtes × 1500): 10,500")
        
        # Analyser la distribution
        from collections import Counter
        tags_counter = Counter()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 7:
                xml_path = parts[6]
                if '/p[' in xml_path:
                    tags_counter['p'] += 1
                elif '/sec[' in xml_path and '/p[' not in xml_path:
                    tags_counter['sec'] += 1
                elif '/bdy[' in xml_path and '/sec[' not in xml_path:
                    tags_counter['bdy'] += 1
                elif xml_path.endswith('/article[1]'):
                    tags_counter['article'] += 1
        
        print("\nDistribution des balises:")
        for tag, count in tags_counter.most_common():
            print(f"  {tag}: {count} éléments ({count/len(lines)*100:.1f}%)")
    
    return filename


if __name__ == "__main__":
    # Nettoyer les anciens runs
    if os.path.exists("data/runs"):
        for file in os.listdir("data/runs"):
            if file.endswith(".txt"):
                os.remove(os.path.join("data/runs", file))
        print("Nettoyage du dossier 'runs' terminé")
    
    # Lancer la génération
    main()
    exercice3_specific_run()