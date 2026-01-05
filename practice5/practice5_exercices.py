import os
import time
from xml_run_manager2 import INEXRunGenerator
from ranked_retrieval import RankedRetrieval

def exercice3_specific_run():
    """
    Exercice 3 spécifique :
    - Indexation des éléments XML (bdy, sec, p)
    - SMART lm uniquement
    - Pas d'approche Fetch & Browse (index direct des éléments)
    - Nom de fichier spécifique
    """
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
    
    # CONFIGURATION EXACTE DE L'EXERCICE 3
    config = {
        'tokenization': 'basic',
        'stemmer': 'nostem',          # Pas de stemming
        'stop_words': 'nostop',       # Pas de stop-words
        'target_tags': ['bdy', 'sec', 'p']  # ⚠️ SEULEMENT ces 3 balises (pas 'article')
    }
    
    # Paramètres POUR EXERCICE 3 (SMART lm)
    run_params = {
        'top_articles': 1600,         # Nombre d'articles à considérer
        'max_elements': 1500,         # Maximum INEX
        'weighting_scheme': 'lm',     # ⚠️ CRITIQUE: SMART lm comme demandé
        'min_element_score': 0.00001
    }
    
    print("=" * 70)
    print("EXERCICE 3: Indexation XML éléments (bdy, sec, p) - SMART lm")
    print("=" * 70)
    
    # ⚠️ IMPORTANT: Pour l'exercice 3, on utilise SEULEMENT l'index des éléments
    # Pas de phase Fetch & Browse, on indexe directement les éléments
    
    # Créer un index d'éléments seulement
    index_data = generator.create_or_load_index(
        xml_dir=XML_DIR,
        index_type='element',  # ⚠️ Uniquement éléments
        config=config,
        max_files=None
    )
    
    index = index_data['index']
    ranker = RankedRetrieval(index)
    
    # Générer le fichier run
    team_name = "AlphaAnaClement"
    group_number = "12"  # À adapter
    
    # Nom de fichier EXACT comme demandé
    filename = f"{team_name}_{group_number}_testXML_lm_element-bdy-sec-p_nostop_nostem.txt"
    filename = os.path.join("data/runs", filename)
    
    os.makedirs("data/runs", exist_ok=True)
    
    results_count = 0
    
    with open(filename, 'w', encoding='utf-8') as f:
        for query_id, query_text in QUERIES.items():
            query_start = time.time()
            
            print(f"\n[Query {query_id}] {query_text[:50]}...")
            
            # Recherche DIRECTE dans les éléments (pas de Fetch & Browse)
            results = ranker.search_query(
                query_text,
                weighting_scheme='lm',  # SMART lm
                top_k=run_params['max_elements']
            )
            
            # Trier et limiter à 1500 par requête
            results = results[:run_params['max_elements']]
            
            print(f"  Éléments trouvés: {len(results)}")
            
            # Écrire les résultats
            rank = 1
            for elem_id, score in results:
                metadata = index.get_metadata(elem_id)
                article_id = metadata.get('parent_doc_id', 'unknown')
                xml_path = metadata.get('xml_path', '/article[1]')
                
                # Format INEX standard
                f.write(
                    f"{query_id} Q0 {article_id} {rank} "
                    f"{score:.6f} {team_name} {xml_path}\n"
                )
                rank += 1
                results_count += 1
            
            query_time = time.time() - query_start
            print(f"  Temps: {query_time:.2f}s")
    
    total_results = 7 * 1500  # 7 requêtes × 1500 éléments
    
    print(f"\n{'='*70}")
    print(f"EXERCICE 3 TERMINÉ")
    print(f"Fichier: {filename}")
    print(f"Résultats: {results_count} / {total_results} attendus")
    print('='*70)
    
    # Vérification
    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"\nVérification:")
        print(f"- Lignes totales: {len(lines)}")
        print(f"- Format première ligne: {lines[0].strip() if lines else 'AUCUNE'}")
        
        # Compter les balises
        from collections import Counter
        tag_counter = Counter()
        for line in lines[:100]:  # Vérifier les 100 premières
            if '/p[' in line:
                tag_counter['p'] += 1
            elif '/sec[' in line:
                tag_counter['sec'] += 1
            elif '/bdy[' in line:
                tag_counter['bdy'] += 1
        
        print(f"\nDistribution (échantillon):")
        for tag, count in tag_counter.items():
            print(f"  {tag}: {count}")
    
    return filename


def main():
    """Fonction principale avec nettoyage"""
    # Nettoyer les anciens runs
    if os.path.exists("data/runs"):
        for file in os.listdir("data/runs"):
            if file.endswith(".txt"):
                os.remove(os.path.join("data/runs", file))
        print("Nettoyage du dossier 'runs' terminé")
    
    # Exécuter l'exercice 3
    filename = exercice3_specific_run()
    
    # Vérification finale
    print("\n" + "=" * 70)
    print("VÉRIFICATION DE CONFORMITÉ À L'EXERCICE 3")
    print("=" * 70)
    
    requirements = [
        ("Nom fichier contient 'testXML'", "testXML" in filename),
        ("Nom fichier contient 'lm'", "lm" in filename),
        ("Nom fichier contient 'element-bdy-sec-p'", "element-bdy-sec-p" in filename),
        ("Nom fichier contient 'nostop'", "nostop" in filename),
        ("Nom fichier contient 'nostem'", "nostem" in filename),
    ]
    
    for req, check in requirements:
        status = "✅" if check else "❌"
        print(f"{status} {req}")
    
    return filename

