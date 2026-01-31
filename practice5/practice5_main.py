import os
import time
from typing import Dict, List, Optional
from xml_run_manager import INEXRunGenerator
from advanced_indexer import WeightedInvertedIndex
from ranked_retrieval import RankedRetrieval

#from practice5_exercices import (exercice1, exercice2, exercice3, exercice5, exercice6 ) #, exercice4_test1, exercice5, exercice6, exercice5_phase1, exercice6_phase1)
from practice5_exercices import exercice1, exercice2, exercice3, exercice5, exercice6 
from bm25_exercices import exercice5_rest, exercice6_rest, exercice6_bm25fr_test, exercice5_bm25fw, exercice6_bm25fr, exo6_bm25fr_opti_bdy, exo6_bm25fr_opti_sec, exo6_bm25fr_opti_p, exo6_bm25fr_opti_title, exercice5_bm25fw_test

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
    
    # ==================== EXÉCUTION ====================

    # Décomenter pour tester les exercices

    #exercice1()
    #exercice2()
    #exercice3()

    #exercice5()
    #exercice6()

    #exercice5_bm25fw()
    #exercice6_bm25fr()

    #exo6_bm25fr_opti_sec()
    #exo6_bm25fr_opti_p()
    #exo6_bm25fr_opti_title()
    #exercice6_bm25fr_test()
    #exo6_bm25fr_opti_bdy()
    #exercice5_bm25fw_test()

    print("\n=== FIN DES EXÉCUTIONS ===\n")

if __name__ == "__main__":
        
    main()
