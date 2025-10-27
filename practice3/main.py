from indexer import InvertedIndex
from statistics import compute_statistics
import time  # pour mesurer le temps d'indexation

def main():

    # Créer et construire l'index
    index = InvertedIndex()

    # Mesurer le temps d'indexation
    start_time = time.time()
    index.build_from_file('Practice_03_data/Text_Only_Ascii_Coll_NoSem')
    indexing_time = time.time() - start_time

    print("\n\t*************************************************")
    print("\t/             INVERTED INDEX                    /")
    print("\t*************************************************")

    # Afficher le dictionnaire complet
    print("\n=== DICTIONNAIRE -- term -> {doc_id: tf} ===")
    index.print_dictionary()

    # Afficher index inversé avec tf
    print("\n=== INDEX INVERSÉ (avec tf) ===")
    index.display_index(with_tf=True)

    # ==========================
    # Calculer les statistiques
    # ==========================
    stats = compute_statistics(index)

    # Afficher les statistiques
    print("\n\t*************************************************")
    print("\t/               COLLECTION STATS                /")
    print("\t*************************************************")
    print(f"Temps total d'indexation (sec): {indexing_time:.2f}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k} : {v:.2f}")
        else:
            print(f"{k} : {v}")

    # ==========================
    # Pause pour que la fenêtre reste ouverte
    # ==========================
    input("\nAppuyez sur Entrée pour fermer...")

if __name__ == "__main__":
    main()
