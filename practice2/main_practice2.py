import time
import gzip
from indexer_practice1 import InvertedIndex
import matplotlib.pyplot as plt

import os
import time


def read_compressed_file(filepath):
    """
    Lit un fichier compressé au format gzip et retourne son contenu sous forme de chaîne de caractères.
    """
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        return f.read()

def get_indexation_performance(collections):
    """
    Cette fonction indexe plusieurs collections de tailles croissantes,
    mesure le temps d'indexation pour chacune, et retourne les résultats.
    """
    all_results = []

    for name, filename in collections:
        print(f"\n=== Indexation de {name} ===")

        #with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as f:
        #    text = f.read()

        text = read_compressed_file(filename)
        index = InvertedIndex()

        start_time = time.time()
        index.build_from_text(text)
        end_time = time.time()

        temps_indexation = end_time - start_time
        print(f" Temps d’indexation : {temps_indexation:.2f} secondes")

        all_results.append((name, temps_indexation))

    return all_results

def display_execution_time(results):
    """
    Affiche les temps d'exécution pour chaque collection.
    """
    print("\n=== Résumé des temps d’indexation ===")
    for name, t in results:
        print(f"{name} -> {t:.2f} s")

def plot_execution_time(sizes, times):
    """
    Trace un graphique des temps d'exécution en fonction de la taille des collections.
    """
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Taille de la collection (Ko)")
    plt.ylabel("Temps d’indexation (s)")
    plt.title("Performance de l’indexation")
    plt.grid(True)
    plt.savefig("performance_indexation.png")  # sauvegarde le graphique
    plt.show()  # facultatif

def main():
    # Liste des fichiers de collection
    collections = [
        ("Petite collection (1 à 10 documents)", "practice2_data/01-Text_Only-Ascii-Coll-1-10-NoSem.gz"),
        ("Petite collection (11 à 20 documents)", "practice2_data/02-Text_Only-Ascii-Coll-11-20-NoSem.gz"),
        ("Collection moyenne (21 à 50 documents)", "practice2_data/03-Text_Only-Ascii-Coll-21-50-NoSem.gz"),
        ("Collection moyenne (51 à 100 documents)", "practice2_data/04-Text_Only-Ascii-Coll-51-100-NoSem.gz"),
        ("Grande collection (101 à 200 documents)", "practice2_data/05-Text_Only-Ascii-Coll-101-200-NoSem.gz"),
        ("Grande collection (201 à 500 documents)", "practice2_data/06-Text_Only-Ascii-Coll-201-500-NoSem.gz"),
        ("Très grande collection (501 à 1000 documents)", "practice2_data/07-Text_Only-Ascii-Coll-501-1000-NoSem.gz"),
        ("Très grande collection (1001 à 2000 documents)", "practice2_data/08-Text_Only-Ascii-Coll-1001-2000-NoSem.gz"),
        ("Énorme collection (2001 à 5000 documents)", "practice2_data/09-Text_Only-Ascii-Coll-2001-5000-NoSem.gz")
    ]

    
    index = InvertedIndex()
    #index.build_from_file('collection.txt')
    file_content = read_compressed_file('practice2_data/01-Text_Only-Ascii-Coll-1-10-NoSem.gz')
    index.build_from_text(file_content)

    print("\n\t*************************************************")
    print("\t/       INVERTED INDEX D'UN DOCUMENT DONNE      /")
    print("\t*************************************************")

    # Afficher le dictionnaire de séquences de tokens
    print("\n=== DICTIONNAIRE -- term -> {doc_id: tf} -- AVEC REGROUPEMENT DES TERMES ===")
    index.print_dictionary_with_size(100)  # Affiche les X premiers termes du dictionnaire
    
    print("\n=== INDEX INVERSÉ (avec tf) ===")
    #index.display_index(with_tf=True)


    print("\n\t*************************************************")
    print("\t/          PERFORMANCE DE L'INDEXATION          /")
    print("\t*************************************************")

    all_results = get_indexation_performance(collections)
    display_execution_time(all_results)

    # Exemple de tailles en Ko (on pourrai l'adapter si le besoin est la )
    sizes = [55, 52, 103, 96, 357, 559, 747, 1200, 4100]
    times = [t for _, t in all_results]

    plot_execution_time(sizes, times)

    """    
    all_results = []
    
    
    for name, filename in collections:
        print(f"\n=== Indexation de {name} ===")

        with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        index = InvertedIndex()

        start_time = time.time()
        index.build_from_text(text)
        end_time = time.time()

        temps_indexation = end_time - start_time
        print(f" Temps d’indexation : {temps_indexation:.2f} secondes")

        all_results.append((name, temps_indexation))
    
    # Affichage final
    print("\n=== Résumé des temps d’indexation ===")
    for name, t in all_results:
        print(f"{name} -> {t:.2f} s")
    
    # Exemple de tailles en Ko (on pourrai l'adapter si le besoin est la )
    sizes = [55, 52, 103, 96, 357, 559, 747, 1200, 4100]
    times = [t for _, t in all_results]

    # Tracer la courbe
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Taille de la collection (Ko)")
    plt.ylabel("Temps d’indexation (s)")
    plt.title("Performance de l’indexation")
    plt.grid(True)
    plt.savefig("performance_indexation.png")  # sauvegarde le graphique
    # plt.show()  # facultatif
    """

if __name__ == "__main__":
    main()
