import os
import matplotlib
matplotlib.use("TkAgg")  # ou "Qt5Agg" si TkAgg ne fonctionne pas
import matplotlib.pyplot as plt
import time
import gzip
from indexer import InvertedIndex


#=============================================
# Indexation et mesure de performance
#=============================================

def read_compressed_file(filepath):
    """
    Lit un fichier compressé au format gzip et retourne son contenu sous forme de chaîne de caractères.
    """
    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        return f.read()

def get_indexation_performance(file):
    """
    Cette fonction indexe un fichier,
    mesure le temps d'indexation, et retourne les résultats.
    """
    #print(f"\n=== Indexation de {name} ===")

    text = read_compressed_file(file)
    index = InvertedIndex()

    start_time = time.time()
    index.build_from_text(text)
    end_time = time.time()

    temps_indexation = end_time - start_time
    #print(f" Temps d’indexation : {temps_indexation:.2f} secondes")

    return (file, temps_indexation)

def display_execution_data_file(result):
    """
    Affiche les données du fichier avant traitement.
    """
    print(f"Total number of tokens occurrences in the entire collection (#tokens)                           | {len(result["words"])} tokens")
    print(f"Total number of distinct tokens in the entire collection (#distinct tokens)                     | {len(set(result["words"]))} tokens")
    print(f"Average length of distinct tokens (#characters)                                                 | {len("".join(set(result["words"]))) / len(set(result["words"]))} characters")

def display_execution_data_index(result):
    """
    Affiche les données index aprés traitement.
    """
    print(f"Total indexing time (#sec)                                                                      | {result["times"]} sec")
    print(f"Total number of terms occurrences in the entire collection (#terms)                             | {len(result["terms"])} terms")
    print(f"Total number of distinct terms in the entire collection, i.e. vocabulary size (#distinct terms) | {len(set(result["terms"]))} terms")
    print(f"Average document length (#terms)                                                                | {len(result["terms"]) / len(result["doc_id"])} terms")
    print(f"Average length of vocabulary terms (#characters)                                                | {len("".join(set(result["terms"]))) / len(set(result["terms"]))} characters")


# ===============================================
# Comparaison des trois modes d'indexation
# ===============================================

def run_experiment(index, file, stopword=False, stemmer=False):
    """Construit l'index pour un mode donné et retourne les statistiques."""
    index.reset()  # réinitialise l'index
    index.stop_word_active = stopword
    index.stemmer_active = stemmer

    print("                                                                                                |")
    print("*************************************************                                               |")
    if stopword and stemmer:
        print("/  INVERTED INDEX WITH STOP WORDS AND STEMMER   /                                               |")
    elif stopword:
        print("/        INVERTED INDEX WITH STOP WORDS         /                                               |")
    else:
        print("/             INVERTED INDEX                    /                                               |")
    print("*************************************************                                               |")
    print("                                                                                                |")

    #print(f"Processing {file}...")
    start = time.time()
    index.build_from_file(file, print_index=False)
    times = (time.time() - start)

    w, t, di = index.get_data()

    return {"times": times, "terms": t, "words": w, "doc_id": di}


def main():
    # ================================================
    # Index inversé
    # ================================================
    index = InvertedIndex()
    # ================================================
    # Compare des modes d'indexation
    # ================================================

    data_path = "practice3_data/Text_Only_Ascii_Coll_NoSem"
    #index = InvertedIndex()

    # Exécution des modes d'indexation
    base = run_experiment(index, data_path)
    display_execution_data_file(base)
    #display_execution_data_index(base)
    #stop = run_experiment(index, data_path, stopword=True)
    #display_execution_data_index(stop)
    #stem = run_experiment(index, data_path, stopword=True, stemmer=True)
    #display_execution_data_index(stem)

    print(index.get_smart_itn())

if __name__ == "__main__":
    main()
