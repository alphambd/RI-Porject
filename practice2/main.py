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

def get_indexation_performance(collections):
    """
    Cette fonction indexe plusieurs collections de tailles croissantes,
    mesure le temps d'indexation pour chacune, et retourne les résultats.
    """
    all_results = []

    for name, filename in collections:
        #print(f"\n=== Indexation de {name} ===")

        text = read_compressed_file(filename)
        index = InvertedIndex()

        start_time = time.time()
        index.build_from_text(text)
        end_time = time.time()

        temps_indexation = end_time - start_time
        #print(f" Temps d’indexation : {temps_indexation:.2f} secondes")

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
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Taille de la collection (Ko)")
    plt.ylabel("Temps d’indexation (s)")
    plt.title("Performance de l’indexation")
    plt.grid(True)
    plt.savefig("performance_indexation.png")  # sauvegarde le graphique
    plt.show()  # facultatif, graphe sauvegardé


# ===============================================
# Comparaison des trois modes d'indexation
# ===============================================

def run_experiment(index, data_path, stopword=False, stemmer=False):
    """Construit l'index pour un mode donné et retourne les statistiques."""
    index.reset()  # réinitialise l'index
    index.stop_word_active = stopword
    index.stemmer_active = stemmer

    times, terms, words, chars, docs, distinct_terms = [], [], [], [], [], []

    print("\n\t*************************************************")
    if stopword and stemmer:
        print("\t/  INVERTED INDEX WITH STOP WORDS AND STEMMER   /")
    elif stopword:
        print("\t/        INVERTED INDEX WITH STOP WORDS         /")
    else:
        print("\t/             INVERTED INDEX                    /")
    print("\t*************************************************")

    for file in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, file)
        #print(f"Processing {file}...")
        start = time.time()
        index.build_from_file(path, print_index=False)
        times.append(time.time() - start)

        t, w, c, d, dt = index.get_data(path)
        terms.append(t)
        words.append(w)
        chars.append(c)
        docs.append(d)
        distinct_terms.append(dt)

    return {"times": times, "terms": terms, "words": words, "chars": chars, "docs": docs, "distinct_terms": distinct_terms}

def avg_terms_per_doc(counts_terms, docs_ids):
    """Calcule la longueur moyenne (#terms/doc)"""
    result = []
    for terms, docs in zip(counts_terms, docs_ids):
        n = len(docs)
        result.append(terms / n if n > 0 else 0)
    return result

def avg_chars_per_term(counts_chars, counts_terms):
    """Calcule la longueur moyenne (#chars/term)"""
    result = []
    for chars, terms in zip(counts_chars, counts_terms):
        result.append(chars / terms if terms > 0 else 0)
    return result

def plot_comparison(x_label, y_label, title, x, y_base, y_stop, y_stem):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_base, '-bo', label='Base')
    plt.plot(x, y_stop, '-ro', label='Stopwords')
    plt.plot(x, y_stem, '-go', label='Stemmer')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # ================================================
    # Index inversé et mesure de performance d'indexation
    # ================================================
    index = InvertedIndex()
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
    
    file_content = read_compressed_file('practice2_data/01-Text_Only-Ascii-Coll-1-10-NoSem.gz')
    index.build_from_text(file_content)

    print("\n\t*************************************************")
    print("\t/       INVERTED INDEX D'UN DOCUMENT DONNE      /")
    print("\t*************************************************")

    # Afficher le dictionnaire de séquences de tokens
    print("\n=== DICTIONNAIRE -- term -> {doc_id: tf} -- AVEC REGROUPEMENT DES TERMES ===")
    index.print_dictionary_with_size(100)  # Affiche les X premiers termes du dictionnaire
    
    #print("\n=== INDEX INVERSÉ (avec tf) ===")
    #index.display_index(with_tf=True)
    

    print("\n\t*************************************************")
    print("\t/          PERFORMANCE DE L'INDEXATION          /")
    print("\t*************************************************")
    
    # Mesurer la performance d'indexation
    all_results = get_indexation_performance(collections)
    display_execution_time(all_results)

    # Exemple de tailles en Ko (on pourrai l'adapter si le besoin est la )
    sizes = [55, 52, 103, 96, 357, 559, 747, 1200, 4100]
    times = [t for _, t in all_results]
    # Tracer la courbe du temps d'indexation
    plot_execution_time(sizes, times)
    

    # ================================================
    # Compare des modes d'indexation
    # ================================================

    data_path = "practice2_data"
    #index = InvertedIndex()

    # Exécution des modes d'indexation
    base = run_experiment(index, data_path)
    stop = run_experiment(index, data_path, stopword=True)
    stem = run_experiment(index, data_path, stopword=True, stemmer=True)

    # Graphiques de comparaison des modes d'indexation
    plot_comparison("#mots", "sec", "Indexing time vs collection size",
                    base["words"], base["times"], stop["times"], stem["times"])

    plot_comparison("#mots", "#terms/doc", "Average doc length vs collection size",
                    base["words"],
                    avg_terms_per_doc(base["terms"], base["docs"]),
                    avg_terms_per_doc(stop["terms"], stop["docs"]),
                    avg_terms_per_doc(stem["terms"], stem["docs"]))

    plot_comparison("#mots", "#chars/term", "Average term length vs collection size",
                    base["words"],
                    avg_chars_per_term(base["chars"], base["terms"]),
                    avg_chars_per_term(stop["chars"], stop["terms"]),
                    avg_chars_per_term(stem["chars"], stem["terms"]))

    plot_comparison("#mots", "#terms", "Vocabulary size vs collection size",
                    base["words"], base["distinct_terms"], stop["distinct_terms"], stem["distinct_terms"])


if __name__ == "__main__":
    main()
