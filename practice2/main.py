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
    
    #Cette fonction indexe plusieurs collections de tailles croissantes,
    #mesure le temps d'indexation pour chacune, et retourne les résultats.

    all_results = []

    for name, filename in collections:
        #print(f"\n=== Indexation de {name} ===")

        text = read_compressed_file(filename)
        index = InvertedIndex()

        start_time = time.time()
        t = index.build_from_text(text)
        end_time = time.time()

        temps_indexation = end_time - start_time
        #print(f" Temps d’indexation : {temps_indexation:.2f} secondes")
        
        # Obtenir la taille de la collection
        file_size = os.path.getsize(filename) / 1024  # Taille en Ko

        all_results.append({
            "name": name,
            "indx_time": temps_indexation,
            "num_terms": index.get_number_of_terms_file(filename),
            "file_size": file_size
        })

    return all_results


def get_indexation_performance_new(index, collections):
    """
    Cette fonction indexe plusieurs collections de tailles croissantes,
    mesure le temps d'indexation pour chacune, et retourne les résultats.
    """
    #index = InvertedIndex()
    all_results = []

    for name, filename in collections:
        #print(f"\n=== Indexation de {name} ===")

        text = read_compressed_file(filename)
        #index = InvertedIndex()

        start_time = time.time()
        index.build_from_text(text, filename)
        end_time = time.time()

        temps_indexation = end_time - start_time
        #print(f" Temps d’indexation : {temps_indexation:.2f} secondes")
        
        # Obtenir la taille de la collection
        file_size = os.path.getsize(filename) / 1024  # Taille en Ko

        all_results.append({
            "name": name,
            "indx_time": temps_indexation,
            "num_terms": index.get_number_of_terms_file(filename),
            "file_size": file_size
        })

    #return all_results
    return {
        "results": all_results,
        "stats": {
            "avg_docs_length": index.avg_document_length(),
            "avg_terms_length": index.avg_term_length(),
            "vocabulary_size": index.get_vocabulary_size()
        }
    }

def display_execution_time(results):
    """
    Affiche les temps d'exécution pour chaque collection.
    """
    print("\n=== Résumé des temps d’indexation ===")
    #for name, t in results:
    #    print(f"{name} -> {t:.2f} s")
    for result in results:
        print(f"{result['name']} -> {result['indx_time']:.2f} s")
    
"""
def plot_execution_time(sizes, times):
    
    #Trace un graphique des temps d'exécution en fonction de la taille des collections.
    
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Taille de la collection (Ko)")
    plt.ylabel("Temps d’indexation (s)")
    plt.title("Performance de l’indexation")
    plt.grid(True)
    plt.savefig("performance_indexation.png")  # sauvegarde le graphique
    plt.show()  # facultatif, graphe sauvegardé
"""
def plot_execution_time(results):
    """
    Trace un graphique des temps d'exécution en fonction du nombre de termes dans les collections.
    """
    times = [result["indx_time"] for result in results]
    sizes = [result["num_terms"] for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Nombre de termes dans la collection")
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

    times, terms, words, chars, docs = [], [], [], [], []

    print("\n\t----------------------------------------------------")
    if stopword and stemmer:
        print("\t/  INVERTED INDEX WITHOUT STOP WORDS AND STEMMER   /")
    elif stopword:
        print("\t/        INVERTED INDEX WITHOUT STOP WORDS         /")
    else:
        print("\t/          INVERTED INDEX WITH STOP WORDS          /")
    print("\t----------------------------------------------------")

    for file in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, file)
        #print(f"Processing {file}...")
        #start = time.time()
        #index.build_from_file(path, print_index=False)
        #times.append(time.time() - start)
        t = index.build_from_file(path, print_index=False)
        times.append(t)

        t, w, c, d = index.get_data(path)
        terms.append(t)
        words.append(w)
        chars.append(c)
        docs.append(d)

    return {"times": times, "terms": terms, "words": words, "chars": chars, "docs": docs}

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
    #plt.legend()
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
    
    #file_content = read_compressed_file('practice2_data/02-Text_Only-Ascii-Coll-11-20-NoSem.gz')
    #index.build_from_text(file_content)
    """
    print("\n\t*************************************************")
    print("\t/       INVERTED INDEX D'UN DOCUMENT DONNE      /")
    print("\t*************************************************")

    # Afficher le dictionnaire de séquences de tokens
    print("\n=== DICTIONNAIRE -- term -> {doc_id: tf} -- AVEC REGROUPEMENT DES TERMES ===")
    index.print_dictionary_with_size(100)  # Affiche les X premiers termes du dictionnaire
    
    #print("\n=== INDEX INVERSÉ (avec tf) ===")
    #index.display_index(with_tf=True)

    
    """

    # Exo 1 : 
    print("\n\t*************************************************")
    print("\t/          PERFORMANCE DE L'INDEXATION          /")
    print("\t*************************************************")
    
    # Mesurer la performance d'indexation
    #results = get_indexation_performance(collections)
    results = get_indexation_performance_new(index, collections)["results"]
    #display_execution_time(results)

    # Exemple de tailles en Ko (on pourrai l'adapter si le besoin est la )
    
    #times = [result["indx_time"] for result in results]
    #num_terms = [result["num_terms"] for result in results]
    #file_sizes = [result["file_size"] for result in results]
    #print("\n\nFile sizes (Ko):", file_sizes, "\n\nIndexing times (s):\n", times, "\n\nNumber of terms:\n ----", num_terms)
    # Tracer la courbe du temps d'indexation
    #plot_execution_time(sizes, times)
    plot_execution_time(results)

    
    print("\n\t*************************************************")
    print("\t/         STATISTIQUES DES COLLECTIONS          /")
    print("\t*************************************************")
    get_indexation_performance_new(index, collections)

    print(f"Longueur moyenne des documents : {index.avg_document_length()} terms")
    print(f"Longueur moyenne des termes : {index.avg_term_length()} caractères")
    print(f"Le vocabulaire contient : {index.get_vocabulary_size()} termes distincts")
    
    # ================================================
    # Compare des modes d'indexation
    # ================================================

    data_path = "practice2_data"
    #index = InvertedIndex()
    """
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
                    base["words"], base["terms"], stop["terms"], stem["terms"])
    """

if __name__ == "__main__":
    main()