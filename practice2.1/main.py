from indexer import InvertedIndex
from boolean_search import BooleanSearch

"""
def main():
    #  Créer et construire l'index
    index = InvertedIndex()
    index.build_from_file('collection.txt')
    # Créer une instance de BooleanSearch
    search = BooleanSearch(index)

    print("\n\t*************************************************")
    print("\t/             INVERTED INDEX                    /")
    print("\t*************************************************")

    # Afficher le dictionnaire de séquences de tokens
    print("\n=== DICTIONNAIRE -- term -> {doc_id: tf} -- AVEC REGROUPEMENT DES TERMES ===")
    index.print_dictionary()

    # Afficher les postings pour un terme donné
    terme = "the"
    print(f"\n=== POSTINGS LIST pour le terme '{terme}' ===\n\t", index.get_postings(terme))

    # Afficher la fréquence de document pour un terme donné
    print(f"\n=== DOCUMENT FREQUENCY pour le terme '{terme}' ===\n\t", index.get_document_frequency(terme))
    
    print("\n=== INDEX INVERSÉ (avec tf) ===")
    index.display_index(with_tf=True)


    print("\n\t*************************************************")
    print("\t/             BOOLEAN SEARCH                    /")
    print("\t*************************************************")

    print("\n=== REQUÊTES BOOLÉENNES ===")

    # Exemples de requêtes booléennes
    q1 = "citizen and kane"
    q2 = "the or godfather"
    q3 = "the and not godfather"
    q4 = "not citizen"
    q5 = "of and wizard"
    q6 = "lawrence or oz"

    # Afficher les résultats des requêtes
    search.print_query(q1, 1)
    search.print_query(q2, 2)
    search.print_query(q3, 3)p
    search.print_query(q4, 4)
    search.print_query(q5, 5)
    search.print_query(q6, 6)

if __name__ == "__main__":
    main()
"""
from statistics_analyzer import compare_with_stop_words, comprehensive_analysis
from advanced_indexer import AdvancedInvertedIndex


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

    collection_file = 'collection.txt'
    stop_words_file = 'stop-words-english4.txt'

    print("\n\t*************************************************")
    print("\t/        COMPARAISON AVEC/SANS STOP WORDS       /")
    print("\t*************************************************")

    # Comparaison avec/sans stop words
    compare_with_stop_words(collection_file, stop_words_file)

    print("\n\t*************************************************")
    print("\t/      ANALYSE COMPARATIVE DES TROIS TECHNIQUES /")
    print("\t*************************************************")

    # Analyse comparative des trois techniques
    results = comprehensive_analysis(collection_file, stop_words_file)

    for method, stats in results.items():
        print(f"\n=== STATISTIQUES POUR LA MÉTHODE: {method.upper()} ===")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value}")

    


if __name__ == "__main__":
    main()
