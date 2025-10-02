from indexer import InvertedIndex
from boolean_search import BooleanSearch

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
    search.print_query(q3, 3)
    search.print_query(q4, 4)
    search.print_query(q5, 5)
    search.print_query(q6, 6)

if __name__ == "__main__":
    main()