# statistics.py
def compute_statistics(index):
    """
    Calculer toutes les statistiques demandées pour la collection
    à partir de l'index inversé.
    """
    # Nombre total de tokens (#tokens)
    total_tokens = sum(sum(tf for tf in postings.values()) for postings in index.dictionary.values())

    # Nombre de tokens distincts (#distinct tokens)
    distinct_tokens = len(index.dictionary)

    # Longueur moyenne des tokens distincts (#caractères)
    avg_length_distinct_tokens = sum(len(token) for token in index.dictionary.keys()) / distinct_tokens if distinct_tokens > 0 else 0

    # Nombre total de termes (#terms) → ici identique à total_tokens
    total_terms = total_tokens

    # Taille du vocabulaire (#distinct terms) → identique à distinct_tokens
    distinct_terms = distinct_tokens

    # Longueur moyenne d'un document (#terms)
    doc_lengths = []
    for doc_id in index.doc_ids:
        length = sum(index.dictionary[term][doc_id] for term in index.dictionary if doc_id in index.dictionary[term])
        doc_lengths.append(length)
    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

    # Longueur moyenne des termes du vocabulaire (#caractères)
    avg_length_vocab_terms = avg_length_distinct_tokens  # même que pour les tokens distincts

    # Retourner les statistiques dans un dictionnaire
    return {
        "total_tokens": total_tokens,
        "distinct_tokens": distinct_tokens,
        "avg_length_distinct_tokens": avg_length_distinct_tokens,
        "total_terms": total_terms,
        "distinct_terms": distinct_terms,
        "avg_doc_length": avg_doc_length,
        "avg_length_vocab_terms": avg_length_vocab_terms
    }
