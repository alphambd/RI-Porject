import math

from smart_ltn import SMARTLTNIndex
import os


def main():
    """Fonction principale pour exécuter le système de récupération SMART ltn"""
    print("=== Système de Récupération SMART ltn ===\n")

    # Créer l'index
    index = SMARTLTNIndex()

    # Charger les documents depuis le fichier spécifié
    file_path = "Practice_03_data/Text_Only_Ascii_Coll_NoSem"

    try:
        num_docs = index.load_documents_from_file(file_path)
        if num_docs == 0:
            print("Aucun document trouvé dans le fichier.")
            return
    except FileNotFoundError:
        print(f"Fichier {file_path} non trouvé.")
        print("Veuillez vous assurer que le fichier existe dans le répertoire courant.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement des documents : {e}")
        return

    # Calculer les statistiques des termes
    print("Calcul des statistiques des termes...")
    index.compute_term_statistics()

    # Calculer les poids des documents
    print("Calcul des poids des documents (ltn)...")
    doc_weights, weighting_time = index.compute_document_weights()

    # Définir la requête
    query = "web ranking scoring algorithm"
    query_terms = index.preprocess_text(query)

    print(f"Requête : '{query}'")
    print(f"Termes de requête traités : {query_terms}\n")

    # Calculer les scores RSV
    rsv_scores = index.compute_rsv(query_terms, doc_weights)

    # Afficher les résultats
    print("=== RÉSULTATS ===\n")

    # 1. Temps total de pondération
    print(f"1. Temps total de pondération : {weighting_time:.6f} secondes")

    # 2. Poids du terme "ranking" dans le document #23724 (s'il existe)
    target_doc_id = 23724
    if target_doc_id in doc_weights:
        ranking_weight = doc_weights[target_doc_id].get("ranking", 0)
        print(f"2. Poids du terme 'ranking' dans le document #{target_doc_id} : {ranking_weight:.6f}")

        # 3. RSV du document #23724
        rsv_target = rsv_scores[target_doc_id]
        print(f"3. RSV du document #{target_doc_id} : {rsv_target:.6f}")
    else:
        # Trouver le premier document qui contient "ranking" pour démonstration
        ranking_docs = [(doc_id, weights.get("ranking", 0))
                        for doc_id, weights in doc_weights.items()
                        if "ranking" in weights]

        if ranking_docs:
            # Trier par poids de "ranking" décroissant
            ranking_docs.sort(key=lambda x: x[1], reverse=True)
            target_doc_id, ranking_weight = ranking_docs[0]
            print(f"2. Poids du terme 'ranking' dans le document #{target_doc_id} : {ranking_weight:.6f}")
            print(f"3. RSV du document #{target_doc_id} : {rsv_scores[target_doc_id]:.6f}")
        else:
            print("2. Aucun document ne contient le terme 'ranking'")
            print("3. Impossible de calculer le RSV pour le document #23724 - document non trouvé")

    # 4. Top-10 des documents
    print("\n4. Top-10 des documents les plus pertinents :")
    print("-" * 50)
    print(f"{'Rang':<6} {'Doc ID':<10} {'RSV':<12} {'Termes Correspondants'}")
    print("-" * 50)

    # Filtrer les documents avec RSV non nul et trier
    non_zero_rsv = {doc_id: score for doc_id, score in rsv_scores.items() if score > 0}
    sorted_docs = sorted(non_zero_rsv.items(), key=lambda x: x[1], reverse=True)[:10]

    for rank, (doc_id, rsv) in enumerate(sorted_docs, 1):
        # Compter combien de termes de requête sont dans ce document
        matching_terms = sum(1 for term in query_terms if term in doc_weights.get(doc_id, {}))
        print(f"{rank:<6} {doc_id:<10} {rsv:<12.6f} {matching_terms}")

    # Afficher les poids détaillés pour le premier document si disponible
    if sorted_docs:
        top_doc_id = sorted_docs[0][0]
        print(f"\nPoids détaillés pour le premier document (#{top_doc_id}) :")
        for term in query_terms:
            if term in doc_weights[top_doc_id]:
                weight = doc_weights[top_doc_id][term]
                print(f"  {term} : {weight:.6f}")

    # Afficher les statistiques de la collection
    print(f"\n=== STATISTIQUES DE LA COLLECTION ===")
    print(f"Nombre total de documents : {index.N}")
    print(f"Taille du vocabulaire : {len(index.vocabulary)}")
    print(f"Fréquence de document pour les termes de requête :")
    for term in query_terms:
        df = index.term_stats.get(term, 0)
        if df > 0:
            idf = math.log(index.N / df)
            print(f"  {term} : df = {df}, idf = {idf:.4f}")
        else:
            print(f"  {term} : df = 0, idf = N/A")


if __name__ == "__main__":
    main()