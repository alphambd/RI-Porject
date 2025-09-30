import re
from collections import defaultdict, Counter
import string


class InvertedIndex:
    def __init__(self):
        # Dictionnaire qui stocke pour chaque mot la liste des documents où il apparaît
        self.dictionary = defaultdict(dict)
        # Liste de tous les identifiants de documents
        self.doc_ids = []

    def preprocess_text(self, text):
        """Nettoie le texte avant de l'ajouter à l'index"""
        # Met tout en minuscules pour éviter les doublons
        text = text.lower()
        # Enlève les apostrophes
        text = text.replace("'", "").replace("'", "")
        # Supprime la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Découpe le texte en mots
        tokens = text.split()
        return tokens

    def add_document(self, doc_id, text):
        """Ajoute un document à l'index inversé"""
        # Nettoie le texte et le découpe en mots
        tokens = self.preprocess_text(text)
        # Compte combien de fois chaque mot apparaît
        term_freq = Counter(tokens)

        # Pour chaque mot, note dans quel document il apparaît et combien de fois
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq

    def build_from_file(self, filename):
        """Construit l'index à partir d'un fichier de documents"""
        # Ouvre et lit le fichier
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Cherche tous les documents dans le fichier
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)

        # Pour chaque document trouvé, l'ajoute à l'index
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.doc_ids.append(doc_id)
            self.add_document(doc_id, doc_text)

    def display_index(self, with_tf=False):
        """Affiche l'index inversé de façon lisible"""
        # Trie les mots par ordre alphabétique
        sorted_terms = sorted(self.dictionary.keys())

        # Pour chaque mot, affiche les documents où il apparaît
        for term in sorted_terms:
            postings = self.dictionary[term]
            df = len(postings)
            if with_tf:
                print(f"{df}=df({term})")
                for doc_id, tf in sorted(postings.items()):
                    print(f"    {tf} {doc_id}")
            else:
                print(f"{df}=df({term})")
                for doc_id in sorted(postings.keys()):
                    print(f"    {doc_id}")

    def get_postings(self, term):
        """Donne la liste des documents contenant un mot"""
        term = term.lower()
        if term in self.dictionary:
            return sorted(self.dictionary[term].keys())
        return []

    def get_document_frequency(self, term):
        """Donne le nombre de documents contenant un mot"""
        term = term.lower()
        return len(self.dictionary.get(term, {}))

    # =========================================================================
    #            MÉTHODES POUR LES REQUÊTES BOOLÉENNES
    # =========================================================================

    @staticmethod
    def AND(list1, list2):
        """Trouve les documents qui contiennent les DEUX mots"""
        # Cette méthode est "static" car elle n'a pas besoin des données de l'instance.
        # Elle prend juste 2 listes et retourne leur intersection
        # Exemple : "chien AND chat" → documents avec chien ET chat.
        return sorted(list(set(list1) & set(list2)))

    @staticmethod
    def OR(list1, list2):
        """Trouve les documents qui contiennent AU MOINS UN des mots"""
        # Cette méthode est "static" car elle n'a pas besoin des données de l'instance.
        # Elle prend juste deux listes et retourne leur union
        # Exemple : "chien OR chat" → documents avec chien OU chat.
        return sorted(list(set(list1) | set(list2)))

    def NOT(self, list1):
        """Trouve les documents qui NE CONTIENNENT PAS le mot"""
        # Cette méthode n'est PAS static car elle a besoin de connaître tous les documents
        # Utilise self.doc_ids pour savoir quels documents existent
        # Exemple : "NOT chien" → tous les documents SAUF ceux avec "chien".
        all_docs = set(self.doc_ids)
        return sorted(list(all_docs - set(list1)))

    def AND_NOT(self, list1, list2):
        """Trouve les documents avec le premier mot MAIS SANS le deuxième"""
        # Combine AND et NOT
        # Exemple : "chien AND NOT chat" → documents avec chien, mais pas chat
        return self.AND(list1, self.NOT(list2))

    def parse_boolean_query(self, query):
        """Comprend une requête de l'utilisateur et trouve les documents"""
        # Transforme la requête en mots : par exemple : "Citizen and Kane" → ['citizen', 'and', 'kane']
        tokens = query.lower().split()
        result = []  # Liste des documents trouvés
        i = 0

        while i < len(tokens):
            # Ignore les mots 'and', 'or' (C'est pour dire qu'on les utilise comme des séparateurs pour combiner deux mots donc on ne doit pas les cherchés dans la liste des mots)
            if tokens[i] in ['and', 'or']:
                i += 1
                continue

            # Si on trouve "NOT", on cherche les documents SANS le mot suivant
            if tokens[i] == 'not':
                if i + 1 < len(tokens):
                    term = tokens[i + 1]
                    postings = self.get_postings(term)
                    if not result:
                        result = self.NOT(postings)
                    else:
                        result = self.AND(result, self.NOT(postings))
                    i += 2  # Saute "not" et le terme
                continue

            # Pour un mot normal
            term = tokens[i]
            postings = self.get_postings(term)

            # Premier mot de la requête
            if not result:
                result = postings
            else:
                # Regarde l'opérateur avant ce mot
                if i > 0 and tokens[i - 1] == 'or':
                    result = self.OR(result, postings)
                elif i > 0 and tokens[i - 1] == 'and not':
                    result = self.AND_NOT(result, postings)
                else:  # AND par défaut
                    result = self.AND(result, postings)
            i += 1

        return result


def main():
    #  Créer et construire l'index, en fait, c'est l'appel du code que pour l'exercice 2 pour pouvoir utiliser l'index inversé
    index = InvertedIndex()
    index.build_from_file('collection.txt')

    print("=== INDEX INVERSÉ ===")
    index.display_index(with_tf=True)

    print("\n=== MAINTENANT L'UTILISATION DE L'INDEX INVERSÉ POUR TROUVER LES REQUÊTES BOOLÉENNES ===")
    print("\n=== REQUÊTES BOOLÉENNES ===")

    # Test des opérateurs booléens avec les données
    print("\n1. citizen AND kane")
    result1 = index.parse_boolean_query("citizen and kane")
    print(f"   Résultat : {result1}")

    print("\n2. the OR godfather")
    result2 = index.parse_boolean_query("the or godfather")
    print(f"   Résultat : {result2}")

    print("\n3. the AND NOT godfather")
    result3 = index.parse_boolean_query("the and not godfather")
    print(f"   Résultat : {result3}")

    print("\n4. NOT citizen")
    result4 = index.parse_boolean_query("not citizen")
    print(f"   Résultat : {result4}")

    print("\n5. of AND wizard")
    result5 = index.parse_boolean_query("of and wizard")
    print(f"   Résultat : {result5}")

    print("\n6. lawrence OR oz")
    result6 = index.parse_boolean_query("lawrence or oz")
    print(f"   Résultat : {result6}")


if __name__ == "__main__":
    main()