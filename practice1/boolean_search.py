from indexer import InvertedIndex


class BooleanSearch:
    def __init__(self, index: InvertedIndex):
        self.index = index

    @staticmethod
    def AND(list1_of_doc, list2_of_doc):
        """Trouve les documents qui contiennent les DEUX mots"""
        # Cette méthode est "static" car elle n'a pas besoin des données de l'instance.
        # Elle prend juste 2 listes et retourne leur intersection
        # Exemple : "chien AND chat" → documents avec chien ET chat en meme temps.
        return sorted(list(set(list1_of_doc) & set(list2_of_doc))) # set pour enlever les doublons

    @staticmethod
    def OR(list1_of_doc, list2_of_doc):
        """Trouve les documents qui contiennent AU MOINS UN des mots"""
        # Cette méthode est "static" car elle n'a pas besoin des données de l'instance.
        # Elle prend juste deux listes et retourne leur union
        # Exemple : "chien OR chat" → documents avec chien OU chat.
        return sorted(list(set(list1_of_doc) | set(list2_of_doc))) # set pour enlever les doublons

    def NOT(self, list1_of_doc):
        """Trouve les documents qui NE CONTIENNENT PAS le mot"""
        # Cette méthode n'est PAS static car elle a besoin de connaître tous les documents
        # Utilise self.index.doc_ids pour savoir quels documents existent
        # Exemple : "NOT chien" → tous les documents SAUF ceux avec "chien".
        all_docs = set(self.index.doc_ids)    # set pour enlever les doublons
        return sorted(list(all_docs - set(list1_of_doc)))

    def AND_NOT(self, list1_of_doc, list2_of_doc):
        """Trouve les documents avec le premier mot MAIS SANS le deuxième"""
        # Combine AND et NOT
        # Exemple : "chien AND NOT chat" → documents avec chien, mais pas chat
        return self.AND(list1_of_doc, self.NOT(list2_of_doc))

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
                    postings = self.index.get_postings(term)
                    if not result:
                        result = self.NOT(postings)
                    else:
                        result = self.AND(result, self.NOT(postings))
                    i += 2  # Saute "not" et le terme
                continue

            # Pour un mot normal
            term = tokens[i]
            postings = self.index.get_postings(term)

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

    def print_query_results(self, queries):
        """Affiche les résultats de plusieurs requêtes"""
        for i, q in enumerate(queries, 1):
            print(f"\n{i}. {q}")
            tokens = q.split()

            # Cas particulier : requêtes avec "NOT" seul (ex: "NOT citizen")
            if tokens[0].lower() == "not":
                term = tokens[1]
                print(f"\t{term} => :", self.index.get_postings(term))
                print(f"\tTous les documents => :", self.index.doc_ids)
            else:
                # Affiche les postings pour chaque terme (sauf les opérateurs)
                for token in tokens:
                    if token.lower() not in ["and", "or", "not"]:
                        print(f"\t{token} => :", self.index.get_postings(token))

            # Exécute la requête et affiche le résultat
            result = self.parse_boolean_query(q)
            print("\n\tRésultat :", result)

    def print_query(self, query, num = 0):
        """Affiche les résultats d'une requête"""
        print(f"\n{num}. {query}" if num else f"\nq{query}")

        tokens = query.split()
        if tokens[0].lower() == "not":
            term = tokens[1]
            print(f"\t{term} => :", self.index.get_postings(term))
            print(f"\tTous les documents => :", self.index.doc_ids)
        else:
            for token in tokens:
                if token.lower() not in ["and", "or", "not"]:
                    print(f"\t{token} => :", self.index.get_postings(token))

        result = self.parse_boolean_query(query)
        print("\n\tRésultat :", result)