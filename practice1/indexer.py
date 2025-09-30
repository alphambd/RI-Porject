import re # for regex operations
from collections import defaultdict, Counter # for dictionary and counting
import string

class InvertedIndex:
    def __init__(self):
        self.dictionary = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_ids = []                    # Liste des IDs de documents (en cas de besoin pour la suite, comme pour la matrice d'incidence par exemple)
    
    def preprocess_text(self, text):
        """Traitement basique du texte"""
        # Convertir en minuscules
        text = text.lower()
        # Traiter le cas des apostrophes (pour schindler's, singin', etc.)
        text = text.replace("’", "'")
        text = re.sub(r"'s?\b", "", text)
        # Supprimer la ponctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenisation
        tokens = text.split()
        return tokens
    
    def add_document(self, doc_id, text):
        """Ajouter un document à l'index"""
        tokens = self.preprocess_text(text)
        # Compter les fréquences des termes
        term_freq = Counter(tokens)
        
        # Mettre à jour le dictionnaire
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
    
    def build_from_file(self, filename):
        """Construire l'index à partir d'un fichier"""
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extraire les documents avec regex
        """
        le premier ([^<]+) : capture le contenu de <docno>...</docno> (ex : D0)
        le deuxième ([^<]+) : capture le contenu texte du document (ex: Citizen Kane)
        """
        doc_pattern = r'<doc><docno>([^<]+)</docno>([^<]+)</doc>'
        matches = re.findall(doc_pattern, content)
        
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            self.doc_ids.append(doc_id)
            self.add_document(doc_id, doc_text)
    
    def display_index(self, with_tf=False):
        """Afficher l'index inversé"""
        # Trier les termes alphabétiquement
        sorted_terms = sorted(self.dictionary.keys())
        
        for term in sorted_terms:
            postings = self.dictionary[term]
            df = len(postings)
            #print("Term : ", term, "Postings : ", postings, "\n\n")
            if with_tf:
                print(f"{df}=df({term})")
                for doc_id, tf in sorted(postings.items()):
                    print(f"    {tf} {doc_id}")
            else:
                print(f"{df}=df({term})")
                for doc_id in sorted(postings.keys()):
                    #print(f"    1 {doc_id}")
                    print(f"    {doc_id}")
    
    def get_postings(self, term):
        """Récupérer la liste de postings pour un terme"""
        term = term.lower()
        if term in self.dictionary:
            return sorted(self.dictionary[term].keys())
        return []
    
    def get_document_frequency(self, term):
        """Récupérer la fréquence de document pour un terme"""
        term = term.lower()
        return len(self.dictionary.get(term, {}))

def main():
    # Créer et construire l'index
    index = InvertedIndex()
    index.build_from_file('collection.txt')

    # Afficher le dictionnaire de séquences de tokens
    print("=== DICTIONNAIRE TOKEN SEQUENCE : term -> {doc_id: tf} ===\n", index.dictionary)

    # Afficher les postings pour un terme donné
    terme = "the"
    print(f"\n=== POSTINGS pour le terme '{terme}' ===\n\t", index.get_postings(terme))

    # Afficher la fréquence de document pour un terme donné
    print(f"=== DOCUMENT FREQUENCY pour le terme '{terme}' ===\n\t", index.get_document_frequency(terme))
    
    print("=== INDEX INVERSÉ (sans tf) ===")
    #index.display_index(with_tf=False)
    
    print("\n=== INDEX INVERSÉ (avec tf) ===")
    index.display_index(with_tf=True)
    
if __name__ == "__main__":
    main()