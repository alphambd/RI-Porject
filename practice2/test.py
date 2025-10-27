import re
from collections import defaultdict, Counter
import gzip
import os
import time
import string
import matplotlib.pyplot as plt

# Porter Stemmer intégré pour éviter les dépendances externes
class PorterStemmer:
    def __init__(self):
        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.j = 0   # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] in 'aeiou':
            return False
        if self.b[i] == 'y':
            if i == self.k0:
                return True
            else:
                return not self.cons(i - 1)
        return True

    def m(self):
        """m() measures the number of consonant sequences between k0 and j."""
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i += 1
            i += 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return True
        return False

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return False
        if self.b[j] != self.b[j-1]:
            return False
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.
           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i-1) or not self.cons(i-2):
            return False
        if self.b[i] in 'wxy':
            return False
        return True

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]: # tiny speed-up
            return False
        if length > (self.k - self.k0 + 1):
            return False
        if self.b[self.k-length+1:self.k+1] != s:
            return False
        self.j = self.k - length
        return True

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j+1] + s + self.b[self.j+length+1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.
           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat
           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable
           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess
           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):   self.setto("ate")
            elif self.ends("bl"): self.setto("ble")
            elif self.ends("iz"): self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                if self.b[self.k] in 'lsz':
                    self.k = self.k + 1
            elif self.m() == 1 and self.cvc(self.k):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if self.ends("y") and self.vowelinstem():
            self.b = self.b[:self.k] + 'i' + self.b[self.k+1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):   self.r("ate")
            elif self.ends("tional"):  self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):      self.r("ence")
            elif self.ends("anci"):    self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):       self.r("ble") # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):    self.r("al")
            elif self.ends("entli"):   self.r("ent")
            elif self.ends("eli"):     self.r("e")
            elif self.ends("ousli"):   self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):   self.r("ize")
            elif self.ends("ation"):   self.r("ate")
            elif self.ends("ator"):    self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):     self.r("al")
            elif self.ends("iveness"): self.r("ive")
            elif self.ends("fulness"): self.r("ful")
            elif self.ends("ousness"): self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):     self.r("al")
            elif self.ends("iviti"):    self.r("ive")
            elif self.ends("biliti"):   self.r("ble")
        elif self.b[self.k - 1] == 'g': # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() deals with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):     self.r("ic")
            elif self.ends("ative"):   self.r("")
            elif self.ends("alize"):   self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):      self.r("ic")
            elif self.ends("ful"):     self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"): pass
            else: return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"): pass
            elif self.ends("ence"): pass
            else: return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"): pass
            else: return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"): pass
            else: return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"): pass
            elif self.ends("ible"): pass
            else: return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"): pass
            elif self.ends("ement"): pass
            elif self.ends("ment"): pass
            elif self.ends("ent"): pass
            else: return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'): pass
            elif self.ends("ou"): pass
            else: return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"): pass
            else: return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"): pass
            elif self.ends("iti"): pass
            else: return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"): pass
            else: return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"): pass
            else: return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"): pass
            else: return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k-1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k - 1

    def stem(self, word):
        """Stem the word and return the stemmed form."""
        # Words of 2 or fewer letters are already stemmed
        if len(word) <= 2:
            return word.lower()

        self.b = word.lower()
        self.k = len(self.b) - 1
        self.k0 = 0
        
        # Strip punctuation and handle the word
        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        
        return self.b[self.k0:self.k+1]

class AdvancedInvertedIndex:
    def __init__(self):
        # Structure principale
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}  # Stocke le texte des documents pour statistiques
        
        # Statistiques globales
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        
        # Options de traitement
        self.stop_word_active = False
        self.stemmer_active = False
        self.stop_words_set = set()
        
        # Résultats par fichier pour les graphiques
        self.file_statistics = {}
        
        # Stemmer
        self.stemmer = PorterStemmer()
    
    def reset(self):
        """Réinitialise complètement l'index - CRITIQUE pour les comparaisons"""
        self.dictionary = defaultdict(dict)
        self.doc_ids = []
        self.all_documents = {}
        self.total_tokens = 0
        self.total_chars = 0
        self.document_lengths = []
        self.file_statistics = {}
    
    def load_stop_words(self, stop_words_file="stop-words-english4.txt"):
        """Charge la liste des stop words depuis un fichier"""
        try:
            with open(stop_words_file, 'r', encoding='utf-8') as file:
                self.stop_words_set = set(line.strip().lower() for line in file if line.strip())
            print(f"✅ {len(self.stop_words_set)} stop words chargés")
        except FileNotFoundError:
            print(f"⚠️  Fichier {stop_words_file} non trouvé")
            self.stop_words_set = set()
    
    def preprocess_text(self, text):
        """Tokenisation selon l'énoncé : terms without digits or special characters"""
        # Conversion minuscules
        text = text.lower()
        
        # Nettoyage selon l'énoncé : supprimer chiffres et caractères spéciaux
        # Conserver uniquement les lettres
        text = re.sub(r"[^a-z\s]", " ", text)
        
        # Tokenisation
        tokens = text.split()
        
        return tokens
    
    def apply_stemming_to_tokens(self, tokens):
        """Applique le stemming token par token"""
        if not self.stemmer_active:
            return tokens
        
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            stemmed_tokens.append(stemmed_token)
        
        return stemmed_tokens
    
    def add_document(self, doc_id, text, filename=None):
        """Ajoute un document avec tous les traitements"""
        # Stocker le document original pour statistiques
        self.all_documents[doc_id] = text
        self.doc_ids.append(doc_id)
        
        # Prétraitement de base
        tokens = self.preprocess_text(text)
        
        # Appliquer le stemming si activé (sur les tokens)
        if self.stemmer_active:
            tokens = self.apply_stemming_to_tokens(tokens)
        
        # Filtrer les stop words si activé
        if self.stop_word_active and self.stop_words_set:
            tokens = [token for token in tokens if token not in self.stop_words_set]
        
        # Mise à jour des statistiques globales
        self.total_tokens += len(tokens)
        self.total_chars += sum(len(token) for token in tokens)
        self.document_lengths.append(len(tokens))
        
        # Calcul des fréquences des termes
        term_freq = Counter(tokens)
        
        # Mise à jour du dictionnaire inversé
        for term, freq in term_freq.items():
            self.dictionary[term][doc_id] = freq
        
        # Statistiques par fichier
        if filename:
            if filename not in self.file_statistics:
                self.file_statistics[filename] = {
                    'documents': 0,
                    'tokens': 0,
                    'chars': 0,
                    'vocabulary': set()
                }
            
            self.file_statistics[filename]['documents'] += 1
            self.file_statistics[filename]['tokens'] += len(tokens)
            self.file_statistics[filename]['chars'] += sum(len(token) for token in tokens)
            self.file_statistics[filename]['vocabulary'].update(term_freq.keys())
    
    def build_from_file(self, filename, verbose=False, print_index=False):
        """Construit l'index depuis un fichier avec options de contrôle"""
        start_time = time.time()
        
        try:
            # Lecture du fichier compressé
            with gzip.open(filename, 'rt', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        except Exception as e:
            print(f"❌ Erreur lecture {filename}: {e}")
            return None
        
        # Extraction des documents - pattern corrigé
        doc_pattern = r'<doc>\s*<docno>([^<]+)</docno>\s*(.*?)\s*</doc>'
        matches = re.findall(doc_pattern, content, re.DOTALL)
        
        if not matches:
            print(f"⚠️  Aucun document trouvé dans {filename}")
            return None
        
        # Indexation de chaque document
        for doc_id, doc_text in matches:
            doc_id = doc_id.strip()
            doc_text = doc_text.strip()
            if doc_text:  # Ne traiter que les documents non vides
                self.add_document(doc_id, doc_text, filename)
        
        end_time = time.time()
        indexing_time = end_time - start_time
        
        # Affichage contrôlé (variants de l'exercice 1)
        if verbose:
            print(f"📊 {filename}: {len(matches)} documents indexés en {indexing_time:.2f}s")
        
        if print_index and len(matches) <= 50:  # Seulement pour petites collections
            self.display_index(limit=20)
        
        return indexing_time
    
    def get_global_statistics(self):
        """Calcule les statistiques globales demandées dans l'exercice 2.1"""
        if not self.doc_ids:
            return {
                'avg_document_length': 0,
                'avg_term_length': 0,
                'vocabulary_size': 0,
                'total_documents': 0,
                'total_tokens': 0
            }
        
        avg_doc_length = self.total_tokens / len(self.doc_ids)
        avg_term_length = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        vocabulary_size = len(self.dictionary)
        
        return {
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length,
            'vocabulary_size': vocabulary_size,
            'total_documents': len(self.doc_ids),
            'total_tokens': self.total_tokens
        }
    
    def get_file_statistics(self, filename):
        """Retourne les statistiques pour un fichier spécifique"""
        if filename not in self.file_statistics:
            return None
        
        stats = self.file_statistics[filename]
        avg_doc_length = stats['tokens'] / stats['documents'] if stats['documents'] > 0 else 0
        avg_term_length = stats['chars'] / stats['tokens'] if stats['tokens'] > 0 else 0
        
        return {
            'documents': stats['documents'],
            'tokens': stats['tokens'],
            'vocabulary_size': len(stats['vocabulary']),
            'avg_document_length': avg_doc_length,
            'avg_term_length': avg_term_length
        }
    
    def display_index(self, limit=0, with_tf=False):
        """Affiche l'index avec contrôle (variants exercice 1)"""
        sorted_terms = sorted(self.dictionary.keys())
        
        if limit > 0:
            sorted_terms = sorted_terms[:limit]
            print(f"📄 Affichage des {limit} premiers termes...")
        
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
    
    # Méthodes utilitaires
    def get_postings(self, term):
        term = term.lower()
        return sorted(self.dictionary.get(term, {}).keys())
    
    def get_document_frequency(self, term):
        term = term.lower()
        return len(self.dictionary.get(term, {}))
    
    def get_vocabulary(self):
        return set(self.dictionary.keys())


class Practice2Analyzer:
    def __init__(self):
        self.collections = [
            ("Coll-1-10", "practice2_data/01-Text_Only-Ascii-Coll-1-10-NoSem.gz", 55),
            ("Coll-11-20", "practice2_data/02-Text_Only-Ascii-Coll-11-20-NoSem.gz", 52),
            ("Coll-21-50", "practice2_data/03-Text_Only-Ascii-Coll-21-50-NoSem.gz", 103),
            ("Coll-51-100", "practice2_data/04-Text_Only-Ascii-Coll-51-100-NoSem.gz", 96),
            ("Coll-101-200", "practice2_data/05-Text_Only-Ascii-Coll-101-200-NoSem.gz", 357),
            ("Coll-201-500", "practice2_data/06-Text_Only-Ascii-Coll-201-500-NoSem.gz", 559),
            ("Coll-501-1000", "practice2_data/07-Text_Only-Ascii-Coll-501-1000-NoSem.gz", 747),
            ("Coll-1001-2000", "practice2_data/08-Text_Only-Ascii-Coll-1001-2000-NoSem.gz", 1200),
            ("Coll-2001-5000", "practice2_data/09-Text_Only-Ascii-Coll-2001-5000-NoSem.gz", 4100)
        ]
        
        self.all_results = {
            'base': [],
            'stopwords': [],
            'stemming': []
        }
    
    def exercise_1_performance_analysis(self):
        """Exercice 1: Analyse de performance sur collections croissantes"""
        print("=" * 60)
        print("EXERCICE 1: ANALYSE DE PERFORMANCE")
        print("=" * 60)
        
        results = []
        
        for name, filename, expected_size in self.collections:
            if not os.path.exists(filename):
                print(f"❌ Fichier manquant: {filename}")
                continue
            
            index = AdvancedInvertedIndex()
            
            # Mesure du temps d'indexation
            start_time = time.time()
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            end_time = time.time()
            
            if indexing_time is None:
                indexing_time = end_time - start_time
            
            # Statistiques
            stats = index.get_global_statistics()
            actual_size = os.path.getsize(filename) / 1024  # Taille en Ko
            
            result = {
                'name': name,
                'file': filename,
                'expected_size_kb': expected_size,
                'actual_size_kb': actual_size,
                'time_seconds': indexing_time,
                'statistics': stats
            }
            
            results.append(result)
            print(f"✅ {name}: {indexing_time:.2f}s, {stats['total_documents']} docs, {stats['vocabulary_size']} termes")
        
        self.all_results['base'] = results
        self.plot_exercise_1(results)
        return results
    
    def plot_exercise_1(self, results):
        """Graphique pour l'exercice 1"""
        sizes = [r['actual_size_kb'] for r in results]
        times = [r['time_seconds'] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(sizes, times, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Temps indexation (s)')
        plt.title('EXERCICE 1: Performance Indexation\nTemps vs Taille Collection')
        plt.grid(True, alpha=0.3)
        
        # Ajouter les annotations
        for i, result in enumerate(results):
            plt.annotate(result['name'], (sizes[i], times[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('graphs/exercise1_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def exercise_2_statistics_analysis(self):
        """Exercice 2: Analyse statistique détaillée"""
        print("\n" + "=" * 60)
        print("EXERCICE 2: STATISTIQUES DE LA COLLECTION")
        print("=" * 60)
        
        # Réutiliser les résultats de l'exercice 1
        base_results = self.all_results['base']
        
        # Afficher les statistiques pour la plus grande collection
        if base_results:
            last_result = base_results[-1]
            stats = last_result['statistics']
            print(f"\n📊 Statistiques pour {last_result['name']}:")
            print(f"  • Longueur moyenne des documents: {stats['avg_document_length']:.1f} termes")
            print(f"  • Longueur moyenne des termes: {stats['avg_term_length']:.1f} caractères")
            print(f"  • Taille du vocabulaire: {stats['vocabulary_size']} termes")
            print(f"  • Nombre total de documents: {stats['total_documents']}")
            print(f"  • Nombre total de tokens: {stats['total_tokens']}")
        
        # Graphiques d'évolution des statistiques
        self.plot_exercise_2_evolution(base_results)
        
        return base_results
    
    def plot_exercise_2_evolution(self, results):
        """Graphiques d'évolution pour l'exercice 2.2"""
        if not results:
            return
            
        sizes = [r['actual_size_kb'] for r in results]
        
        plt.figure(figsize=(15, 10))
        
        # Graphique 1: Longueur moyenne des documents
        plt.subplot(2, 2, 1)
        avg_doc_lengths = [r['statistics']['avg_document_length'] for r in results]
        plt.plot(sizes, avg_doc_lengths, 'go-', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes par document')
        plt.title('2.2 - Longueur moyenne des documents')
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Longueur moyenne des termes
        plt.subplot(2, 2, 2)
        avg_term_lengths = [r['statistics']['avg_term_length'] for r in results]
        plt.plot(sizes, avg_term_lengths, 'ro-', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Caractères par terme')
        plt.title('2.2 - Longueur moyenne des termes')
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Taille du vocabulaire (Loi de Heaps)
        plt.subplot(2, 2, 3)
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        total_tokens = [r['statistics']['total_tokens'] for r in results]
        plt.plot(total_tokens, vocab_sizes, 'bo-', linewidth=2)
        plt.xlabel('Nombre total de tokens')
        plt.ylabel('Termes uniques')
        plt.title('2.2 - Taille du vocabulaire (Loi de Heaps)')
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Temps d'indexation
        plt.subplot(2, 2, 4)
        times = [r['time_seconds'] for r in results]
        plt.plot(sizes, times, 'mo-', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Temps (s)')
        plt.title('2.2 - Temps d\'indexation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('graphs/exercise2_statistics_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def exercise_3_stop_words(self):
        """Exercice 3: Analyse avec stop words"""
        print("\n" + "=" * 60)
        print("EXERCICE 3: ANALYSE AVEC STOP WORDS")
        print("=" * 60)
        
        results = []
        
        for name, filename, expected_size in self.collections:
            if not os.path.exists(filename):
                continue
            
            index = AdvancedInvertedIndex()
            index.stop_word_active = True
            index.load_stop_words()
            
            start_time = time.time()
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            end_time = time.time()
            
            if indexing_time is None:
                indexing_time = end_time - start_time
            
            stats = index.get_global_statistics()
            actual_size = os.path.getsize(filename) / 1024
            
            result = {
                'name': name,
                'file': filename,
                'actual_size_kb': actual_size,
                'time_seconds': indexing_time,
                'statistics': stats
            }
            
            results.append(result)
            print(f"✅ {name} avec stop words: {stats['vocabulary_size']} termes (réduction: {self.calculate_reduction(name, stats['vocabulary_size'], 'base')}%)")
        
        self.all_results['stopwords'] = results
        self.plot_exercise_3_evolution(results)
        return results
    
    def calculate_reduction(self, collection_name, new_vocab_size, reference_type):
        """Calcule le pourcentage de réduction du vocabulaire"""
        if not self.all_results[reference_type]:
            return 0
        
        # Trouver la collection correspondante
        for result in self.all_results[reference_type]:
            if result['name'] == collection_name:
                original_vocab = result['statistics']['vocabulary_size']
                if original_vocab > 0:
                    reduction = ((original_vocab - new_vocab_size) / original_vocab) * 100
                    return round(reduction, 1)
        return 0
    
    def plot_exercise_3_evolution(self, results):
        """Graphique d'évolution pour l'exercice 3.2"""
        if not results or not self.all_results['base']:
            return
            
        sizes = [r['actual_size_kb'] for r in results]
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, vocab_sizes, 'ro-', linewidth=2, label='Avec stop words')
        
        # Comparaison avec la baseline
        base_sizes = [r['actual_size_kb'] for r in self.all_results['base']]
        base_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['base']]
        plt.plot(base_sizes, base_vocab, 'bo-', linewidth=2, label='Baseline')
        
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes uniques')
        plt.title('EXERCICE 3.2 - Vocabulaire avec Stop Words')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/exercise3_stopwords_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def exercise_4_stemming(self):
        """Exercice 4: Analyse avec stemming"""
        print("\n" + "=" * 60)
        print("EXERCICE 4: ANALYSE AVEC STEMMING")
        print("=" * 60)
        
        results = []
        
        for name, filename, expected_size in self.collections:
            if not os.path.exists(filename):
                continue
            
            index = AdvancedInvertedIndex()
            index.stop_word_active = True
            index.stemmer_active = True
            index.load_stop_words()
            
            start_time = time.time()
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            end_time = time.time()
            
            if indexing_time is None:
                indexing_time = end_time - start_time
            
            stats = index.get_global_statistics()
            actual_size = os.path.getsize(filename) / 1024
            
            result = {
                'name': name,
                'file': filename,
                'actual_size_kb': actual_size,
                'time_seconds': indexing_time,
                'statistics': stats
            }
            
            results.append(result)
            reduction_stem = self.calculate_reduction(name, stats['vocabulary_size'], 'base')
            reduction_stop = self.calculate_reduction(name, stats['vocabulary_size'], 'stopwords')
            print(f"✅ {name} avec stemming: {stats['vocabulary_size']} termes (réduction: {reduction_stem}% depuis base, {reduction_stop}% depuis stop words)")
        
        self.all_results['stemming'] = results
        self.plot_exercise_4_evolution(results)
        return results
    
    def plot_exercise_4_evolution(self, results):
        """Graphique d'évolution pour l'exercice 4.2"""
        if not results or not self.all_results['base'] or not self.all_results['stopwords']:
            return
            
        sizes = [r['actual_size_kb'] for r in results]
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, vocab_sizes, 'go-', linewidth=2, label='Avec stemming')
        
        # Comparaison avec les autres approches
        base_sizes = [r['actual_size_kb'] for r in self.all_results['base']]
        base_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['base']]
        plt.plot(base_sizes, base_vocab, 'bo-', linewidth=2, label='Baseline')
        
        stop_sizes = [r['actual_size_kb'] for r in self.all_results['stopwords']]
        stop_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['stopwords']]
        plt.plot(stop_sizes, stop_vocab, 'ro-', linewidth=2, label='Stop words')
        
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes uniques')
        plt.title('EXERCICE 4.2 - Vocabulaire avec Stemming')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('graphs/exercise4_stemming_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Génère le rapport practice2_report.txt complet"""
        print("\n" + "=" * 60)
        print("GÉNÉRATION DU RAPPORT COMPLET")
        print("=" * 60)
        
        with open('practice2_report.txt', 'w', encoding='utf-8') as f:
            f.write("PRACTICAL SESSION 2 - RAPPORT COMPLET\n")
            f.write("=" * 50 + "\n\n")
            
            # Exercice 1
            f.write("=== EXERCICE 1: PERFORMANCE ===\n")
            f.write("Collection | Taille (Ko) | Temps (s) | Documents | Vocabulaire\n")
            f.write("-" * 80 + "\n")
            for result in self.all_results['base']:
                f.write(f"{result['name']} | {result['actual_size_kb']:.0f} | {result['time_seconds']:.2f} | ")
                f.write(f"{result['statistics']['total_documents']} | {result['statistics']['vocabulary_size']}\n")
            
            # Exercice 2
            f.write("\n=== EXERCICE 2: STATISTIQUES ===\n")
            last_result = self.all_results['base'][-1] if self.all_results['base'] else None
            if last_result:
                stats = last_result['statistics']
                f.write(f"Longueur moyenne documents: {stats['avg_document_length']:.1f} termes\n")
                f.write(f"Longueur moyenne termes: {stats['avg_term_length']:.1f} caractères\n")
                f.write(f"Taille vocabulaire: {stats['vocabulary_size']} termes\n")
                f.write(f"Nombre total documents: {stats['total_documents']}\n")
                f.write(f"Nombre total tokens: {stats['total_tokens']}\n")
            
            # Exercice 3
            f.write("\n=== EXERCICE 3: STOP WORDS ===\n")
            if self.all_results['base'] and self.all_results['stopwords']:
                base_last = self.all_results['base'][-1]
                stop_last = self.all_results['stopwords'][-1]
                base_vocab = base_last['statistics']['vocabulary_size']
                stop_vocab = stop_last['statistics']['vocabulary_size']
                reduction = ((base_vocab - stop_vocab) / base_vocab) * 100
                f.write(f"Vocabulaire original: {base_vocab} termes\n")
                f.write(f"Avec stop words: {stop_vocab} termes\n")
                f.write(f"Réduction: {reduction:.1f}%\n")
            
            # Exercice 4
            f.write("\n=== EXERCICE 4: STEMMING ===\n")
            if self.all_results['base'] and self.all_results['stemming']:
                base_last = self.all_results['base'][-1]
                stem_last = self.all_results['stemming'][-1]
                base_vocab = base_last['statistics']['vocabulary_size']
                stem_vocab = stem_last['statistics']['vocabulary_size']
                reduction_total = ((base_vocab - stem_vocab) / base_vocab) * 100
                
                if self.all_results['stopwords']:
                    stop_last = self.all_results['stopwords'][-1]
                    stop_vocab = stop_last['statistics']['vocabulary_size']
                    reduction_stem_only = ((stop_vocab - stem_vocab) / stop_vocab) * 100
                    f.write(f"Vocabulaire original: {base_vocab} termes\n")
                    f.write(f"Avec stop words: {stop_vocab} termes\n")
                    f.write(f"Avec stemming: {stem_vocab} termes\n")
                    f.write(f"Réduction totale: {reduction_total:.1f}%\n")
                    f.write(f"Réduction stemming seul: {reduction_stem_only:.1f}%\n")
                else:
                    f.write(f"Vocabulaire original: {base_vocab} termes\n")
                    f.write(f"Avec stemming: {stem_vocab} termes\n")
                    f.write(f"Réduction totale: {reduction_total:.1f}%\n")
            
            # Observations
            f.write("\n=== OBSERVATIONS GLOBALES ===\n")
            f.write("1. Loi de Heaps: Croissance du vocabulaire observée\n")
            f.write("2. Stop words: Réduction significative du vocabulaire\n")
            f.write("3. Stemming: Réduction additionnelle importante\n")
            f.write("4. Performance: Temps d'indexation scalable\n")
            f.write("5. Tokenisation: Respect de la consigne 'terms without digits or special characters'\n")
        
        print("✅ Rapport généré: practice2_report.txt")
    
    def plot_final_comparison(self):
        """Graphique final de comparaison des trois approches"""
        if not all(self.all_results.values()):
            print("❌ Données manquantes pour la comparaison finale")
            return
        
        plt.figure(figsize=(15, 12))
        
        # Données communes (tailles)
        sizes = [r['actual_size_kb'] for r in self.all_results['base']]
        
        # Graphique 1: Vocabulaire
        plt.subplot(2, 2, 1)
        plt.plot(sizes, [r['statistics']['vocabulary_size'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        plt.plot(sizes, [r['statistics']['vocabulary_size'] for r in self.all_results['stopwords']], 
                'ro-', label='Stop words', linewidth=2)
        plt.plot(sizes, [r['statistics']['vocabulary_size'] for r in self.all_results['stemming']], 
                'go-', label='Stemming', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes uniques')
        plt.title('Comparaison - Taille du vocabulaire')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Temps d'indexation
        plt.subplot(2, 2, 2)
        plt.plot(sizes, [r['time_seconds'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        plt.plot(sizes, [r['time_seconds'] for r in self.all_results['stopwords']], 
                'ro-', label='Stop words', linewidth=2)
        plt.plot(sizes, [r['time_seconds'] for r in self.all_results['stemming']], 
                'go-', label='Stemming', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Temps (s)')
        plt.title('Comparaison - Temps d\'indexation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Longueur documents
        plt.subplot(2, 2, 3)
        plt.plot(sizes, [r['statistics']['avg_document_length'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        plt.plot(sizes, [r['statistics']['avg_document_length'] for r in self.all_results['stopwords']], 
                'ro-', label='Stop words', linewidth=2)
        plt.plot(sizes, [r['statistics']['avg_document_length'] for r in self.all_results['stemming']], 
                'go-', label='Stemming', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes/document')
        plt.title('Comparaison - Longueur documents')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Réduction cumulative
        plt.subplot(2, 2, 4)
        base_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['base']]
        stop_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['stopwords']]
        stem_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['stemming']]
        
        reduction_stop = [((base - stop) / base) * 100 for base, stop in zip(base_vocab, stop_vocab)]
        reduction_stem = [((base - stem) / base) * 100 for base, stem in zip(base_vocab, stem_vocab)]
        
        plt.plot(sizes, reduction_stop, 'ro-', label='Stop words', linewidth=2)
        plt.plot(sizes, reduction_stem, 'go-', label='Stemming', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Réduction vocabulaire (%)')
        plt.title('Comparaison - Réduction du vocabulaire')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('graphs/final_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Fonction principale"""
    # Créer le dossier graphs s'il n'existe pas
    os.makedirs('graphs', exist_ok=True)
    
    analyzer = Practice2Analyzer()
    
    # Exécuter toutes les analyses
    analyzer.exercise_1_performance_analysis()
    analyzer.exercise_2_statistics_analysis()
    analyzer.exercise_3_stop_words()
    analyzer.exercise_4_stemming()
    
    # Générer les rapports et graphiques finaux
    analyzer.generate_comprehensive_report()
    analyzer.plot_final_comparison()
    
    print("\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("📊 Graphiques sauvegardés dans le dossier 'graphs/'")
    print("📄 Rapport généré: 'practice2_report.txt'")

if __name__ == "__main__":
    main()