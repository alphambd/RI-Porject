import os
import matplotlib.pyplot as plt
import time
import gzip
import json
from advanced_indexer import AdvancedInvertedIndex

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
            #print(f"✅ {name}: {indexing_time:.2f}s, {stats['total_documents']} docs, {stats['vocabulary_size']} termes")
        
        self.all_results['base'] = results
        #self.plot_exercise_1(results)
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
        
        # Graphiques d'évolution des statistiques
        self.plot_exercise_2_evolution(base_results)
        
        return base_results
    
    def plot_exercise_2_evolution(self, results):
        """Graphiques d'évolution pour l'exercice 2.2"""
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
        plt.plot(sizes, vocab_sizes, 'bo-', linewidth=2)
        plt.xlabel('Taille collection (Ko)')
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
            print(f"✅ {name} avec stop words: {stats['vocabulary_size']} termes")
        
        self.all_results['stopwords'] = results
        self.plot_exercise_3_evolution(results)
        return results
    
    def plot_exercise_3_evolution(self, results):
        """Graphique d'évolution pour l'exercice 3.2"""
        sizes = [r['actual_size_kb'] for r in results]
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, vocab_sizes, 'ro-', linewidth=2, label='Avec stop words')
        
        # Comparaison avec la baseline
        if self.all_results['base']:
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
            print(f"✅ {name} avec stemming: {stats['vocabulary_size']} termes")
        
        self.all_results['stemming'] = results
        self.plot_exercise_4_evolution(results)
        return results
    
    def plot_exercise_4_evolution(self, results):
        """Graphique d'évolution pour l'exercice 4.2"""
        sizes = [r['actual_size_kb'] for r in results]
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, vocab_sizes, 'go-', linewidth=2, label='Avec stemming')
        
        # Comparaison avec les autres approches
        if self.all_results['base']:
            base_sizes = [r['actual_size_kb'] for r in self.all_results['base']]
            base_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['base']]
            plt.plot(base_sizes, base_vocab, 'bo-', linewidth=2, label='Baseline')
        
        if self.all_results['stopwords']:
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
            
            # Exercice 3
            f.write("\n=== EXERCICE 3: STOP WORDS ===\n")
            if self.all_results['base'] and self.all_results['stopwords']:
                base_vocab = self.all_results['base'][-1]['statistics']['vocabulary_size']
                stop_vocab = self.all_results['stopwords'][-1]['statistics']['vocabulary_size']
                reduction = ((base_vocab - stop_vocab) / base_vocab) * 100
                f.write(f"Vocabulaire original: {base_vocab} termes\n")
                f.write(f"Avec stop words: {stop_vocab} termes\n")
                f.write(f"Réduction: {reduction:.1f}%\n")
            
            # Exercice 4
            f.write("\n=== EXERCICE 4: STEMMING ===\n")
            if self.all_results['base'] and self.all_results['stemming']:
                base_vocab = self.all_results['base'][-1]['statistics']['vocabulary_size']
                stem_vocab = self.all_results['stemming'][-1]['statistics']['vocabulary_size']
                reduction = ((base_vocab - stem_vocab) / base_vocab) * 100
                f.write(f"Vocabulaire original: {base_vocab} termes\n")
                f.write(f"Avec stemming: {stem_vocab} termes\n")
                f.write(f"Réduction totale: {reduction:.1f}%\n")
            
            # Observations
            f.write("\n=== OBSERVATIONS GLOBALES ===\n")
            f.write("1. Loi de Heaps: Croissance du vocabulaire observée\n")
            f.write("2. Stop words: Réduction significative du vocabulaire\n")
            f.write("3. Stemming: Réduction additionnelle importante\n")
            f.write("4. Performance: Temps d'indexation scalable\n")
        
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
    
    stats = analyzer.exercise_1_performance_analysis()
    print("\n\navg document length:", stats[-1]['statistics']['avg_document_length'])
    print("avg term length:", stats[-1]['statistics']['avg_term_length'])
    print("vocabulary size:", stats[-1]['statistics']['vocabulary_size'])
    

    """
    # Initialiser l'analyseur
    analyzer = Practice2Analyzer()
    
    # Exécuter tous les exercices
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
    """
if __name__ == "__main__":
    main()
    """
    indexer = AdvancedInvertedIndex()
    analyser = Practice2Analyzer()

    string = "This is a sample text, with punctuation! Let's see: how it works."
    tokens = indexer.preprocess_text(string)
    print(tokens)
    """