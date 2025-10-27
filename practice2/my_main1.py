import os
import matplotlib.pyplot as plt
import time
import gzip
import json
from my_indexer import AdvancedInvertedIndex

class Practice2Analyzer:
    def __init__(self, data_path="practice2_data"):
        self.data_path = data_path
        self.collections = self.load_files(data_path)
        self.all_results = {
            'base': [],
            'stopwords': [],
            'stemming': []
        }
    
    def load_files(self, path):
        """
        Charge automatiquement les fichiers depuis le dossier spécifié
        Retourne une liste de tuples: (nom_simple, chemin_fichier)
        """
        collections = []
        
        if not os.path.exists(path):
            print(f"❌ Dossier non trouvé: {path}")
            return collections
        
        # Lister tous les fichiers .gz du dossier
        gz_files = [f for f in os.listdir(path) if f.endswith('.gz')]
        gz_files.sort()  # Tri pour avoir un ordre cohérent
        
        # Créer la liste avec des noms simples file1, file2, etc.
        for i, filename in enumerate(gz_files, 1):
            filepath = os.path.join(path, filename)
            simple_name = f"file{i}"
            collections.append((simple_name, filepath))
            print(f"✅ {simple_name}: {filename}")
        
        print(f"📁 {len(collections)} fichiers chargés depuis {path}")
        return collections
    
    def exercise_1_performance_analysis(self):
        """Exercice 1: Analyse de performance sur collections croissantes"""
        print("=" * 60)
        print("EXERCICE 1: ANALYSE DE PERFORMANCE")
        print("=" * 60)
        
        if not self.collections:
            print("❌ Aucune collection chargée. Vérifiez le chemin des données.")
            return []
        
        results = []
        
        for name, filename in self.collections:
            if not os.path.exists(filename):
                print(f"❌ Fichier manquant: {filename}")
                continue
            
            # Créer un NOUVEL index pour chaque fichier
            index = AdvancedInvertedIndex()
            
            # Mesure du temps d'indexation
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            
            if indexing_time is None:
                continue
            
            # Statistiques
            stats = index.get_global_statistics()
            actual_size = os.path.getsize(filename) / 1024  # Taille en Ko
            
            result = {
                'name': name,
                'file': filename,
                'actual_size_kb': actual_size,
                'time_seconds': indexing_time,
                'statistics': stats
            }
            
            results.append(result)
            print(f"  {name}: {indexing_time:.2f}s, {stats['total_documents']} docs, {stats['vocabulary_size']} termes")
        
        self.all_results['base'] = results
        self.plot_exercise_1(results)
        return results
    
    def plot_exercise_1(self, results):
        """Graphique pour l'exercice 1"""
        if not results:
            return []
            
        sizes = [r['actual_size_kb'] for r in results]
        times = [r['time_seconds'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Taille de la collection (Ko)')
        plt.ylabel('Temps d\'indexation (secondes)')
        plt.title('Exercice 1: Temps d\'indexation vs Taille de la collection')
        plt.grid(True, alpha=0.3)
        
        # Sauvegarde du graphique
        os.makedirs('graphs', exist_ok=True)
        plt.savefig('graphs/exercise1_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

    def exercise_2_statistics_analysis(self):
        """Exercice 2: Analyse statistique détaillée"""
        print("\n" + "=" * 60)
        print("EXERCICE 2: STATISTIQUES DE LA COLLECTION")
        print("=" * 60)
        
        # Réutiliser les résultats de l'exercice 1
        base_results = self.all_results['base']
        
        if not base_results:
            print("❌ Aucun résultat disponible pour l'exercice 2")
            return []
        
        # Affichage des statistiques pour le dernier fichier (le plus grand)
        last_stats = base_results[-1]['statistics']
        print(f"\n📊 Statistiques pour {base_results[-1]['name']}:")
        print(f"  • Longueur moyenne des documents: {last_stats['avg_document_length']:.2f} termes")
        print(f"  • Longueur moyenne des termes: {last_stats['avg_term_length']:.2f} caractères")
        print(f"  • Taille du vocabulaire: {last_stats['vocabulary_size']} termes distincts")
        print(f"  • Nombre total de documents: {last_stats['total_documents']}")
        print(f"  • Nombre total de tokens: {last_stats['total_tokens']}")
        
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
        plt.xlabel('Taille de la collection (Ko)')
        plt.ylabel('Termes par document')
        plt.title('Exercice 2.2 - Longueur moyenne des documents')
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Longueur moyenne des termes
        plt.subplot(2, 2, 2)
        avg_term_lengths = [r['statistics']['avg_term_length'] for r in results]
        plt.plot(sizes, avg_term_lengths, 'ro-', linewidth=2)
        plt.xlabel('Taille de la collection (Ko)')
        plt.ylabel('Caractères par terme')
        plt.title('Exercice 2.2 - Longueur moyenne des termes')
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Taille du vocabulaire
        plt.subplot(2, 2, 3)
        vocab_sizes = [r['statistics']['vocabulary_size'] for r in results]
        plt.plot(sizes, vocab_sizes, 'bo-', linewidth=2)
        plt.xlabel('Taille de la collection (Ko)')
        plt.ylabel('Termes uniques')
        plt.title('Exercice 2.2 - Taille du vocabulaire')
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Temps d'indexation
        plt.subplot(2, 2, 4)
        times = [r['time_seconds'] for r in results]
        plt.plot(sizes, times, 'mo-', linewidth=2)
        plt.xlabel('Taille de la collection (Ko)')
        plt.ylabel('Temps (s)')
        plt.title('Exercice 2.2 - Temps d\'indexation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('graphs/exercise2_statistics_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def exercise_3_stop_words(self):
        """Exercice 3: Analyse avec stop words (sur le dernier fichier)"""
        print("\n" + "=" * 60)
        print("EXERCICE 3: ANALYSE AVEC STOP WORDS")
        print("=" * 60)
        
        if not self.collections:
            print("❌ Aucune collection chargée")
            return None
        
        # Se concentrer sur le dernier fichier (le plus grand)
        last_file = self.collections[-1]  # Dernier fichier
        name, filename = last_file
        
        if not os.path.exists(filename):
            print(f"❌ Fichier manquant: {filename}")
            return None
        
        # Index avec stop words
        index_stop = AdvancedInvertedIndex()
        index_stop.stop_word_active = True
        index_stop.load_stop_words()
        
        indexing_time = index_stop.build_from_file(filename, verbose=True, print_index=False)
        stats_stop = index_stop.get_global_statistics()
        
        # Comparaison avec la baseline (dernier fichier)
        base_stats = self.all_results['base'][-1]['statistics'] if self.all_results['base'] else None
        
        print(f"\n📊 COMPARAISON pour {name}:")
        print(f"  AVANT stop words:")
        if base_stats:
            print(f"    • Vocabulaire: {base_stats['vocabulary_size']} termes")
            print(f"    • Long. doc moyenne: {base_stats['avg_document_length']:.2f} termes")
            print(f"    • Long. terme moyenne: {base_stats['avg_term_length']:.2f} caractères")
        
        print(f"  APRÈS stop words:")
        print(f"    • Vocabulaire: {stats_stop['vocabulary_size']} termes")
        print(f"    • Long. doc moyenne: {stats_stop['avg_document_length']:.2f} termes")
        print(f"    • Long. terme moyenne: {stats_stop['avg_term_length']:.2f} caractères")
        
        if base_stats:
            reduction = ((base_stats['vocabulary_size'] - stats_stop['vocabulary_size']) / base_stats['vocabulary_size']) * 100
            print(f"    • Réduction du vocabulaire: {reduction:.1f}%")
        
        result = {
            'name': name,
            'file': filename,
            'time_seconds': indexing_time,
            'statistics': stats_stop
        }
        
        self.all_results['stopwords'] = [result]
        return result

    def exercise_4_stemming(self):
        """Exercice 4: Analyse avec stemming (sur le dernier fichier)"""
        print("\n" + "=" * 60)
        print("EXERCICE 4: ANALYSE AVEC STEMMING")
        print("=" * 60)
        
        if not self.collections:
            print("❌ Aucune collection chargée")
            return None
        
        # Se concentrer sur le dernier fichier (le plus grand)
        last_file = self.collections[-1]
        name, filename = last_file
        
        if not os.path.exists(filename):
            print(f"❌ Fichier manquant: {filename}")
            return None
        
        # Index avec stop words + stemming
        index_stem = AdvancedInvertedIndex()
        index_stem.stop_word_active = True
        index_stem.stemmer_active = True
        index_stem.load_stop_words()
        
        indexing_time = index_stem.build_from_file(filename, verbose=True, print_index=False)
        stats_stem = index_stem.get_global_statistics()
        
        # Comparaison avec les autres approches
        base_stats = self.all_results['base'][-1]['statistics'] if self.all_results['base'] else None
        stop_stats = self.all_results['stopwords'][0]['statistics'] if self.all_results['stopwords'] else None
        
        print(f"\n📊 COMPARAISON pour {name}:")
        
        if base_stats:
            print(f"  Baseline:")
            print(f"    • Vocabulaire: {base_stats['vocabulary_size']} termes")
        
        if stop_stats:
            print(f"  Stop words:")
            print(f"    • Vocabulaire: {stop_stats['vocabulary_size']} termes")
        
        print(f"  Stop words + Stemming:")
        print(f"    • Vocabulaire: {stats_stem['vocabulary_size']} termes")
        print(f"    • Long. doc moyenne: {stats_stem['avg_document_length']:.2f} termes")
        print(f"    • Long. terme moyenne: {stats_stem['avg_term_length']:.2f} caractères")
        
        if base_stats:
            reduction = ((base_stats['vocabulary_size'] - stats_stem['vocabulary_size']) / base_stats['vocabulary_size']) * 100
            print(f"    • Réduction vs baseline: {reduction:.1f}%")
        
        result = {
            'name': name,
            'file': filename,
            'time_seconds': indexing_time,
            'statistics': stats_stem
        }
        
        self.all_results['stemming'] = [result]
        return result

    def plot_final_comparison(self):
        """Graphique final de comparaison des trois approches"""
        if not all(self.all_results.values()):
            print("❌ Données manquantes pour la comparaison finale")
            return
        
        # Vérifier que nous avons des données pour toutes les approches
        if (len(self.all_results['base']) == 0 or 
            len(self.all_results['stopwords']) == 0 or 
            len(self.all_results['stemming']) == 0):
            print("❌ Données incomplètes pour la comparaison finale")
            return
        
        plt.figure(figsize=(15, 12))
        
        # Données communes (tailles) - seulement pour la baseline
        sizes = [r['actual_size_kb'] for r in self.all_results['base']]
        
        # Graphique 1: Vocabulaire
        plt.subplot(2, 2, 1)
        plt.plot(sizes, [r['statistics']['vocabulary_size'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        
        # Pour stopwords et stemming, nous n'avons que le dernier point
        if len(self.all_results['stopwords']) > 0:
            last_stop = self.all_results['stopwords'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']  # Taille du dernier fichier baseline
            plt.plot(last_size, last_stop['statistics']['vocabulary_size'], 
                    'ro', markersize=8, label='Stop words')
        
        if len(self.all_results['stemming']) > 0:
            last_stem = self.all_results['stemming'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']  # Taille du dernier fichier baseline
            plt.plot(last_size, last_stem['statistics']['vocabulary_size'], 
                    'go', markersize=8, label='Stemming')
        
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes uniques')
        plt.title('Comparaison - Taille du vocabulaire')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Temps d'indexation
        plt.subplot(2, 2, 2)
        plt.plot(sizes, [r['time_seconds'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        
        if len(self.all_results['stopwords']) > 0:
            last_stop = self.all_results['stopwords'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']
            plt.plot(last_size, last_stop['time_seconds'], 
                    'ro', markersize=8, label='Stop words')
        
        if len(self.all_results['stemming']) > 0:
            last_stem = self.all_results['stemming'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']
            plt.plot(last_size, last_stem['time_seconds'], 
                    'go', markersize=8, label='Stemming')
        
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Temps (s)')
        plt.title('Comparaison - Temps d\'indexation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Longueur documents
        plt.subplot(2, 2, 3)
        plt.plot(sizes, [r['statistics']['avg_document_length'] for r in self.all_results['base']], 
                'bo-', label='Base', linewidth=2)
        
        if len(self.all_results['stopwords']) > 0:
            last_stop = self.all_results['stopwords'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']
            plt.plot(last_size, last_stop['statistics']['avg_document_length'], 
                    'ro', markersize=8, label='Stop words')
        
        if len(self.all_results['stemming']) > 0:
            last_stem = self.all_results['stemming'][0]
            last_size = self.all_results['base'][-1]['actual_size_kb']
            plt.plot(last_size, last_stem['statistics']['avg_document_length'], 
                    'go', markersize=8, label='Stemming')
        
        plt.xlabel('Taille collection (Ko)')
        plt.ylabel('Termes/document')
        plt.title('Comparaison - Longueur documents')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Réduction cumulative (uniquement pour le dernier point)
        plt.subplot(2, 2, 4)
        
        if (len(self.all_results['base']) > 0 and 
            len(self.all_results['stopwords']) > 0 and 
            len(self.all_results['stemming']) > 0):
            
            base_vocab = [r['statistics']['vocabulary_size'] for r in self.all_results['base']]
            last_base_vocab = base_vocab[-1]
            last_stop_vocab = self.all_results['stopwords'][0]['statistics']['vocabulary_size']
            last_stem_vocab = self.all_results['stemming'][0]['statistics']['vocabulary_size']
            
            # Calcul des réductions pour le dernier point
            reduction_stop = ((last_base_vocab - last_stop_vocab) / last_base_vocab) * 100
            reduction_stem = ((last_base_vocab - last_stem_vocab) / last_base_vocab) * 100
            
            # Affichage sous forme de barres pour une seule valeur
            approaches = ['Stop words', 'Stemming']
            reductions = [reduction_stop, reduction_stem]
            colors = ['red', 'green']
            
            bars = plt.bar(approaches, reductions, color=colors, alpha=0.7)
            plt.ylabel('Réduction vocabulaire (%)')
            plt.title('Réduction du vocabulaire (Coll-2001-5000)')
            plt.grid(True, alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for bar, reduction in zip(bars, reductions):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{reduction:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('graphs/final_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Fonction principale avec chargement automatique"""
    # Créer le dossier graphs s'il n'existe pas
    os.makedirs('graphs', exist_ok=True)
    
    # Initialiser l'analyseur avec chargement automatique
    analyzer = Practice2Analyzer(data_path="practice2_data")
    
    # Vérifier que des fichiers ont été chargés
    if not analyzer.collections:
        print("❌ Aucun fichier trouvé. Vérifiez le dossier practice2_data/")
        return
    
    # Afficher les fichiers chargés
    print("📁 Fichiers chargés:")
    for name, filepath in analyzer.collections:
        filename = os.path.basename(filepath)
        print(f"  {name}: {filename}")
    
    # Exécution de tous les exercices
    exercise1_results = analyzer.exercise_1_performance_analysis()
    exercise2_results = analyzer.exercise_2_statistics_analysis()
    exercise3_result = analyzer.exercise_3_stop_words()
    exercise4_result = analyzer.exercise_4_stemming()
    
    # Graphique de comparaison finale
    analyzer.plot_final_comparison()
    
    # AFFICHAGE SYNTHÉTIQUE EXACTEMENT COMME DANS L'IMAGE
    print("\n" + "=" * 60)
    print("RÉSULTATS SYNTHÉTIQUES")
    print("=" * 60)
    
    # Exercice 1
    if exercise1_results:
        exo1 = exercise1_results[-1]  # Dernier fichier (le plus grand)
        print(f"- Exo 1 : indexation {exo1['time_seconds']:.2f}sec ({exo1['statistics']['total_documents']} docs), linéaire.")
    
    # Exercice 2
    if exercise2_results:
        exo2 = exercise2_results[-1]
        stats = exo2['statistics']
        print(f"- Exo 2 : stats = ({stats['avg_document_length']:.0f} terms, {stats['avg_term_length']:.2f} char, {stats['vocabulary_size']} distinct terms), {exo2['time_seconds']:.2f}sec")
    
    # Exercice 3
    if exercise3_result:
        stats = exercise3_result['statistics']
        print(f"- Exo 3 : stats = ({stats['avg_document_length']:.0f} terms, {stats['avg_term_length']:.2f} char, {stats['vocabulary_size']} distinct terms), {exercise3_result['time_seconds']:.2f}sec")
    
    # Exercice 4
    if exercise4_result:
        stats = exercise4_result['statistics']
        print(f"- Exo 4 : stats = ({stats['avg_document_length']:.0f} terms, {stats['avg_term_length']:.2f} char, {stats['vocabulary_size']} distinct terms), {exercise4_result['time_seconds']:.2f}sec")

if __name__ == "__main__":
    main()