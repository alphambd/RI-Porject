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
    
    def load_files(self, path, print_file_name = False):
        """Charge automatiquement les fichiers depuis le dossier spécifié"""
        collections = []
        
        if not os.path.exists(path):
            print(f" Dossier non trouvé: {path}")
            return collections
        
        # Lister tous les fichiers .gz du dossier
        gz_files = [f for f in os.listdir(path) if f.endswith('.gz')]
        gz_files.sort()
        
        # Créer la liste avec des noms simples file1, file2, etc.
        #print("Fichiers chargés : ")
        for i, filename in enumerate(gz_files, 1):
            filepath = os.path.join(path, filename)
            simple_name = f"file{i}"
            collections.append((simple_name, filepath))
            if print_file_name:
                print(f"  {filename}")
        
        #print(f" {len(collections)} fichiers chargés depuis {path}")
        return collections
    
    def run_indexation_experiment(self, config_name, stop_words=False, stemming=False):
        """Exécute l'indexation avec une configuration spécifique"""
        print(f"\n Configuration: {config_name}")
        
        results = []
        index = AdvancedInvertedIndex()
        
        if stop_words:
            index.stop_word_active = True
            index.load_stop_words()
        
        if stemming:
            index.stemmer_active = True
        
        for name, filename in self.collections:
            if not os.path.exists(filename):
                continue
            
            # Mesure du temps d'indexation
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            
            if indexing_time is None:
                continue
            
            # Statistiques
            #size = os.path.getsize(filename) / 1024 # 
            stats = index.get_global_statistics()
            result = {
                'name': name,
                'file': filename,
                'total_tokens': stats['total_tokens'],
                'time_seconds': indexing_time,
                'statistics': stats
            }
            
            results.append(result)
            print(f"  {name}: {stats['total_documents']} docs, {stats['vocabulary_size']} termes, {stats['total_tokens']} tokens, indexé en {indexing_time:.2f}s")
            
            # Réinitialiser pour le fichier suivant
            index.reset()
            if stop_words:
                index.stop_word_active = True
                index.load_stop_words()
            if stemming:
                index.stemmer_active = True
        
        self.all_results[config_name] = results
        return results
    
    def compute_statistics(self, results):
        """Calcule et affiche les statistiques pour un ensemble de résultats"""
        if not results:
            return None
        
        last_result = results[-1]
        stats = last_result['statistics']
        
        print(f"\n Statistiques pour {last_result['name']}:")
        #print(f"  • Documents: {stats['total_documents']}")
        #print(f"  • Tokens totaux: {stats['total_tokens']}")
        print(f"  • Longueur moyenne documents: {stats['avg_document_length']:.2f} termes")
        print(f"  • Longueur moyenne termes: {stats['avg_term_length']:.2f} caractères")
        print(f"  • Taille vocabulaire: {stats['vocabulary_size']} termes distincts")
        print(f"  • Temps indexation: {last_result['time_seconds']:.2f}s")
        
        return stats

    def plot_single_metric(self, x_data, y_datas, labels, colors, x_label, y_label, title, filename):
        """Génère un graphique pour une métrique spécifique - VERSION CORRIGÉE"""
        plt.figure(figsize=(10, 6))
        
        for y_data, label, color in zip(y_datas, labels, colors):
            if len(y_data) == len(x_data):
                # Données complètes (baseline) - ligne avec marqueurs
                if color in ['bo-', 'ro-', 'go-', 'mo-']:  # Format matplotlib traditionnel
                    plt.plot(x_data, y_data, color, label=label, linewidth=2, markersize=6)
                else:  # Couleur simple
                    plt.plot(x_data, y_data, color=color, marker='o', linestyle='-', 
                            label=label, linewidth=2, markersize=6)
            elif len(y_data) == 1 and len(x_data) > 0:
                # Données avec un seul point - seulement marqueur
                if color in ['ro', 'go', 'mo']:  # Extraire la couleur du format
                    actual_color = color[0]  # 'r', 'g', 'm', etc.
                    plt.plot(x_data[-1], y_data[0], marker='o', color=actual_color, 
                            label=label, markersize=8, linestyle='')
                else:
                    plt.plot(x_data[-1], y_data[0], marker='o', color=color, 
                            label=label, markersize=8, linestyle='')
            else:
                print(f"Erreur :  Format de données inattendu pour {label}: {len(y_data)} points vs {len(x_data)}")
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # FORMATER L'AXE X POUR ÉVITER LA NOTATION SCIENTIFIQUE (0.0, 0.2, 0.4, ..., x.e)
        plt.ticklabel_format(style='plain', axis='x')
        
        os.makedirs('graphs', exist_ok=True)
        plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self):
        """Génère tous les graphiques de comparaison"""
        if not all(self.all_results.values()):
            print(" Données manquantes pour la comparaison")
            return
        
        base_results = self.all_results['base']
        stop_results = self.all_results['stopwords']
        stem_results = self.all_results['stemming']
        
        # Données communes
        sizes = [r['total_tokens'] for r in base_results]
        
        # 1. Temps d'indexation
        times_base = [r['time_seconds'] for r in base_results]
        times_stop = [stop_results[0]['time_seconds']] if stop_results else []
        times_stem = [stem_results[0]['time_seconds']] if stem_results else []
        
        self.plot_single_metric(
            sizes, [times_base, times_stop, times_stem],
            ['Base', 'Stop words', 'Stemming'], 
            ['bo-', 'red', 'green'],  # Baseline avec format, autres avec couleurs simples
            'Taille collection (Ko)', 'Temps (s)',
            'Comparaison - Temps d\'indexation', 'comparison_time.png'
        )
        
        # 2. Taille du vocabulaire
        vocab_base = [r['statistics']['vocabulary_size'] for r in base_results]
        vocab_stop = [stop_results[0]['statistics']['vocabulary_size']] if stop_results else []
        vocab_stem = [stem_results[0]['statistics']['vocabulary_size']] if stem_results else []
        
        self.plot_single_metric(
            sizes, [vocab_base, vocab_stop, vocab_stem],
            ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
            'Taille collection (Ko)', 'Termes uniques',
            'Comparaison - Taille du vocabulaire', 'comparison_vocabulary.png'
        )
        
        # 3. Longueur moyenne des documents
        doclen_base = [r['statistics']['avg_document_length'] for r in base_results]
        doclen_stop = [stop_results[0]['statistics']['avg_document_length']] if stop_results else []
        doclen_stem = [stem_results[0]['statistics']['avg_document_length']] if stem_results else []
        
        self.plot_single_metric(
            sizes, [doclen_base, doclen_stop, doclen_stem],
            ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
            'Taille collection (Ko)', 'Termes/document',
            'Comparaison - Longueur moyenne documents', 'comparison_doc_length.png'
        )
        
        # 4. Longueur moyenne des termes
        termlen_base = [r['statistics']['avg_term_length'] for r in base_results]
        termlen_stop = [stop_results[0]['statistics']['avg_term_length']] if stop_results else []
        termlen_stem = [stem_results[0]['statistics']['avg_term_length']] if stem_results else []
        
        self.plot_single_metric(
            sizes, [termlen_base, termlen_stop, termlen_stem],
            ['Base', 'Stop words', 'Stemming'], ['bo-', 'red', 'green'],
            'Taille collection (Ko)', 'Caractères/terme',
            'Comparaison - Longueur moyenne termes', 'comparison_term_length.png'
        )
    
    def plot_evolution(self):
        """Génère les graphiques d'évolution pour la baseline"""
        base_results = self.all_results['base']
        
        if not base_results:
            return
        
        sizes = [r['total_tokens'] for r in base_results]
        print("****** SIZES ***** ", sizes)
        # Graphiques d'évolution
        metrics = [
            ([r['statistics']['avg_document_length'] for r in base_results], 
             'Termes par document', 'Longueur moyenne des documents', 'evolution_doc_length.png'),
            
            ([r['statistics']['avg_term_length'] for r in base_results],
             'Caractères par terme', 'Longueur moyenne des termes', 'evolution_term_length.png'),
            
            ([r['statistics']['vocabulary_size'] for r in base_results],
             'Termes uniques', 'Taille du vocabulaire', 'evolution_vocabulary.png'),
            
            ([r['time_seconds'] for r in base_results],
             'Temps (s)', 'Temps d\'indexation', 'evolution_time.png')
        ]
        
        for y_data, y_label, title, filename in metrics:
            self.plot_single_metric(
                sizes, [y_data], ['Base'], ['bo-'],
                'Taille collection (Ko)', y_label, title, filename
            )
    
def main():
    """Fonction principale refactorisée"""
    # Initialisation
    os.makedirs('graphs', exist_ok=True)
    analyzer = Practice2Analyzer(data_path="practice2_data")

    
    # Exécution des expériences
    print("\n TEMPS D'INDEXATION")
    print("=" * 60)
    
    # -------------------------------------------
    # EXO 1
    # -------------------------------------------
    analyzer.run_indexation_experiment('base')
    #analyzer.compute_statistics(analyzer.all_results['base'])
    """
    base_results = analyzer.all_results['base']
    
    # Données communes
    sizes = [r['total_tokens'] for r in base_results]
    
    # Courbe du temps d'indexation
    times_base = [r['time_seconds'] for r in base_results]
    
    analyzer.plot_single_metric(
        sizes, [times_base],
        ['Base'], 
        ['bo-'],
        '#mots', 'time (s)',
        '1. indexing time vs size of the coll', 'comparison_time.png'
    )
    
    # -------------------------------------------
    # EXO 2
    # -------------------------------------------
    print("\n COLLECTION STATISTIQUES")
    print("=" * 60)
    analyzer.compute_statistics(analyzer.all_results['base'])
    
    # Longueur moyenne des documents
    doclen_base = [r['statistics']['avg_document_length'] for r in base_results]
    
    analyzer.plot_single_metric(
        sizes, [doclen_base],
        ['Base'], ['bo-'],
        '#mots', '#terms',
        '2.1. avg doc length vs size of the coll', 'graph_doc_length.png'
    )

    # Longueur moyenne des terms 
    termlen_base = [r['statistics']['avg_term_length'] for r in base_results]
        
    analyzer.plot_single_metric(
        sizes, [termlen_base],
        ['Base'], ['bo-'],
        '#mots', '#chars',
        '2.2.  avg terms length vs size of the coll', 'graph_term_length.png'
    )

    # Taille du vocabulaire
    vocab_base = [r['statistics']['vocabulary_size'] for r in base_results]
        
    analyzer.plot_single_metric(
        sizes, [vocab_base],
        ['Base'], ['bo-'],
        '#mots', 'terms',
        '2.3.  vocabulary size vs size of the col', 'graph_vocabulary.png'
    )
    
    
    # -------------------------------------------
    # EXO 3
    # -------------------------------------------
    last_file = [analyzer.collections[-1]]  # Uniquement le dernier fichier
    analyzer.collections = last_file  # Temporairement réduire aux derniers fichiers

    print("\n STOPWORDS")
    print("=" * 60)
    
    analyzer.run_indexation_experiment('stopwords', True)
    analyzer.compute_statistics(analyzer.all_results['stopwords'])
    stop_results = analyzer.all_results['stopwords']

    # Données communes
    sizes = [r['total_tokens'] for r in base_results] # ? recalculer 
    
    # Longueur moyenne des documents
    doclen_stop = [stop_results[0]['statistics']['avg_document_length']] if stop_results else []
    
    analyzer.plot_single_metric(
        sizes, [doclen_stop],
        ['Stopwords'], ['red'],
        '#mots', '#terms',
        '3.1. avg doc length vs size of the coll', 'graph_doc_length.png'
    )
    
    # Longueur moyenne des terms 
    termlen_stop = [stop_results[0]['statistics']['avg_term_length']] if stop_results else []
        
    analyzer.plot_single_metric(
        sizes, [termlen_stop],
        ['Stopwords'], ['red'],
        '#mots', '#chars',
        '3.2.  avg terms length vs size of the coll', 'graph_term_length.png'
    )

    # Taille du vocabulaire
    vocab_stop = [stop_results[0]['statistics']['vocabulary_size']] if stop_results else []
        
    analyzer.plot_single_metric(
        sizes, [vocab_stop],
        ['Stopwords'], ['red'],
        '#mots', 'terms',
        '3.3.  vocabulary size vs size of the col', 'graph_vocabulary.png'
    )

    # -------------------------------------------
    # EXO 4
    # -------------------------------------------
    print("\n STOPWORDS and STEMMING")
    print("=" * 60)

    analyzer.run_indexation_experiment('stemming', True, True)
    analyzer.compute_statistics(analyzer.all_results['stemming'])
    stem_results = analyzer.all_results['stemming']

    # Données communes
    sizes = [r['total_tokens'] for r in base_results] # ? recalculer 
    
    # Longueur moyenne des documents
    doclen_stem = [stem_results[0]['statistics']['avg_document_length']] if stem_results else []
    
    analyzer.plot_single_metric(
        sizes, [doclen_stem],
        ['Stemming'], ['green'],
        '#mots', '#terms',
        '4.1. avg doc length vs size of the coll', 'graph_doc_length.png'
    )
    
    # Longueur moyenne des terms 
    termlen_stem = [stem_results[0]['statistics']['avg_term_length']] if stem_results else []
        
    analyzer.plot_single_metric(
        sizes, [termlen_stem],
        ['Stemming'], ['green'],
        '#mots', '#chars',
        '4.2.  avg terms length vs size of the coll', 'graph_term_length.png'
    )

    # Taille du vocabulaire
    vocab_stem = [stem_results[0]['statistics']['vocabulary_size']] if stem_results else []
        
    analyzer.plot_single_metric(
        sizes, [vocab_stem],
        ['Stemming'], ['green'],
        '#mots', 'terms',
        '4.3.  vocabulary size vs size of the col', 'graph_vocabulary.png'
    )
    
    
    """
    last_file = [analyzer.collections[-1]]  # Uniquement le dernier fichier
    analyzer.collections = last_file  # Temporairement réduire aux derniers fichiers
    
    analyzer.run_indexation_experiment('stopwords', stop_words=True)
    analyzer.compute_statistics(analyzer.all_results['stopwords'])
    
    # Expérience 3: Stemming (dernier fichier seulement)
    print("\n" + "=" * 60)
    analyzer.run_indexation_experiment('stemming', stop_words=True, stemming=True)
    analyzer.compute_statistics(analyzer.all_results['stemming'])
    
    # Restaurer toutes les collections pour les graphiques
    analyzer.collections = analyzer.load_files("practice2_data")
    
    # Génération des graphiques
    print("\n GÉNÉRATION DES GRAPHIQUES")
    print("=" * 60)
    
    #analyzer.plot_evolution()        # Graphiques d'évolution (baseline)

    analyzer.plot_comparison()       # Graphiques de comparaison
    
    
if __name__ == "__main__":
    main()