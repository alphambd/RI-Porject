import os
import matplotlib.pyplot as plt
from advanced_indexer import AdvancedInvertedIndex

class StatsAnalyzer:
    def __init__(self, data_path="practice2_data"):
        self.data_path = data_path
        self.collections = self.load_files(data_path)
        self.all_results = {
            'base': [],
            'stopwords': [],
            'stemming': []
        }
    
    def load_files(self, path, print_file_name=False):
        """Charge automatiquement les fichiers depuis le dossier spécifié"""
        collections = []
        
        if not os.path.exists(path):
            print(f" Dossier non trouvé: {path}")
            return collections
        
        gz_files = [f for f in os.listdir(path) if f.endswith('.gz')]
        gz_files.sort()
        
        for i, filename in enumerate(gz_files, 1):
            filepath = os.path.join(path, filename)
            simple_name = f"file{i}"
            collections.append((simple_name, filepath))
            if print_file_name:
                print(f"  {filename}")
        
        return collections
    
    def run_indexation_experiment(self, config_name, stop_words=False, stemming=False, use_all_files=True):
        """Exécute l'indexation avec une configuration spécifique"""
        print(f"\n Configuration: {config_name}")
        
        results = []
        index = AdvancedInvertedIndex()
        
        if stop_words:
            index.stop_word_active = True
            index.load_stop_words()
        
        if stemming:
            index.stemmer_active = True
        
        # Sélection des fichiers à traiter
        files_to_process = self.collections if use_all_files else [self.collections[-1]]
        
        for name, filename in files_to_process:
            if not os.path.exists(filename):
                continue
            
            indexing_time = index.build_from_file(filename, verbose=False, print_index=False)
            
            if indexing_time is None:
                continue
            
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
    
    def compute_statistics(self, results, config_name):
        """Calcule et affiche les statistiques pour un ensemble de résultats"""
        if not results:
            return None
        
        last_result = results[-1]
        stats = last_result['statistics']
        
        print(f"\n {config_name} - Statistiques pour {last_result['name']}:")
        print(f"  • Longueur moyenne documents: {stats['avg_document_length']:.2f} termes")
        print(f"  • Longueur moyenne termes: {stats['avg_term_length']:.2f} caractères")
        print(f"  • Taille vocabulaire: {stats['vocabulary_size']} termes distincts")
        print(f"  • Temps indexation: {last_result['time_seconds']:.2f}s")
        
        return stats

    def plot_single_metric(self, x_data, y_data, label, color, x_label, y_label, title, filename):
        """Génère un graphique simple pour une métrique"""
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, color, label=label, linewidth=2, markersize=6)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='x')
        
        os.makedirs('graphs', exist_ok=True)
        plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_comparison_metric(self, x_data, y_datas, labels, colors, x_label, y_label, title, filename):
            
        """Génère un graphique de comparaison pour une métrique - VERSION CORRIGÉE"""
        plt.figure(figsize=(10, 6))
        
        for y_data, label, color_spec in zip(y_datas, labels, colors):
            if len(y_data) == len(x_data):
                # Données complètes (baseline) - utiliser le format matplotlib
                plt.plot(x_data, y_data, color_spec, label=label, linewidth=2, markersize=6)
            elif len(y_data) == 1 and len(x_data) > 0:
                # Données avec un seul point - utiliser couleur simple
                if color_spec in ['ro', 'go', 'mo']:  # Extraire la couleur du format
                    actual_color = color_spec[0]  # 'r', 'g', 'm', etc.
                    plt.plot(x_data[-1], y_data[0], marker='o', color=actual_color, 
                            label=label, markersize=8, linestyle='')
                else:
                    plt.plot(x_data[-1], y_data[0], marker='o', color=color_spec, 
                            label=label, markersize=8, linestyle='')
    
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='x')
        
        os.makedirs('graphs', exist_ok=True)
        plt.savefig(f'graphs/{filename}', dpi=300, bbox_inches='tight')
        plt.show()
