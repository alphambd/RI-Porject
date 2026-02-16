import os
import matplotlib.pyplot as plt
import numpy as np

# Créer le dossier graphs s'il n'existe pas
if not os.path.exists('graphs'):
    os.makedirs('graphs')

def plot_multi_field_optimization(fields_data, model_name='BM25FR'):
    """
    Affiche les courbes d'optimisation pour plusieurs champs sur une grille de 2 colonnes
    """
    n_fields = len(fields_data)
    n_cols = 2
    n_rows = (n_fields + n_cols - 1) // n_cols  # Arrondi supérieur
    
    # Créer une figure avec sous-graphiques en grille 2 colonnes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    # Aplatir axes pour un accès plus facile
    if n_rows > 1 or n_cols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Masquer les sous-graphiques inutilisés
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    for idx, (field_name, data) in enumerate(fields_data.items()):
        ax = axes_flat[idx]
        x_values = data['alphas']
        y_values = data['magps']
        
        # Trier par ordre croissant d'alpha pour une meilleure visualisation
        sorted_idx = np.argsort(x_values)
        x_sorted = np.array(x_values)[sorted_idx]
        y_sorted = np.array(y_values)[sorted_idx]
        
        # Tracer la courbe
        ax.plot(x_sorted, y_sorted, 'b-o', linewidth=2, markersize=8)
        
        # Trouver et marquer l'optimum
        opt_idx = np.argmax(y_values)
        opt_x = x_values[opt_idx]
        opt_y = y_values[opt_idx]
        
        #ax.plot(opt_x, opt_y, 'r*', markersize=20,
        ax.plot(opt_x, opt_y, 'ro', markersize=20, 
                label=f'Optimum: α={opt_x}, MAgP={opt_y:.4f}')
        
        # Ajouter les labels de valeurs pour tous les points
        for x, y in zip(x_values, y_values):
            ax.annotate(f'{y:.4f}', 
                       (x, y), 
                       textcoords="offset points", 
                       xytext=(0, 10 if y != max(y_values) else 15), 
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # Personnalisation
        ax.set_xlabel('Alpha (α)', fontsize=11)
        ax.set_ylabel('MAgP', fontsize=11)
        ax.set_title(f'{model_name} - Champ "{field_name}"', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Ajouter une ligne horizontale pour la baseline si fournie
        if 'baseline' in data:
            ax.axhline(y=data['baseline'], color='gray', linestyle='--', 
                      label=f'Baseline: {data["baseline"]:.4f}')
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    filename = f'graphs/{model_name}_multi_field_optimization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_comparative_optimization(fields_data, model_name='BM25FR'):
    """
    Version alternative: Toutes les courbes sur le même graphique pour comparaison
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for idx, (field_name, data) in enumerate(fields_data.items()):
        x_values = data['alphas']
        y_values = data['magps']
        
        # Trier
        sorted_idx = np.argsort(x_values)
        x_sorted = np.array(x_values)[sorted_idx]
        y_sorted = np.array(y_values)[sorted_idx]
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Tracer la courbe
        plt.plot(x_sorted, y_sorted, 
                color=color, 
                marker=marker, 
                linewidth=2, 
                markersize=7,
                label=f'{field_name}')
        
        # Marquer l'optimum
        opt_idx = np.argmax(y_values)
        opt_x = x_values[opt_idx]
        opt_y = y_values[opt_idx]
        #plt.plot(opt_x, opt_y, color=color, marker='*', markersize=15)
        plt.plot(opt_x, opt_y, color=color, marker='o', markersize=15)
        
        # Ajouter annotation pour l'optimum
        plt.annotate(f'{field_name}: α={opt_x}\nMAgP={opt_y:.4f}', 
                    (opt_x, opt_y),
                    textcoords="offset points",
                    xytext=(10, -20 if idx % 2 == 0 else -35),
                    ha='left',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
    
    plt.xlabel('Alpha (α)', fontsize=12)
    plt.ylabel('MAgP', fontsize=12)
    plt.title(f'{model_name} - Comparaison des optimisations par champ', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Champs', fontsize=10, title_fontsize=11)
    
    # Ajouter baseline si disponible
    baselines = []
    for data in fields_data.values():
        if 'baseline' in data:
            baselines.append(data['baseline'])
    if baselines:
        baseline_val = np.mean(baselines)
        plt.axhline(y=baseline_val, color='black', linestyle=':', 
                   alpha=0.7, label=f'Baseline moyenne: {baseline_val:.4f}')
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    filename = f'graphs/{model_name}_comparative_optimization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_specific_fields(fields_to_plot, model_name='BM25FR'):
    """
    Pour ne visualiser que certains champs spécifiques sur une grille de 2 colonnes
    """
    all_data = {
        'BM25FR': {
            'sec': ([0.5, 0.3, 1.0, 1.5, 1.8], [0.2055, 0.2051, 0.1882, 0.1874, 0.1849]),
            'p': ([0.5, 0.3, 1.0, 1.5, 1.8], [0.1993, 0.2026, 0.1969, 0.1932, 0.1925]),
            'title': ([1.0, 1.5, 2.0, 2.5, 3.0], [0.1969, 0.1969, 0.1969, 0.1968, 0.1969]),
            'bdy': ([0.8, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [0.1997, 0.2075, 0.2087, 0.2102, 0.2115, 0.2123, 0.2125, 0.2132])
        },
        'BM25FW': {
            'p': ([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5], [0.1454, 0.1457, 0.1591, 0.1660, 0.1686, 0.1697, 0.1717, 0.1722, 0.1762]),
            'sec': ([1.5, 2.0, 2.5], [0.1752, 0.1766, 0.1795]),
            'bdy': ([1.0, 1.5, 2.0, 2.5], [0.1824, 0.1829, 0.1874, 0.19]),
            'title': ([1.0, 1.5, 2.0, 2.5], [0.19, 0.1806, 0.1717, 0.1629]),
            'rest': ([1.0, 1.5, 2.0, 2.5], [0.2016, 0.205, 0.2068, 0.1913])
        }
    }
    
    n_fields = len(fields_to_plot)
    n_cols = 2
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    if n_rows > 1 or n_cols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Masquer les sous-graphiques inutilisés
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields_to_plot)))
    
    for idx, field in enumerate(fields_to_plot):
        ax = axes_flat[idx]
        if field in all_data[model_name]:
            x_vals, y_vals = all_data[model_name][field]
            
            # Trier
            sorted_idx = np.argsort(x_vals)
            x_sorted = np.array(x_vals)[sorted_idx]
            y_sorted = np.array(y_vals)[sorted_idx]
            
            ax.plot(x_sorted, y_sorted, 
                    color=colors[idx], 
                    marker='*', 
                    linewidth=2.5,
                    markersize=8,
                    label=f'{field}')
            
            # Marquer l'optimum
            opt_idx = np.argmax(y_vals)
            opt_x = x_vals[opt_idx]
            opt_y = y_vals[opt_idx]
            ax.plot(opt_x, opt_y, 
                    color=colors[idx], 
                    marker='*', 
                    markersize=18)
            
            # Ajouter les labels de valeurs
            for x, y in zip(x_vals, y_vals):
                ax.annotate(f'{y:.4f}', 
                           (x, y), 
                           textcoords="offset points", 
                           xytext=(0, 10 if y != max(y_vals) else 15), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[idx], alpha=0.2))
            
            ax.set_xlabel('Alpha (α)', fontsize=11)
            ax.set_ylabel('MAgP', fontsize=11)
            ax.set_title(f'{model_name} - Champ "{field}"', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    field_str = '_'.join(fields_to_plot)
    filename = f'graphs/{model_name}_fields_{field_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_robertson_optimization():
    """
    Fonction dédiée pour visualiser toutes les optimisations de Robertson
    Basée sur les données de votre rapport
    """
    
    # Données BM25FR
    bm25fr_data = {
        'sec': {
            'alphas': [0.5, 0.3, 1.0, 1.5, 1.8],
            'magps': [0.2055, 0.2051, 0.1882, 0.1874, 0.1849],
            'baseline': 0.2088
        },
        'p': {
            'alphas': [0.5, 0.3, 1.0, 1.5, 1.8],
            'magps': [0.1993, 0.2026, 0.1969, 0.1932, 0.1925],
            'baseline': 0.2088
        },
        'title': {
            'alphas': [1.0, 1.5, 2.0, 2.5, 3.0],
            'magps': [0.1969, 0.1969, 0.1969, 0.1968, 0.1969],
            'baseline': 0.2088
        },
        'bdy': {
            'alphas': [0.8, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'magps': [0.1997, 0.2075, 0.2087, 0.2102, 0.2115, 0.2123, 0.2125, 0.2132],
            'baseline': 0.2088
        }
    }
    
    # Données BM25FW
    bm25fw_data = {
        'p': {
            'alphas': [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5],
            'magps': [0.1454, 0.1457, 0.1591, 0.1660, 0.1686, 0.1697, 0.1717, 0.1722, 0.1762],
            'baseline': 0.0817
        },
        'sec': {
            'alphas': [1.5, 2.0, 2.5],
            'magps': [0.1752, 0.1766, 0.1795],
            'baseline': 0.0817
        },
        'bdy': {
            'alphas': [1.0, 1.5, 2.0, 2.5],
            'magps': [0.1824, 0.1829, 0.1874, 0.19],
            'baseline': 0.0817
        },
        'title': {
            'alphas': [1.0, 1.5, 2.0, 2.5],
            'magps': [0.19, 0.1806, 0.1717, 0.1629],
            'baseline': 0.0817
        },
        'rest': {
            'alphas': [1.0, 1.5, 2.0, 2.5],
            'magps': [0.2016, 0.205, 0.2068, 0.1913],
            'baseline': 0.0817
        }
    }
    
    print("=== BM25FR - Optimisation Robertson ===")
    print("Affichage des courbes par champ...")
    plot_multi_field_optimization(bm25fr_data, 'BM25FR')
    
    print("\n=== BM25FR - Comparaison entre champs ===")
    plot_comparative_optimization(bm25fr_data, 'BM25FR')
    
    print("\n=== BM25FW - Optimisation Robertson ===")
    print("Affichage des courbes par champ...")
    plot_multi_field_optimization(bm25fw_data, 'BM25FW')
    
    print("\n=== BM25FW - Comparaison entre champs ===")
    plot_comparative_optimization(bm25fw_data, 'BM25FW')

def plot_specific_fields(fields_to_plot, model_name='BM25FR'):
    """
    Pour ne visualiser que certains champs spécifiques sur une grille de 2 colonnes
    """
    all_data = {
        'BM25FR': {
            'sec': ([0.5, 0.3, 1.0, 1.5, 1.8], [0.2055, 0.2051, 0.1882, 0.1874, 0.1849]),
            'p': ([0.5, 0.3, 1.0, 1.5, 1.8], [0.1993, 0.2026, 0.1969, 0.1932, 0.1925]),
            'title': ([1.0, 1.5, 2.0, 2.5, 3.0], [0.1969, 0.1969, 0.1969, 0.1968, 0.1969]),
            'bdy': ([0.8, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [0.1997, 0.2075, 0.2087, 0.2102, 0.2115, 0.2123, 0.2125, 0.2132])
        },
        'BM25FW': {
            'p': ([0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5], [0.1454, 0.1457, 0.1591, 0.1660, 0.1686, 0.1697, 0.1717, 0.1722, 0.1762]),
            'sec': ([1.5, 2.0, 2.5], [0.1752, 0.1766, 0.1795]),
            'bdy': ([1.0, 1.5, 2.0, 2.5], [0.1824, 0.1829, 0.1874, 0.19]),
            'title': ([1.0, 1.5, 2.0, 2.5], [0.19, 0.1806, 0.1717, 0.1629]),
            'rest': ([1.0, 1.5, 2.0, 2.5], [0.2016, 0.205, 0.2068, 0.1913])
        }
    }
    
    n_fields = len(fields_to_plot)
    n_cols = 2
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    
    if n_rows > 1 or n_cols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Masquer les sous-graphiques inutilisés
    for i in range(n_fields, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(fields_to_plot)))
    
    for idx, field in enumerate(fields_to_plot):
        ax = axes_flat[idx]
        if field in all_data[model_name]:
            x_vals, y_vals = all_data[model_name][field]
            
            # Trier
            sorted_idx = np.argsort(x_vals)
            x_sorted = np.array(x_vals)[sorted_idx]
            y_sorted = np.array(y_vals)[sorted_idx]
            
            ax.plot(x_sorted, y_sorted, 
                    color=colors[idx], 
                    marker='o', 
                    linewidth=2.5,
                    markersize=8,
                    label=f'{field}')
            
            # Marquer l'optimum
            opt_idx = np.argmax(y_vals)
            opt_x = x_vals[opt_idx]
            opt_y = y_vals[opt_idx]
            ax.plot(opt_x, opt_y, 
                    color=colors[idx], 
                    marker='*', 
                    markersize=18)
            
            # Ajouter les labels de valeurs
            for x, y in zip(x_vals, y_vals):
                ax.annotate(f'{y:.4f}', 
                           (x, y), 
                           textcoords="offset points", 
                           xytext=(0, 10 if y != max(y_vals) else 15), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[idx], alpha=0.2))
            
            ax.set_xlabel('Alpha (α)', fontsize=11)
            ax.set_ylabel('MAgP', fontsize=11)
            ax.set_title(f'{model_name} - Champ "{field}"', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    field_str = '_'.join(fields_to_plot)
    filename = f'graphs/{model_name}_fields_{field_str}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

# Utilisation
if __name__ == "__main__":
    # Pour voir toutes les optimisations Robertson
    #plot_robertson_optimization()
    
    # Pour ne voir que title, bdy, p et sec en BM25FR
    #plot_specific_fields(['title', 'bdy', 'p', 'sec'], 'BM25FR')
    
    # Pour BM25FW
    plot_specific_fields(['p', 'sec', 'bdy', 'title', 'rest'], 'BM25FW')