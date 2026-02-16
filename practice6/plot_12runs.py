import matplotlib.pyplot as plt
import numpy as np
import os

# Créer le dossier graphs s'il n'existe pas
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Données extraites du tableau
data = {
    '2009011': {'Itm': 0.02, 'Itc': 0.04, 'bm25': 0.05},
    '2009036': {'Itm': 0.17, 'Itc': 0.14, 'bm25': 0.24},
    '2009067': {'Itm': 0.11, 'Itc': 0.09, 'bm25': 0.13},
    '2009073': {'Itm': 0.10, 'Itc': 0.08, 'bm25': 0.09},
    '2009074': {'Itm': 0.06, 'Itc': 0.05, 'bm25': 0.04},
    '2009078': {'Itm': 0.78, 'Itc': 0.07, 'bm25': 0.77},
    '2009083': {'Itm': 0.18, 'Itc': 0.13, 'bm25': 0.37}
}

def plot_query_comparison_bar(data):
    """
    Graphique en barres groupées pour comparer les modèles par requête
    """
    queries = list(data.keys())
    models = ['Itm', 'Itc', 'bm25']
    
    # Préparer les données
    x = np.arange(len(queries))  # positions des groupes
    width = 0.25  # largeur des barres
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Couleurs pour chaque modèle
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Créer les barres pour chaque modèle
    bars = []
    for i, model in enumerate(models):
        values = [data[q][model] for q in queries]
        bar = ax.bar(x + i * width, values, width, 
                    label=model, color=colors[i], edgecolor='black', linewidth=1)
        bars.append(bar)
        
        # Ajouter les valeurs sur les barres
        for j, v in enumerate(values):
            if v > 0:
                ax.text(x[j] + i * width, v + 0.01, f'{v:.2f}', 
                       ha='center', va='bottom', fontsize=9, rotation=90 if v < 0.1 else 0)
    
    # Personnalisation
    ax.set_xlabel('Requêtes', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Comparaison des modèles Itm, Itc et BM25 par requête', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(queries, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)  # Limite à 1.0 pour les scores
    
    # Ajouter une ligne horizontale à 0.5 pour référence
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = 'graphs/comparaison_modeles_barres.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_query_comparison_line(data):
    """
    Graphique en lignes pour voir l'évolution par requête
    """
    queries = list(data.keys())
    models = ['Itm', 'Itc', 'bm25']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Couleurs et marqueurs
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    for i, model in enumerate(models):
        values = [data[q][model] for q in queries]
        ax.plot(queries, values, marker=markers[i], linewidth=2, 
               markersize=8, label=model, color=colors[i])
        
        # Ajouter les valeurs
        for j, v in enumerate(values):
            ax.annotate(f'{v:.2f}', (queries[j], v), 
                       textcoords="offset points", xytext=(0, 10), 
                       ha='center', fontsize=9)
    
    ax.set_xlabel('Requêtes', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Évolution des scores Itm, Itc et BM25 par requête', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    
    # Rotation des labels des requêtes
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = 'graphs/comparaison_modeles_lignes.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_query_comparison_stacked(data):
    """
    Version avec barres empilées pour voir la contribution
    """
    queries = list(data.keys())
    models = ['Itm', 'Itc', 'bm25']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Préparer les données empilées
    itm_values = [data[q]['Itm'] for q in queries]
    itc_values = [data[q]['Itc'] for q in queries]
    bm25_values = [data[q]['bm25'] for q in queries]
    
    # Créer les barres empilées
    ax.bar(queries, itm_values, label='Itm', color='#FF6B6B', edgecolor='black')
    ax.bar(queries, itc_values, bottom=itm_values, label='Itc', color='#4ECDC4', edgecolor='black')
    ax.bar(queries, bm25_values, bottom=np.array(itm_values) + np.array(itc_values), 
           label='BM25', color='#45B7D1', edgecolor='black')
    
    ax.set_xlabel('Requêtes', fontsize=12)
    ax.set_ylabel('Scores cumulés', fontsize=12)
    ax.set_title('Contribution des modèles par requête (barres empilées)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.2)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Sauvegarder
    filename = 'graphs/comparaison_modeles_empilees.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

def plot_query_heatmap(data):
    """
    Heatmap pour visualiser les performances
    """
    queries = list(data.keys())
    models = ['Itm', 'Itc', 'bm25']
    
    # Créer une matrice de valeurs
    values = np.array([[data[q][m] for m in models] for q in queries])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Afficher les valeurs dans les cellules
    for i in range(len(queries)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{values[i, j]:.2f}',
                          ha="center", va="center", color="black" if values[i, j] < 0.5 else "white")
    
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(queries)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(queries)
    ax.set_xlabel('Modèles', fontsize=12)
    ax.set_ylabel('Requêtes', fontsize=12)
    ax.set_title('Heatmap des performances par modèle et requête', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Score')
    plt.tight_layout()
    
    # Sauvegarder
    filename = 'graphs/comparaison_modeles_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {filename}")
    
    plt.show()

# Exécution
if __name__ == "__main__":
    print("=== Graphique en barres groupées ===")
    plot_query_comparison_bar(data)
    
    print("\n=== Graphique en lignes ===")
    plot_query_comparison_line(data)
    
    print("\n=== Graphique en barres empilées ===")
    plot_query_comparison_stacked(data)
    
    print("\n=== Heatmap ===")
    plot_query_heatmap(data)
    
    # Version simplifiée pour reproduire exactement l'image
    def plot_exact_reproduction(data):
        """
        Reproduction exacte du style de l'image
        """
        queries = list(data.keys())
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Itm
        ax1.bar(queries, [data[q]['Itm'] for q in queries], 
               color='skyblue', edgecolor='black')
        ax1.set_ylabel('Itm', fontsize=11)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Itc
        ax2.bar(queries, [data[q]['Itc'] for q in queries], 
               color='lightcoral', edgecolor='black')
        ax2.set_ylabel('Itc', fontsize=11)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # BM25
        ax3.bar(queries, [data[q]['bm25'] for q in queries], 
               color='lightgreen', edgecolor='black')
        ax3.set_ylabel('BM25', fontsize=11)
        ax3.set_xlabel('Requêtes', fontsize=11)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.suptitle('Comparaison Itm, Itc et BM25 par requête', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = 'graphs/reproduction_exacte.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {filename}")
        
        plt.show()
    
    print("\n=== Reproduction exacte de l'image ===")
    plot_exact_reproduction(data)