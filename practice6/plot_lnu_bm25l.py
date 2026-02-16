import matplotlib.pyplot as plt
import numpy as np
import os

# Créer le dossier graphs
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Données LNU (inchangées)
lnu_slopes = [0.05, 0.1, 0.2, 0.3, 0.4]
lnu_magps = [0.0107, 0.0107, 0.0107, 0.0108, 0.0107]

# Données BM25L avec les vraies valeurs
bm25l_data = [
    {'params': 'k1=1.0,b=0.7,δ=1.2', 'magp': 0.2103},
    {'params': 'k1=1.0,b=0.7,δ=0.2', 'magp': 0.2102},
    {'params': 'k1=1.0,b=0.7,δ=0.8', 'magp': 0.2099},
    {'params': 'k1=1.2,b=0.6,δ=0.5', 'magp': 0.2098},
    {'params': 'k1=1.0,b=0.75,δ=0.5', 'magp': 0.2095},
    {'params': 'k1=0.8,b=0.75,δ=0.5', 'magp': 0.2087},
    {'params': 'k1=1.2,b=0.75,δ=0.5', 'magp': 0.2034},
    {'params': 'k1=1.4,b=0.75,δ=0.5', 'magp': 0.2025},
    {'params': 'k1=1.6,b=0.75,δ=0.5', 'magp': 0.198},
    {'params': 'k1=1.0,b=0.8,δ=0.5', 'magp': 0.1971},
    {'params': 'k1=1.2,b=0.8,δ=0.5', 'magp': 0.1949},
    {'params': 'k1=1.0,b=0.7,δ=0.5', 'magp': 0.1934},
    {'params': 'k1=1.0,b=0.7,δ=1.0', 'magp': 0.193}
]

# Trier par MAgP décroissant pour mieux voir
bm25l_data_sorted = sorted(bm25l_data, key=lambda x: x['magp'], reverse=True)
params = [d['params'] for d in bm25l_data_sorted]
magps = [d['magp'] for d in bm25l_data_sorted]

# Graphique
plt.figure(figsize=(14, 6))

# Sous-graphique LNU
plt.subplot(1, 2, 1)
plt.plot(lnu_slopes, lnu_magps, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Slope (s)', fontsize=11)
plt.ylabel('MAgP', fontsize=11)
plt.title('LNU - Impact du paramètre slope', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 0.015)

for x, y in zip(lnu_slopes, lnu_magps):
    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=9)

# Sous-graphique BM25L
plt.subplot(1, 2, 2)
x_pos = np.arange(len(params))
bars = plt.bar(x_pos, magps, color='skyblue', edgecolor='navy', alpha=0.7)

# Colorer les barres selon la performance
for i, (bar, magp) in enumerate(zip(bars, magps)):
    if magp >= 0.21:
        bar.set_color('green')
    elif magp >= 0.20:
        bar.set_color('yellowgreen')
    else:
        bar.set_color('lightcoral')

# Ligne de baseline BM25
plt.axhline(y=0.2113, color='red', linestyle='--', linewidth=2, 
           label=f'Baseline BM25 (0.2113)')

plt.xticks(x_pos, [f'{i+1}' for i in range(len(params))], rotation=0, fontsize=9)
plt.xlabel('Configuration #', fontsize=11)
plt.ylabel('MAgP', fontsize=11)
plt.title('BM25L - Toutes configurations (triées)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.legend(loc='lower right')
plt.ylim(0.18, 0.22)

# Ajouter les valeurs sur les barres
for i, (x, y, param) in enumerate(zip(x_pos, magps, params)):
    plt.text(x, y + 0.001, f'{y:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    # Ajouter le paramètre en petit
    if i % 2 == 0:  # Un sur deux pour éviter le surcharge
        plt.text(x, 0.181, param, ha='center', va='bottom', fontsize=7, rotation=90, alpha=0.7)

plt.tight_layout()
plt.savefig('graphs/lnu_bm25l_correct.png', dpi=300, bbox_inches='tight')
plt.show()

# Version avec focus sur les variations de δ
plt.figure(figsize=(12, 5))

# Extraire les données pour δ fixe (k1=1.0, b=0.7)
delta_data = [d for d in bm25l_data if d['params'].startswith('k1=1.0,b=0.7')]
delta_data = sorted(delta_data, key=lambda x: float(x['params'].split('δ=')[1]))

delta_params = [d['params'].split('δ=')[1] for d in delta_data]
delta_magps = [d['magp'] for d in delta_data]

plt.subplot(1, 2, 1)
plt.plot(delta_params, delta_magps, 'go-', linewidth=2, markersize=10)
plt.xlabel('δ (delta)', fontsize=11)
plt.ylabel('MAgP', fontsize=11)
plt.title('BM25L - Variation de δ (k1=1.0, b=0.7)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.2113, color='red', linestyle='--', label='Baseline')
plt.legend()

for x, y in zip(delta_params, delta_magps):
    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=9)

# Extraire les données pour différentes combinaisons k1,b (δ=0.5)
kb_data = [d for d in bm25l_data if 'δ=0.5' in d['params'] and d['params'] != 'k1=1.0,b=0.7,δ=0.5']
kb_data = sorted(kb_data, key=lambda x: x['magp'], reverse=True)

kb_labels = [d['params'].replace(',δ=0.5', '') for d in kb_data]
kb_magps = [d['magp'] for d in kb_data]

plt.subplot(1, 2, 2)
x_kb = np.arange(len(kb_labels))
plt.bar(x_kb, kb_magps, color='orange', edgecolor='darkorange', alpha=0.7)
plt.xticks(x_kb, kb_labels, rotation=45, ha='right', fontsize=9)
plt.xlabel('Paramètres (k1, b)', fontsize=11)
plt.ylabel('MAgP', fontsize=11)
plt.title('BM25L - Variation k1/b (δ=0.5)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0.2113, color='red', linestyle='--', label='Baseline')
plt.legend()

for i, (x, y) in enumerate(zip(x_kb, kb_magps)):
    plt.text(x, y + 0.001, f'{y:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('graphs/bm25l_variations.png', dpi=300, bbox_inches='tight')
plt.show()

# Version ultra-minimale avec juste les données
print("\n=== RÉSUMÉ BM25L ===")
print(f"Meilleure config: {bm25l_data_sorted[0]['params']} = {bm25l_data_sorted[0]['magp']:.4f}")
print(f"Pire config: {bm25l_data_sorted[-1]['params']} = {bm25l_data_sorted[-1]['magp']:.4f}")
print(f"Baseline BM25: 0.2113")