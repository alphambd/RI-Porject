# Practice 4 - Information Retrieval

## Choix Techniques - Exercice 5

### Méthode de Tokenisation
- **Approche** : Tokenisation standard basée sur les espaces, ponctuation et conversion en minuscules
- **Implémentation** : Utilisation du tokeniseur intégré à la classe `WeightedInvertedIndex`

### Algorithme de Racinisation (Stemming)
- **Algorithme** : Porter Stemmer
- **Justification** : Algorithme standard pour l'anglais, bien adapté à la réduction des variantes morphologiques

### Liste de Mots-Vides
- **Liste utilisée** : stop671 (671 mots-vides anglais)
- **Source** : Fichier `stop-words-english4.txt`
- **Justification** : Liste complète et standard pour le traitement de texte anglais

### Fonctions de Pondération
- **LTN** : TF logarithmique (1 + log(tf)), IDF, pas de normalisation
- **LTC** : TF logarithmique, IDF, normalisation cosinus  
- **BM25** : Modèle probabiliste avec paramètres standards

### Paramètres de Pondération BM25
- **k1 = 1.2** : Paramètre de saturation du TF
- **b = 0.75** : Paramètre de normalisation de la longueur des documents
- **Justification** : Valeurs par défaut recommandées dans la littérature

## Exercice 6 : Optimisation BM25
- **k1 testé** : 0.0 à 4.0 (pas de 0.2) → 21 valeurs
- **b testé** : 0.0 à 1.0 (pas de 0.1) → 11 valeurs  
- **Total** : 32 combinaisons × 7 requêtes = 224 runs
- **Stratégie** : Exploration exhaustive de l'espace 2D des paramètres