Auteur:
Diallo Alpha
Boulay Clement
Sanou Ana

---

# README – Exécution des Practices 1 à 6

#### Lien GitHub du projet : 
https://github.com/alphambd/RI-Porject 

## Pré-requis

* Python 3.8+
* Bibliothèques Python : `os`, `shutil`, `time`, `pickle`, `hashlib`, `re`, `xml.etree.ElementTree`, `collections`, `gzip`, `math`

---

## Exécution par practice

### Practice 1 – Statistiques de la collection 

Commande pour l'execution
```bash
python practice1/main.py
```
Remarque : Si l’exécution directe du script (ex. practice1/main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice1 avant de lancer le script.

Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans
le dossier de la practice suivante.

* Fichier utilisé : `collection.txt`
* Compte les tokens bruts et calcule les statistiques globales (longueur moyenne, vocabulaire, etc.)

---

### Practice 2 
Commande pour l'execution
```bash
python practice2/main.py
```
Remarque : Si l’exécution directe du script (ex. practice2/main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice2 avant de lancer le script.


Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans
le dossier de la practice suivante.

* Dossier des données : `data/Practice_02_data/`
* Applique le stemming et supprime les stop-words avant de calculer les statistiques.

---

### Practice 3 
Commande pour l'execution
```bash
python practice3/practice3_main.py
```
Remarque : Si l’exécution directe du script (ex. practice3/practice3_main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice3 avant de lancer le script.

Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans 
le dossier de la practice suivante.


* Dossier des données : `data/Practice_03_data/`
* Calcule le score des documents selon le modèle ltn et affiche le Top-10 des documents pertinents.

---

### Practice 4 
Commande pour l'execution
```bash
python practice4/practice4_main.py
```
Remarque : Si l’exécution directe du script (ex. practice4/practice4_main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice4 avant de lancer le script.

Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans 
le dossier de la practice suivante.

* Fichier utilisé : `data/Text_Only_Ascii_Coll_NoSem`
* Calcule le score avec normalisation cosinus (ltc).
* Les normes cosine sont pré-calculées et mises en cache pour accélérer l’exécution.
* * Calcule les scores pour chaque document et affiche le classement Top-10.


---

### Practice 5 
Commande pour l'execution
```bash
python practice5/practice5_main.py
```
Remarque : Si l’exécution directe du script (ex. practice5/practice5_main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice5 avant de lancer le script.

Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans le 
dossier de la practice suivante.

* Dossier des données : `data/Practice_05_data/`
* Calcule les scores pour chaque document et affiche le classement Top-10.

---

### Practice 6 
Commande pour l'execution
```bash
python practice6/practice6_main.py
```
Remarque : Si l’exécution directe du script (ex. practice6/practice6_main.py) renvoie une erreur de fichier introuvable ou de permission, 
placez-vous d’abord dans le dossier de la practice correspondante avec cd practice6 avant de lancer le script.

Remarque : Une fois l’exécution d’une practice terminée, revenez au dossier parent avec cd .. avant de vous placer dans le
dossier de la practice suivante.

* Dossier des données : `data/Practice_05_data/`
* Traite chaque article comme un document indépendant.
* Pour HTML, pondère les champs titre, corps, paragraphes selon Wilkinson ou Robertson.
* Affiche les résultats classés par pertinence pour chaque article.

---

## Notes

* Pour activer le stemming ou les stop-words, ajuster les paramètres `use_stemmer=True` et `use_stop_words=True` dans le script.
* Les caches de normes cosine sont stockés dans `data/norm_cache/`.

---

