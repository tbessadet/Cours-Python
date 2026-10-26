# 🐍 Cheat Sheet NumPy

> Basée sur les notebooks *Tutoriel numpy - création de matrice*, *Numpy Matrices* et *Numpy advanced*.
> Convention : `import numpy as np`

## Sommaire
1. [Création de tableaux (arrays)](#1-création-de-tableaux-arrays)
2. [Vecteurs vs matrices, dimensions et reshape](#2-vecteurs-vs-matrices-dimensions-et-reshape)
3. [Indexation et slicing](#3-indexation-et-slicing)
4. [Concaténation et répétition](#4-concaténation-et-répétition)
5. [Opérations arithmétiques : terme à terme vs matriciel](#5-opérations-arithmétiques--terme-à-terme-vs-matriciel)
6. [Fonctions d'agrégation et de statistiques](#6-fonctions-dagrégation-et-de-statistiques)
7. [Recherche, sélection conditionnelle : `np.where`](#7-recherche-sélection-conditionnelle--npwhere)
8. [Tests booléens sur des matrices](#8-tests-booléens-sur-des-matrices)
9. [Broadcasting (dispatch dynamique)](#9-broadcasting-dispatch-dynamique)
10. [Vectorisation de fonctions](#10-vectorisation-de-fonctions)
11. [Types de données](#11-types-de-données)
12. [Sauvegarde / chargement](#12-sauvegarde--chargement)
13. [Boucles utiles avec numpy](#13-boucles-utiles-avec-numpy-rappel-python-pas-numpy-à-proprement-parler)
14. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Création de tableaux (arrays)

**Philosophie générale** : la plupart des fonctions de création prennent **un seul argument**, qui est un **tuple** dès qu'il y a plusieurs dimensions.
👉 `np.ones(10,2)` **ne marche pas** — il faut `np.ones((10,2))`.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.array(liste)` | `liste` (liste ou liste de listes) | Convertit une structure python (liste, liste de listes) en tableau numpy. C'est le pont entre python "pur" et numpy. |
| `np.arange(start, stop, step)` | `start` (déf. 0), `stop`, `step` (déf. 1) | Équivalent numpy de `range`. `stop` est **exclu**. Ex : `np.arange(0,20,2)` → `[0,2,4,...,18]`. |
| `np.linspace(start, stop, n)` | `start`, `stop`, `n` = nombre de points | Génère `n` valeurs régulièrement espacées entre `start` et `stop`. Contrairement à `arange`, **`stop` est inclus**. |
| `np.zeros(shape, dtype)` | `shape` = tuple des dimensions, `dtype` optionnel (`int`, `bool`...) | Tableau rempli de 0. Ex : `np.zeros((3,3,2))` → tableau de zéros de dimensions 3×3×2. |
| `np.ones(shape, dtype)` | idem `zeros` | Tableau rempli de 1. Ex : `np.ones((10,2,3))` → dimensions 10×2×3. |
| `np.eye(n)` | `n` = taille (matrice carrée) | Matrice identité n×n (1 sur la diagonale, 0 ailleurs). `np.eye(10)*5` → diagonale à 5. |
| `np.random.rand(d0, d1, ...)` | dimensions passées **séparément** (pas en tuple !) | Valeurs aléatoires **uniformes entre 0 et 1**. ⚠️ Syntaxe différente de `ones`/`zeros` : `np.random.rand(5,6)` et **pas** `np.random.rand((5,6))`. |
| `np.random.randn(d0, d1, ...)` | dimensions séparées | Valeurs aléatoires selon une **loi normale centrée réduite** (moyenne 0, écart-type 1). |
| `np.random.randint(low, high, size)` | `low` (inclus), `high` (exclus), `size` = shape | Entiers aléatoires. Ex : `np.random.randint(0,3,(n))` → n entiers dans {0,1,2}. |
| `np.loadtxt(fichier, delimiter=...)` | `delimiter` (déf. espace) | Charge une matrice depuis un fichier texte. Le fichier n'est jamais modifié par la suite. |
| `np.savetxt(fichier, mat, fmt=, delimiter=)` | `fmt` (ex `'%.5f'`), `delimiter` | Sauvegarde une matrice dans un fichier texte lisible (échange avec Excel, Matlab...). |
| `pickle.dump(obj, fichier)` / `pickle.load(fichier)` | — | Pas numpy à proprement parler, mais très utilisé pour sauvegarder/charger n'importe quel objet python (dict, matrices, listes...) sans perte de type. |

```python
np.zeros((3,3,2))        # tableau de zéros, dimensions 3x3x2
np.ones(5)                # vecteur de 5 uns
np.eye(4)                 # matrice identité 4x4
np.random.rand(5,6)       # matrice 5x6 de floats uniformes dans [0,1[
np.arange(0,8)            # [0,1,2,3,4,5,6,7]
np.linspace(0,10,15)      # 15 valeurs régulières entre 0 et 10 (bornes incluses)
```

---

## 2. Vecteurs vs matrices, dimensions et reshape

C'est **le piège n°1 de numpy** : un vecteur `np.array([1,2,3])` (shape `(3,)`) n'est **pas** la même chose qu'une matrice ligne `np.array([[1,2,3]])` (shape `(1,3)`).

| Fonction / attribut | Paramètres essentiels | Explication |
|---|---|---|
| `arr.shape` | — | Tuple des dimensions. `n, m = mat.shape` pour récupérer nb lignes / colonnes en une ligne. |
| `arr.reshape(dim1, dim2, ...)` | Une des dimensions peut être `-1` | Change la forme d'un tableau **sans changer les données**. `-1` = "numpy calcule tout seul cette dimension". Ex : `v.reshape(-1,1)` → transforme un vecteur en vecteur colonne. |
| `arr.T` / `arr.transpose()` / `np.transpose(arr)` | — | Transposée. Les trois syntaxes sont équivalentes (objet vs fonction). |

```python
v = np.array([1, 2, 3])      # vecteur, shape (3,)
m_ligne = v.reshape(1, -1)   # matrice ligne, shape (1,3)
m_col   = v.reshape(-1, 1)   # matrice colonne, shape (3,1)
```

⚠️ **Attention aux extractions** : extraire une seule ligne/colonne avec un indice simple renvoie un **vecteur** (dimension perdue), alors qu'une syntaxe "range" la conserve en **matrice** :
```python
A[2, :]     # vecteur, shape (3,)      -> une seule ligne
A[2:3, :]   # matrice, shape (1,3)     -> la dimension "ligne" est conservée
A[:, 1]     # vecteur, shape (n,)      -> une seule colonne
A[:, 1:2]   # matrice, shape (n,1)     -> la dimension "colonne" est conservée
```

---

## 3. Indexation et slicing

Syntaxe générale : `A[ligne, colonne]` (le premier indice = lignes, le second = colonnes).

| Syntaxe | Explication |
|---|---|
| `A[i, j]` | Valeur unique à la ligne `i`, colonne `j`. |
| `A[i, :]` | Toute la ligne `i`. |
| `A[:, j]` | Toute la colonne `j`. |
| `A[start:stop]` | Bloc de valeurs, `stop` exclu. Chaque terme peut être omis (`A[::]`, `A[:3]`, `A[3:]`). |
| `A[start:stop:step]` | Bloc avec un pas. Ex : `A[::2]` → un élément sur deux. |
| `A[-1]` / `A[-3:]` | Indices négatifs = à partir de la fin. `-1` = dernier élément. |
| `A[[i,j,k], :]` | Sélection de lignes précises via une **liste d'indices** (utile pour contours, blocs...). Ex : `mat[[0,-1],:] = 1` met la première et dernière ligne à 1. |
| `A[booléens]` | **Indexation booléenne** : `x[y==-1]` sélectionne les lignes de `x` où la condition sur `y` est vraie. Contrainte : `x` et `y` doivent avoir la même 1ère dimension, `y` doit être un vecteur. |

```python
mat = np.zeros((6,8))
mat[[0,-1], :] = 1     # première et dernière ligne à 1
mat[:, [0,-1]] = 1     # première et dernière colonne à 1 (=contour)
```

---

## 4. Concaténation et répétition

Même philosophie que `zeros`/`ones` : **un seul argument, sous forme de tuple** contenant les tableaux à assembler.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.vstack((A, B, ...))` | tuple de tableaux | Empile **verticalement** (ajoute des lignes). Les tableaux doivent avoir le même nombre de colonnes. |
| `np.hstack((A, B, ...))` | tuple de tableaux | Empile **horizontalement** (ajoute des colonnes). Les tableaux doivent avoir le même nombre de lignes. |
| `np.tile(arr, reps)` | `arr`, `reps` = nombre de répétitions (entier, ou tuple pour répéter selon plusieurs dimensions) | **Répète** un tableau comme un carrelage (d'où le nom "tile"). `reps` entier → répète à la suite (sur la dernière dimension) ; `reps` en tuple → répète indépendamment sur chaque dimension. Pratique pour dupliquer un vecteur en plusieurs lignes/colonnes sans écrire de boucle. |
| `np.repeat(arr, n, axis)` | `n` = nb de répétitions par élément, `axis` optionnel | Différent de `tile` : répète **chaque élément individuellement** `n` fois (au lieu de répéter le tableau entier). Sans `axis`, aplatit d'abord le tableau. |

```python
col1 = np.arange(1,11).reshape(-1,1)
col2 = np.random.rand(10,1)
M = np.hstack((col1, col2))   # assemble deux colonnes côte à côte
M2 = np.vstack((np.array([[1,2]]), M))  # ajoute une ligne en haut

v = np.array([1, 2, 3])
np.tile(v, 2)        # [1 2 3 1 2 3]            (répète la suite)
np.tile(v, (3, 1))   # matrice 3x3, v recopié sur 3 lignes
np.repeat(v, 2)      # [1 1 2 2 3 3]            (répète chaque élément)
```

---

## 5. Opérations arithmétiques : terme à terme vs matriciel

C'est **le piège n°2 de numpy**.

| Opérateur / fonction | Explication |
|---|---|
| `A + s`, `A - s`, `A * s` (scalaire `s`) | Opération appliquée à **chaque élément**. |
| `A * B` | Produit **terme à terme** (element-wise). Nécessite que `A` et `B` aient la **même dimension** (ou soient "broadcastables", voir §9). |
| `A @ B` ou `A.dot(B)` | **Produit matriciel** (au sens algèbre linéaire). Le nombre de colonnes de `A` doit être égal au nombre de lignes de `B`. `@` est la syntaxe la plus claire ; `.dot` est équivalente. |

```python
A * B    # terme à terme (mêmes dimensions requises)
A @ B    # produit matriciel (dims compatibles : (n,d)@(d,m) -> (n,m))
```

⚠️ Mélanger vecteurs et matrices "en forme de vecteur" (shape `(1,n)` ou `(n,1)`) avec `*` ou `@` peut donner des résultats **qui ne plantent pas mais qui sont faux** (broadcasting silencieux). Toujours vérifier les `.shape` en cas de doute.

---

## 6. Fonctions d'agrégation et de statistiques

Toutes ces fonctions existent en **syntaxe objet** (`arr.mean()`) et en **syntaxe fonction** (`np.mean(arr)`) — équivalentes.

Paramètre clé : **`axis`**
- pas d'argument → calcul sur **toute la matrice**
- `axis=0` → calcul **par colonne** (résultat = 1 valeur par colonne)
- `axis=1` → calcul **par ligne** (résultat = 1 valeur par ligne)

| Fonction | Explication |
|---|---|
| `arr.min()` / `arr.max()` | Valeur minimale / maximale. |
| `arr.argmin()` / `arr.argmax()` | **Indice** de la valeur min / max (pas la valeur elle-même). |
| `arr.mean()` | Moyenne. |
| `arr.std()` | Écart-type. |
| `arr.sum()` | Somme. |
| `arr.prod()` | Produit de tous les éléments. |
| `arr.cumsum()` | Somme cumulée. |
| `np.sort(arr, axis)` | Tri (par défaut ligne par ligne, `axis=0` pour trier par colonne). |
| `np.round(arr)` / `np.ceil(arr)` / `np.floor(arr)` | Arrondi au plus proche / au-dessus / en-dessous. |
| `np.minimum(a, b)` / `np.maximum(a, b)` | Minimum/maximum **élément par élément** entre deux tableaux (ou un tableau et un scalaire, pour seuiller). ⚠️ Différent de `min`/`max` classiques de python ! |

⚠️ **La dimension réduite disparaît du résultat** : sur une matrice `(n,m)`, `arr.sum(axis=1)` (ou `.mean`, `.std`, `.min`, `.max`, `.prod`, `.argmin`, `.argmax` avec `axis`) renvoie un **vecteur** de shape `(n,)`, pas une matrice colonne `(n,1)`. C'est le même piège que l'extraction `A[i,:]` vue en §2. Pour conserver la dimension (utile pour ensuite faire du broadcasting proprement), ajouter `keepdims=True` :
```python
m1.sum(axis=1)                  # shape (n,)   -> vecteur, dimension perdue
m1.sum(axis=1, keepdims=True)   # shape (n,1)  -> matrice colonne, dimension conservée
```
👉 Exception : `arr.cumsum(axis=...)` **ne réduit pas** la dimension (résultat cumulatif, même shape que l'entrée) ; c'est seulement `arr.cumsum()` **sans axis** qui aplatit tout le tableau en un vecteur 1D.

```python
m1.mean(axis=0)      # moyenne de chaque colonne
m1.std(1)             # écart-type de chaque ligne
np.minimum(m1, 0.5)   # seuille toutes les valeurs à 0.5 max
```

---

## 7. Recherche, sélection conditionnelle : `np.where`

| Usage | Syntaxe | Explication |
|---|---|---|
| Récupérer les indices | `idx = np.where(condition)` | Renvoie un **tuple** d'indices (lignes, colonnes). Sur un vecteur/une colonne : `idx, = np.where(...)` (virgule pour "déballer" le tuple à 1 élément). |
| Transformer une matrice | `np.where(condition, val_si_vrai, val_si_faux)` | Construit une **nouvelle matrice** selon la condition. Ex : `np.where(a<0, 0, a)` met à 0 tous les négatifs. |
| Compter | `np.where(cond, 1, 0).sum()` ou `np.sum(cond)` | Compte le nombre d'éléments vérifiant une condition. |

⚠️ **Piège des priorités** : avec des conditions combinées, il faut **parenthéser chaque comparaison** :
```python
np.where((a == 4) & (b == 0), 1., 0.)   # OK
# np.where(a == 4 & b == 0, 1., 0.)     # KO ! & est prioritaire sur ==
```

Autres outils de recherche utiles :

| Fonction | Explication |
|---|---|
| `np.unique(arr, return_counts=True)` | Valeurs uniques d'un tableau, avec en option le nombre d'occurrences de chacune (table de contingence). |
| `(arr > x).all()` | `True` si **tous** les éléments vérifient la condition. |
| `(arr > x).any()` | `True` si **au moins un** élément vérifie la condition. |

---

## 8. Tests booléens sur des matrices

En python classique, `if matrice > 0:` **plante** dès que la matrice a plus d'un élément (ambigu : et si certains sont vrais et d'autres faux ?). Il faut agréger le résultat avec `.all()` ou `.any()` avant de faire un `if`.

```python
if (m > 1).all():
    print("tous les éléments sont > 1")
if (m > 1).any():
    print("au moins un élément est > 1")
```

---

## 9. Broadcasting (dispatch dynamique)

Numpy essaie automatiquement d'appliquer une opération même quand les dimensions ne correspondent pas exactement, à condition qu'elles soient **"compatibles"** (une des dimensions vaut 1, ou est absente). C'est pratique... mais source de bugs silencieux si on ne le maîtrise pas.

```python
A = np.array([[0., 1], [2, 3], [4, 5], [6, 7]])   # shape (4,2)
B = A * [1, 2]            # multiplie chaque LIGNE par [1,2] (vecteur ligne, broadcast sur les lignes)
B = A * [[1],[2],[3],[4]] # multiplie chaque COLONNE (vecteur colonne, broadcast sur les colonnes)
```

👉 Règle pratique : pour appliquer une opération par ligne, présenter un vecteur **ligne** (shape `(1,n)` ou liste simple) ; pour une opération par colonne, présenter un vecteur **colonne** (shape `(n,1)`).

---

## 10. Vectorisation de fonctions

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.vectorize(fonction)` | `fonction` = fonction python classique acceptant **un seul scalaire** | Transforme une fonction scalaire en fonction applicable élément par élément sur tout un tableau, sans boucle explicite. Usage marginal mais pratique pour éviter les boucles `for`. |

```python
def theta(x):
    return 1 if x >= 0 else 0

theta_vec = np.vectorize(theta)
theta_vec(np.array([-2,-1,0,1,2]))   # [0 0 1 1 1]
```

---

## 11. Types de données

| Fonction / usage | Explication |
|---|---|
| `np.zeros(shape, int)` | Force le type des éléments (ici entiers). |
| `np.zeros(shape, bool)` | Tableau de booléens (`False` par défaut). |
| `np.ones(shape, bool)` | Tableau de booléens à `True`. |

---

## 12. Sauvegarde / chargement

| Fonction | Cas d'usage |
|---|---|
| `np.savetxt(fichier, arr, fmt=, delimiter=)` / `np.loadtxt(fichier, delimiter=)` | Format texte lisible, idéal pour échanger avec Excel (CSV), Matlab, Java... |
| `pickle.dump(obj, open(fichier,"wb"))` / `pickle.load(open(fichier,"rb"))` | Sérialisation générique python : matrices, dicts, listes, objets quelconques. Format binaire, pas lisible à l'oeil mais très pratique et standard en python. |

---

## 13. Boucles utiles avec numpy (rappel python, pas numpy à proprement parler)

| Fonction | Explication |
|---|---|
| `zip(v0, v1)` | Itère en parallèle sur deux tableaux/listes de même taille. |
| `enumerate(v)` | Itère en donnant à la fois l'indice et la valeur. |

```python
for val0, val1 in zip(v0, v1):
    print(val0, val1)
for i, val in enumerate(v1):
    print(i, val)
```

---

## 📌 Récapitulatif des pièges classiques

1. **`np.ones(10,2)` ne marche pas** → il faut `np.ones((10,2))` (un seul argument = tuple), sauf pour `np.random.rand(10,2)` qui prend les dimensions séparément.
2. **Vecteur ≠ matrice ligne/colonne** : `shape (3,)` ≠ `shape (1,3)` ≠ `shape (3,1)`. Utiliser `reshape(1,-1)` ou `reshape(-1,1)` pour lever l'ambiguïté.
3. **`*` (terme à terme) ≠ `@` (produit matriciel)** — ne jamais confondre.
4. **Conditions combinées dans `np.where`** : toujours parenthéser chaque comparaison avant `&` ou `|`.
5. **`if matrice > 0:` plante** dès que la matrice a plus d'un élément → utiliser `.all()` / `.any()`.
6. **Extraire une ligne/colonne avec un indice simple** (`A[i,:]`) renvoie un **vecteur**, pas une matrice → utiliser `A[i:i+1,:]` si on veut garder une matrice.
7. **Le broadcasting ne plante pas toujours quand il devrait** : en cas de résultat surprenant, toujours vérifier les `.shape` de chaque opérande.
