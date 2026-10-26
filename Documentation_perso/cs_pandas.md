# 🐼 Cheat Sheet Pandas

> Basée sur le notebook *Tuto pandas*.
> Convention : `import pandas as pd` (souvent combiné à `import numpy as np`)

**Idée générale** : un `DataFrame` pandas ≈ une matrice numpy + des **noms de colonnes** (et souvent un index de lignes) → une philosophie proche des bases de données / tableurs Excel. Chaque ligne = une observation, chaque colonne peut avoir un type différent (texte, date, nombre...).

## Sommaire
1. [Chargement et inspection des données](#1-chargement-et-inspection-des-données)
2. [Sélection de colonnes et de lignes](#2-sélection-de-colonnes-et-de-lignes)
3. [Statistiques de base](#3-statistiques-de-base)
4. [`groupby` : agrégation par catégorie](#4-groupby--agrégation-par-catégorie)
5. [Valeurs manquantes et tri](#5-valeurs-manquantes-et-tri)
6. [Ajout / calcul de nouvelles colonnes](#6-ajout--calcul-de-nouvelles-colonnes)
7. [Discrétisation et tableaux croisés](#7-discrétisation-et-tableaux-croisés)
8. [Fusion de tables (`merge`)](#8-fusion-de-tables-merge)
9. [Affichage direct depuis pandas](#9-affichage-direct-depuis-pandas)
10. [Passerelles numpy ↔ pandas](#10-passerelles-numpy--pandas-à-retenir)
11. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Chargement et inspection des données

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `pd.read_csv(fichier, parse_dates=[...])` | `parse_dates` = liste des colonnes (index ou nom) à interpréter comme des dates | Équivalent pandas de `np.loadtxt`, mais gère nativement les colonnes de types différents (texte, dates, nombres...). |
| `df.head(n)` | `n` = nombre de lignes (déf. 5) | Affiche les **premières** lignes — pour un premier aperçu rapide. |
| `df.tail(n)` | `n` | Affiche les **dernières** lignes. |
| `df.dtypes` | — | Type de chaque colonne (entier, float, texte "object", date...). |
| `type(objet)` | — | À utiliser systématiquement pour savoir si on manipule un `DataFrame`, une `Series`, un `array` numpy... car les fonctions disponibles diffèrent selon le type. |
| `df.describe()` | — | Résumé statistique automatique (moyenne, écart-type, quartiles...) de toutes les colonnes numériques. |

```python
prices_pd = pd.read_csv("data.csv", parse_dates=[-1])   # -1 = dernière colonne
prices_pd.head(15)
prices_pd.dtypes
```

---

## 2. Sélection de colonnes et de lignes

| Syntaxe | Explication |
|---|---|
| `df["colonne"]` | Sélectionne **une colonne** (renvoie une `Series`, pas un DataFrame). |
| `df[["col1","col2"]]` | Sélectionne **plusieurs colonnes** (renvoie un `DataFrame`). Attention : double crochets. |
| `df.loc[lignes, colonnes]` | Sélection par **étiquettes** (noms de colonnes, index ou conditions booléennes). `:` = toutes les lignes/colonnes. |
| `df[df.colonne == valeur]` ou `df.loc[df.colonne == valeur]` | Filtrage par condition, équivalent à l'indexation booléenne de numpy. |
| `serie.values` | Convertit une colonne (`Series`) en tableau **numpy** — c'est la passerelle principale entre pandas et numpy. |

```python
prices_pd["State"]                                   # une colonne (Series)
prices_pd.loc[:, ["MedQ", "HighQ"]].mean()            # moyenne de 2 colonnes
prices_pd.loc[prices_pd.State == "New York", ["MedQ","HighQ"]].mean()  # filtré sur un état
les_etats = np.unique(prices_pd["State"].values)      # retour dans numpy
```

⚠️ **`unique` existe des deux côtés** : `np.unique(serie.values)` (trié) et `pd.unique(serie)` (ordre d'apparition, pas forcément trié) — les deux ne garantissent pas le même ordre.

---

## 3. Statistiques de base

Les mêmes noms que numpy sont presque tous disponibles directement sur une colonne ou un DataFrame : `.mean()`, `.std()`, `.var()`, `.sum()`, `.min()`, `.max()`...

```python
prices_pd["MedQ"].mean()
np.var(prices_pd.loc[prices_pd.State == s, ["HighQ"]].values)  # variance via numpy sur des données extraites
```

👉 Il y a très souvent **deux façons** de résoudre un problème : rester en numpy (`np.where`, opérations matricielles) ou passer par les outils pandas (`groupby`, `.loc`...). Les deux sont valables, mais peuvent donner des résultats **d'apparence différente** si on ne prête pas attention à la manière dont les données sont regroupées (voir `groupby` ci-dessous).

---

## 4. `groupby` : agrégation par catégorie

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `df[["col_cat","col_val"]].groupby(["col_cat"]).mean()` | colonne(s) de regroupement en argument de `groupby` | Calcule une statistique (`mean`, `sum`, `std`...) **pour chaque valeur distincte** de la colonne catégorielle. Équivalent pandas d'une boucle "pour chaque catégorie, filtrer puis agréger", mais en une ligne et optimisé. |

```python
prix_moyens = prices_pd[["State", "LowQ"]].groupby(["State"]).mean()
```

⚠️ **Piège classique** : si la colonne contient des valeurs manquantes (`NaN`), une boucle manuelle et un `groupby` peuvent donner des résultats différents selon la façon dont chacun gère les `NaN` — toujours vérifier d'où vient un résultat "étonnant".

---

## 5. Valeurs manquantes et tri

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `df.sort_values(by=[...], inplace=False)` | `by` = colonne(s) de tri, `inplace` : si `True`, modifie `df` directement (pas de retour) ; si `False`, renvoie une copie triée. | Trie les lignes. Trier avant un remplissage de valeurs manquantes permet un remplissage "cohérent" (ex. propager la dernière valeur connue dans l'ordre chronologique). |
| `df.fillna(...)` | ex. `method='ffill'` (propage la dernière valeur valide) | Remplace les valeurs manquantes (`NaN`). |

```python
prices_sorted = prices_pd.sort_values(by=['State', 'date'], inplace=False)
```

---

## 6. Ajout / calcul de nouvelles colonnes

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `df["nouvelle_col"] = ...` | — | Ajoute (ou remplace) une colonne. |
| `serie.apply(fonction)` | `fonction` = fonction python (souvent une `lambda`) appliquée à chaque élément | Équivalent pandas de `np.vectorize` : applique une fonction élément par élément sur une colonne. |

```python
table_var = {s: np.var(prices_pd.loc[prices_pd.State==s, ["HighQ"]].values) for s in states}
prices_pd["HighQ_var"] = prices_pd["State"].apply(lambda x: table_var[x])
```

---

## 7. Discrétisation et tableaux croisés

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `pd.cut(valeurs, n, labels=)` | `valeurs` (souvent `.values.flatten()` pour repasser en vecteur), `n` = nombre d'intervalles | Découpe des valeurs continues en `n` catégories (bins) — équivalent d'une discrétisation manuelle en numpy, mais automatisé. |
| `pd.value_counts(objet)` | — | Compte les occurrences de chaque catégorie/valeur. |
| `pd.crosstab(col1, col2)` | `col1`, `col2` = deux colonnes catégorielles | Construit une **table de contingence** (comptage croisé sur deux dimensions) — équivalent pandas d'un double comptage numpy. |

```python
prices_pd['cat_LowQ'] = pd.cut(prices_pd["LowQ"].values, 20, labels=np.arange(20))
comptage = pd.crosstab(prices_pd["State"], prices_pd["cat_LowQ"])
plt.imshow(comptage)   # visualisable directement comme une matrice numpy
```

---

## 8. Fusion de tables (`merge`)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `df1.merge(df2, on="colonne")` | `on` = colonne commune servant de clé de jointure | Fusionne deux DataFrames sur une colonne commune (équivalent d'un `JOIN` SQL). Utile par exemple pour comparer deux sous-ensembles de données côte à côte (ex. prix dans deux états différents, alignés par date). |

```python
prix_ny = prices_pd[prices_pd['State'] == 'New York']
prix_ca = prices_pd[prices_pd['State'] == 'California']
prix_ca_ny = prix_ca.merge(prix_ny, on='date')
```

---

## 9. Affichage direct depuis pandas

Pandas propose des raccourcis d'affichage qui s'appuient sur matplotlib en interne — pratiques pour un premier coup d'œil rapide, sans repasser explicitement par `plt`.

| Fonction | Explication |
|---|---|
| `serie.hist()` | Histogramme direct d'une colonne. |
| `df.groupby([...]).mean().plot()` | Enchaîne agrégation + tracé (ex. série temporelle des moyennes par date). |
| `df.plot(kind='bar' / 'kde')` | Barres ou estimation de densité. |
| `df.boxplot()` | Boîte à moustaches directement depuis un DataFrame. |

```python
prices_pd['LowQ'].hist()
prices_pd[['LowQ','date']].groupby(['date']).mean().plot()   # série temporelle
```

---

## 10. Passerelles numpy ↔ pandas (à retenir)

| Sens | Comment faire |
|---|---|
| pandas → numpy | `serie.values` ou `df.values` (renvoie un tableau numpy). |
| numpy → pandas | `pd.DataFrame({"nom_col": array, ...})` (construit un DataFrame en nommant les colonnes). |

```python
data = pd.DataFrame({"x": x, "y": y})   # x, y = tableaux numpy -> DataFrame nommé
les_etats = np.unique(prices_pd["State"].values)  # DataFrame -> numpy
```

---

## 📌 Récapitulatif des pièges classiques

1. **`df["col"]` renvoie une `Series`**, `df[["col"]]` renvoie un `DataFrame` — pas le même type, pas les mêmes méthodes disponibles.
2. **Boucle manuelle vs `groupby`** peuvent donner des résultats différents en présence de `NaN` — toujours vérifier la cohérence des deux approches sur un cas simple.
3. **`inplace=True`** modifie l'objet **directement** (pas de retour à réaffecter) — à utiliser avec précaution, surtout en exploration interactive.
4. **`np.unique` vs `pd.unique`** : le premier trie, le second respecte l'ordre d'apparition — les résultats peuvent sembler incohérents si on ne le sait pas.
5. Toujours faire `type(objet)` en cas de doute : DataFrame, Series et array numpy ne partagent pas toutes les mêmes méthodes.
6. Pour discrétiser (`pd.cut`) une colonne extraite d'un DataFrame multi-index, penser à repasser par `.values.flatten()` pour obtenir un vecteur simple.
