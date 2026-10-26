# 🧪 Cheat Sheet Scikit-learn — Prétraitement, sélection de variables & pipelines

> Basée sur les notebooks *Sélection de caractéristiques* et *Chaine de traitements*.
> Convention : `import numpy as np`, `import pandas as pd`

## Sommaire
1. [Le fléau de la dimensionnalité](#1-le-fléau-de-la-dimensionnalité)
2. [Sélection de variables](#2-sélection-de-variables)
3. [PCA pour la réduction de dimension](#3-pca-pour-la-réduction-de-dimension)
4. [Régularisation et parcimonie (L1 / L2 / Elastic Net)](#4-régularisation-et-parcimonie-l1--l2--elastic-net)
5. [Encodage des variables discrètes](#5-encodage-des-variables-discrètes)
6. [Création et discrétisation de variables](#6-création-et-discrétisation-de-variables)
7. [Normalisation des données](#7-normalisation-des-données)
8. [Gestion des données manquantes](#8-gestion-des-données-manquantes)
9. [Pipelines et ColumnTransformer](#9-pipelines-et-columntransformer)
10. [Optimisation d'une chaîne complète](#10-optimisation-dune-chaîne-complète)
11. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Le fléau de la dimensionnalité

Ajouter des variables **inutiles** (bruit) dégrade les performances d'un modèle, même si les variables informatives restent présentes : plus il y a de dimensions, plus il est facile de "sur-apprendre" du bruit.

```python
# ajout de dimensions de bruit à des données jouets
ndim_noise = 20
Noise = np.random.randn(len(X), ndim_noise) * clusters_std[0]
Xn = np.concatenate((X, Noise), axis=1)
```

👉 Constat classique : la performance en test se dégrade progressivement quand on ajoute des dimensions de bruit, même si la performance en apprentissage peut rester bonne (signe de sur-apprentissage) — d'où l'intérêt des sections suivantes.

---

## 2. Sélection de variables

| Fonction / stratégie | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| Corrélation brute | — (numpy) | — | Stratégie naïve mais parfois très efficace : `np.abs(X_train.T @ y_train)` donne un score de corrélation par variable avec l'étiquette. |
| `SequentialFeatureSelector(estimator, n_features_to_select=, direction=)` | `sklearn.feature_selection` | `estimator` = modèle servant de critère de sélection, `n_features_to_select`, `direction` = `'forward'` (ajoute progressivement les meilleures variables) ou `'backward'` (retire progressivement les pires) | Sélectionne un sous-ensemble de variables en optimisant itérativement le score d'un modèle. ⚠️ Potentiellement **très coûteux** (ré-entraîne le modèle à chaque étape) — `backward` est en général plus coûteux que `forward` car il part de toutes les variables. |
| `.fit(X, y)` puis `.transform(X)` | — | — | Convention scikit-learn : `fit` calcule quelles dimensions garder, `transform` applique effectivement le filtre (à appliquer identiquement sur train **et** test). |
| `.get_support()` | — | — | Renvoie un masque booléen des variables retenues. |

```python
from sklearn.feature_selection import SequentialFeatureSelector

estimator = svm.SVC(kernel="linear")
selector = SequentialFeatureSelector(estimator, n_features_to_select=2)
selector.fit(X_train, y_train)
Xnew = selector.transform(X_train)   # filtre appliqué
```

---

## 3. PCA pour la réduction de dimension

L'ACP (analyse en composantes principales) cherche les combinaisons de variables qui expliquent le mieux la variance des données — **sans regarder les étiquettes `y`**.

| Fonction / attribut | Paramètres essentiels | Explication |
|---|---|---|
| `PCA(n_components=)` | `n_components` optionnel (sinon garde tous les axes) | Objet de réduction de dimension. |
| `.fit(X)` | — | Calcule les valeurs propres et vecteurs propres (axes principaux) sur `X`. |
| `.transform(X)` | — | Projette `X` sur les axes calculés. |
| `.explained_variance_ratio_` | — | Proportion de variance expliquée par chaque axe (permet de décider combien d'axes conserver). |
| `.singular_values_` | — | Valeurs singulières associées à chaque axe (liées aux valeurs propres). |

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train)
plt.plot(pca.explained_variance_ratio_)   # combien d'axes sont vraiment utiles ?

Xnew = pca.transform(X_train)   # projection sur tous les axes
```

💡 **Sur des données bruitées** : projeter sur 1-2 axes (ceux qui portent l'essentiel de la variance) peut donner des performances quasi optimales et bien meilleures qu'en gardant toutes les dimensions bruitées — la PCA agit alors comme un **débruitage**.

---

## 4. Régularisation et parcimonie (L1 / L2 / Elastic Net)

Idée : intégrer la sélection de variables **directement dans l'apprentissage**, via un terme de pénalité sur les poids `w`.

$$Parcimonie = \frac{\text{nombre de } w_j \ne 0}{d}$$

| Modèle | Import | Formulation | Paramètres essentiels | Explication |
|---|---|---|---|---|
| Ridge (L2) | `from sklearn.linear_model import RidgeClassifier` (ou `Ridge` en régression) | $\mathcal L = \sum_i(\ldots)^2 + C\|w\|^2$ | `alpha` = force de régularisation | Pénalise les poids sans jamais les annuler complètement : réduit le sur-apprentissage mais **ne fait pas de sélection stricte**. |
| Lasso (L1) | `from sklearn.linear_model import lasso_path` | $\mathcal L = \sum_i(\ldots)^2 + C\sum_j\lvert w_j\rvert$ | — (fonction calcule tout le "chemin" de régularisation d'un coup) | Pénalité qui **annule exactement** certains poids → sélection de variables automatique (parcimonie). |
| Elastic Net | `from sklearn.linear_model import ElasticNet` | Combinaison L1 + L2 | `alpha`, `l1_ratio` (proportion de L1 vs L2) | Compromis : profite de la stabilité de L2 tout en gardant un peu de parcimonie via L1. |

```python
from sklearn.linear_model import lasso_path

alphas, coef_, active = lasso_path(X_train, y_train)  # calcule tous les modèles pour différents alpha d'un coup
# coef_[:,i] = poids du modèle pour le i-ème niveau de régularisation
parcimonie = np.where(np.abs(coef_[:,i]) > 1e-5, 1, 0).mean()  # % de poids non nuls
```

⚠️ **Pénalisation extrême** → aucune dimension retenue (modèle trivial, mauvaise performance). **Pénalisation faible** → retour à la situation non régularisée (bonne performance mais risque de sur-apprentissage). Le bon compromis se trouve en balayant les valeurs de régularisation et en traçant performance **et** parcimonie en fonction de celle-ci.

---

## 5. Encodage des variables discrètes

Les modèles de ML (hors arbres) ne savent pas exploiter directement des catégories textuelles/discrètes : il faut les transformer en variables numériques.

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `OneHotEncoder(handle_unknown=, sparse_output=)` | `sklearn.preprocessing` | `handle_unknown='ignore'` (ne plante pas sur une catégorie jamais vue en apprentissage), `sparse_output=False` (renvoie un tableau numpy dense plutôt qu'une matrice creuse) | Transforme chaque catégorie en une colonne binaire (0/1). `.categories_` liste les catégories détectées. |

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(Xbrut)
X = enc.transform(Xbrut)   # dimension augmente : une colonne par catégorie
```

---

## 6. Création et discrétisation de variables

| Fonction / idée | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| Création manuelle de variable | — | — | Construire une nouvelle colonne à partir d'une transformation métier (ex. `(np.floor(x) % 2)*2-1` pour rendre un problème en damier séparable linéairement). Souvent plus efficace qu'un modèle non-linéaire complexe si on connaît le problème. |
| `KBinsDiscretizer(n_bins=, encode=, strategy=)` | `sklearn.preprocessing` | `n_bins` = nombre d'intervalles, `encode` = `'onehot'`/`'ordinal'`, `strategy` = `'uniform'`/`'quantile'`/`'kmeans'` | Transforme une variable continue en catégories (bins) — utile quand une variable a une distribution instable ou multi-modale qui déstabilise les modèles. |

```python
from sklearn.preprocessing import KBinsDiscretizer

enc = KBinsDiscretizer(n_bins=4, encode='onehot', strategy='uniform')
Xtmp = enc.fit_transform(X[:, dim].reshape(-1, 1)).toarray()   # une variable -> plusieurs colonnes binaires
```

---

## 7. Normalisation des données

Les algorithmes de ML sont souvent déstabilisés quand les variables ne sont pas sur des échelles comparables.

| Fonction | Import | Explication |
|---|---|---|
| `StandardScaler()` | `sklearn.preprocessing` | Normalisation gaussienne standard : centre-réduit chaque colonne ($\tilde X_j \sim \mathcal N(0,1)$). Le choix par défaut dans la majorité des cas. |
| `MinMaxScaler()` | `sklearn.preprocessing` | Ramène chaque colonne entre 0 et 1. Souvent utilisé pour des signaux. |
| Normalisation probabiliste (par ligne) | — | Chaque **individu** (pas chaque colonne) somme à 1 — hypothèse multinomiale, typique pour du texte (comparer un texte de 100 mots à un texte de 1000 mots). |

```python
from sklearn.preprocessing import StandardScaler

scal = StandardScaler()
Xn = scal.fit_transform(X_train)   # fit + transform en une fois sur le train
Xn_test = scal.transform(X_test)   # attention : seulement transform sur le test (pas fit !)
```

⚠️ **L'impact de la normalisation dépend du modèle** : très important pour SVM/régression logistique/perceptron (sensibles à l'échelle), beaucoup plus faible pour les arbres/forêts/XGBoost (les seuils sont invariants par changement d'échelle monotone).

---

## 8. Gestion des données manquantes

| Stratégie | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| Suppression des lignes | — | — | Simple, mais perd de l'information — à éviter si beaucoup de lignes sont concernées. |
| Remplacement manuel par la moyenne | — | — | `X[:,j] = np.where(X[:,j]==valeur_manquante, moyenne, X[:,j])`. |
| `SimpleImputer(missing_values=, strategy=)` | `sklearn.impute` | `missing_values=np.nan`, `strategy` = `'mean'`/`'median'`/`'most_frequent'`/`'constant'` | Version scikit-learn, généralisable et intégrable dans une pipeline. ⚠️ Fonctionne sur du **numérique** : il faut d'abord transformer les valeurs manquantes textuelles (ex `'?'`) en `np.nan` (`pd.read_csv(..., na_values='?')`). |

```python
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
Xnum = imp_mean.fit_transform(X)
```

---

## 9. Pipelines et ColumnTransformer

Une **pipeline** enchaîne plusieurs étapes de traitement (prétraitement + modèle) en un seul objet, qui expose lui-même `fit`/`predict`/`score` comme un modèle classique.

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `Pipeline([('nom1', etape1), ('nom2', etape2), ...])` | `sklearn.pipeline` | Liste de tuples `(nom, objet)` | Chaîne les étapes dans l'ordre. Chaque étape (sauf la dernière) doit avoir `fit`/`transform` ; la dernière peut être un modèle (`fit`/`predict`). |
| `make_pipeline(etape1, etape2, ...)` | idem | — | Raccourci de `Pipeline` sans avoir à nommer soi-même les étapes (noms générés automatiquement). |
| `ColumnTransformer([('nom', transfo, colonnes), ...])` | `sklearn.compose` | `colonnes` = liste d'indices ou de noms de colonnes | Applique des transformations **différentes selon les colonnes** (ex. `StandardScaler` sur les colonnes numériques, `OneHotEncoder` sur les colonnes catégorielles), puis concatène le résultat. |

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# pipeline simple : sélection de variables + classifieur
pipe = Pipeline([('sel. var', selector), ('classif', svm.SVC(kernel='linear'))])
pipe.fit(X_train, y_train)
pipe.predict(X_test)

# ColumnTransformer : traitements différents par colonne
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "fare"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["embarked", "pclass"]),
])
log_reg = make_pipeline(preprocessor, LogisticRegression())
log_reg.fit(X, y)
```

---

## 10. Optimisation d'une chaîne complète

`GridSearchCV` fonctionne directement sur une `Pipeline` : il suffit de préfixer chaque paramètre par le **nom de l'étape** suivi de `__`.

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('selvar', selector), ('classif', svm.SVC())])

# V1 : toutes les combinaisons "en vrac" (coûteux, teste des combinaisons absurdes)
grid = {
    "selvar__n_features_to_select": [2, 3, 4, 5],
    "classif__C": [0.1, 1, 5, 10, 100],
    "classif__gamma": [0.1, 0.5, 1, 2, 5],
    "classif__kernel": ["linear", "rbf"],
}

# V2 : sous-dictionnaires pour éviter les combinaisons qui n'ont pas de sens
# (ex. tester gamma pour un noyau linéaire, où gamma n'existe pas)
grid = [
    {"selvar__n_features_to_select": [2, 3, 4, 5]},
    {"classif__kernel": ["linear"], "classif__C": [0.1, 1, 5, 10, 100]},
    {"classif__kernel": ["rbf"], "classif__gamma": [0.1, 0.5, 1, 2, 5]},
]

search = GridSearchCV(pipe, grid, n_jobs=2)
search.fit(X_train, y_train)
print(search.best_params_)
```

---

## 📌 Récapitulatif des pièges classiques

1. **Ne jamais `fit` un scaler/imputer/encoder sur le jeu de test** : seulement `.transform()` sur le test, `.fit_transform()` sur le train — sinon on introduit une fuite d'information (*data leakage*).
2. **L'ajout de variables de bruit dégrade toujours la performance en test**, même si elle reste bonne en apprentissage (signe classique de sur-apprentissage / fléau de la dimensionnalité).
3. **`SequentialFeatureSelector` en mode `backward` est en général plus coûteux** que `forward` : bien vérifier le temps de calcul avant de lancer sur de grandes dimensions.
4. **La PCA est non supervisée** : elle ne regarde jamais `y`, donc les axes de plus grande variance ne sont pas forcément les plus discriminants pour la tâche de classification.
5. **La régularisation L1 (Lasso) fait de la vraie sélection de variables** (poids exactement nuls), contrairement à L2 (Ridge) qui réduit les poids sans les annuler.
6. **L'impact de la normalisation dépend fortement du modèle** : cruciale pour SVM/régression logistique, quasi nulle pour les arbres/XGBoost.
7. **`GridSearchCV` sur une pipeline** utilise la syntaxe `nom_etape__nom_parametre` — un oubli de préfixe fait planter la recherche.
8. **Tester toutes les combinaisons de paramètres "en vrac"** peut inclure des combinaisons absurdes (ex. `gamma` pour un noyau linéaire) — utiliser des sous-dictionnaires pour rester raisonnable en temps de calcul.
