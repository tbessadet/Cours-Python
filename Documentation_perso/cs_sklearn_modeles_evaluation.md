# 🤖 Cheat Sheet Scikit-learn — Modèles, Évaluation & Validation croisée

> Basée sur les notebooks *Notebook d'introduction à Scikit-learn* et *Métriques, pièges et chaines de traitement*.
> Convention : `import numpy as np`, `import matplotlib.pyplot as plt`

## Sommaire
1. [Le workflow scikit-learn : fit / predict / predict_proba](#1-le-workflow-scikit-learn--fit--predict--predict_proba)
2. [Données : jeux d'exemple et séparation train/test](#2-données--jeux-dexemple-et-séparation-traintest)
3. [Catalogue de modèles de classification](#3-catalogue-de-modèles-de-classification)
4. [Introspection des modèles](#4-introspection-des-modèles)
5. [Vocabulaire de l'évaluation : matrice de confusion et métriques de base](#5-vocabulaire-de-lévaluation--matrice-de-confusion-et-métriques-de-base)
6. [Métriques d'évaluation](#6-métriques-dévaluation)
7. [Cas déséquilibré : courbe ROC et AUC](#7-cas-déséquilibré--courbe-roc-et-auc)
8. [Validation croisée](#8-validation-croisée)
9. [Sélection de modèle et d'hyperparamètres](#9-sélection-de-modèle-et-dhyperparamètres)
10. [Étendre scikit-learn : créer son propre estimateur](#10-étendre-scikit-learn--créer-son-propre-estimateur)
11. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Le workflow scikit-learn : fit / predict / predict_proba

Tous les modèles scikit-learn partagent la **même interface**, quelle que soit leur famille (bayésien, SVM, arbre, forêt...) : c'est tout l'intérêt de la programmation objet et de l'héritage ici — l'utilisateur manipule des objets interchangeables.

| Méthode | Paramètres essentiels | Explication |
|---|---|---|
| `mod = Modele(**hyperparams)` | dépend du modèle | Création du classifieur (initialisation des hyperparamètres, pas encore d'apprentissage). |
| `mod.fit(X, y)` | `X` = matrice `(n, d)`, `y` = vecteur `(n,)` | Apprentissage sur les données d'entraînement. |
| `mod.predict(X)` | `X` = matrice `(n, d)` | Prédit la **classe** pour chaque ligne de `X`. ⚠️ Attend toujours un **ensemble** de points, jamais un seul individu isolé : pour prédire un seul point, il faut l'envelopper dans une liste (`mod.predict([x])`). |
| `mod.predict_proba(X)` | `X` = matrice `(n, d)` | Renvoie un **score par classe** (matrice `(n, nb_classes)`), même pour des modèles non bayésiens. Utile pour mesurer la confiance, faire du rejet d'ambiguïté, tracer des courbes précision/rappel. |
| `mod.decision_function(X)` | `X` = matrice `(n, d)` | Score de décision continu (dépend du classifieur, contrairement à `predict` qui est générique). Utilisé notamment pour construire une courbe ROC. |
| `mod.score(X, y)` | `X`, `y` | Raccourci qui calcule directement une métrique par défaut (accuracy pour un classifieur). |

```python
from sklearn import naive_bayes

mod = naive_bayes.GaussianNB()
mod.fit(Xapp, Yapp)
yhat = mod.predict([Xtest[0]])          # ATTENTION : liste, même pour un seul point
score = mod.predict_proba([Xtest[0]])   # un score par classe
```

---

## 2. Données : jeux d'exemple et séparation train/test

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sklearn.datasets.load_iris()` | — | Jeu de données jouet classique, renvoyé sous forme de dictionnaire (`.data`, `.target`, `.keys()`). |
| `sklearn.datasets.make_blobs(n_samples=, centers=, cluster_std=, n_features=, random_state=)` | `centers` = liste de coordonnées des centres, `cluster_std` = dispersion (peut être une liste, une valeur par centre) | Génère des données synthétiques en amas gaussiens — pratique pour visualiser un classifieur en 2D. `n_samples` peut être une liste pour générer des classes **déséquilibrées** (ex. `[200, 15]`). |
| `sklearn.model_selection.train_test_split(X, y, test_size=, random_state=)` | `test_size` = proportion ou nombre du jeu de test, `random_state` = graine pour la reproductibilité | Sépare aléatoirement les données en apprentissage/test. Alternative "à la main" avec `np.random.permutation(len(X))`. |

```python
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=100, centers=[[-2,-2],[2,2]], cluster_std=[1.5,1.5], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
```

---

## 3. Catalogue de modèles de classification

| Modèle | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| Bayésien naïf gaussien | `from sklearn.naive_bayes import GaussianNB` | — | Suppose l'indépendance des variables et une loi normale par variable/classe. Rapide, bonne baseline. |
| SVM | `from sklearn.svm import SVC, LinearSVC` | `kernel` (`'linear'`, `'rbf'`...), `gamma` (largeur du noyau gaussien), `C` (régularisation, plus C est grand moins on tolère les erreurs) | Cherche une frontière à marge maximale. `LinearSVC` = version optimisée pour le noyau linéaire. |
| Régression logistique | `from sklearn.linear_model import LogisticRegression` | `C` (inverse de la régularisation) | Modèle linéaire probabiliste, coefficients interprétables (`.coef_`). |
| Arbre de décision | `from sklearn.tree import DecisionTreeClassifier, plot_tree` | `max_depth` | Découpe récursivement l'espace selon des règles de seuils. Modèle le plus **explicable** (règles lisibles). |
| Forêt aléatoire | `from sklearn.ensemble import RandomForestClassifier` | `n_estimators` (nb d'arbres), `max_depth` | Vote d'un ensemble d'arbres, chacun construit sur un sous-échantillon aléatoire de données et de variables. |
| Gradient boosting / XGBoost | `import xgboost as xgb` puis `xgb.XGBClassifier()` (ou `XGBRegressor`) | `n_estimators`, `max_depth` | Forêt construite **itérativement** pour corriger les erreurs des arbres précédents. Souvent l'état de l'art sur données tabulaires. Interface identique à sklearn (`fit`/`predict`). |
| Perceptron | `from sklearn.linear_model import Perceptron` | — | Modèle linéaire simple, historique, multiclasse par défaut. |
| Ridge / Lasso / ElasticNet | `from sklearn.linear_model import RidgeClassifier, lasso_path, ElasticNet` | `alpha` (force de régularisation) | Modèles linéaires régularisés — voir la cheat sheet *Prétraitement & sélection de variables* pour le détail de leur usage en sélection de variables. |

```python
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

mod_svm = svm.SVC(kernel='rbf', gamma=0.1)
mod_svm.fit(X_train, y_train)

mod_rf = RandomForestClassifier(max_depth=10, n_estimators=100)
mod_rf.fit(X_train, y_train)

mod_xgb = xgb.XGBClassifier().fit(X_train, y_train)
```

👉 **Généricité vs spécificité** : tous les modèles partagent `fit`/`predict`/`score`, mais chacun a ses attributs spécifiques après apprentissage (voir §4) — il faut consulter la documentation du modèle utilisé pour les exploiter.

---

## 4. Introspection des modèles

Que valent les paramètres appris ? Chaque famille de modèle expose ses propres attributs (suffixés par `_`, convention scikit-learn pour "calculé après `fit`").

| Modèle | Attribut(s) | Explication |
|---|---|---|
| `GaussianNB` | `.theta_` (moyennes), `.var_` (variances, `.sigma_` dans les anciennes versions) | Une gaussienne par variable et par classe. |
| Modèle linéaire (`LogisticRegression`, `SVC(kernel='linear')`...) | `.coef_` | Poids associés à chaque variable — leur amplitude (en valeur absolue) donne une idée de l'importance de chaque variable, **à condition que les variables soient sur des échelles comparables** (voir cheat sheet Prétraitement). |
| `SVC` | `.support_vectors_` | Les points qui déterminent effectivement la frontière (vecteurs de support). |
| `RandomForestClassifier` | `.n_estimators`, `.max_depth`, `.estimators_[i].get_depth()` | Hyperparamètres effectifs et profondeur réelle de chaque arbre. |
| `XGBClassifier` | `.feature_importances_` | Score d'importance par variable (à interpréter avec prudence — les combinaisons de variables corrélées peuvent fausser l'analyse). |

```python
print(mod1.theta_)              # NB : moyennes par classe/variable
plt.bar(np.arange(len(mod.coef_[0])), mod.coef_[0])  # poids d'un modèle linéaire
```

---

## 5. Vocabulaire de l'évaluation : matrice de confusion et métriques de base

Avant de manipuler les fonctions scikit-learn (§6), il faut maîtriser la définition mathématique de ces métriques — ce sont les briques de base de tout ce qui suit (courbe ROC, précision/rappel, F1...).

**Les 4 quantités de base**, à partir d'une matrice de confusion, pour une classe désignée comme "positive" :

| Quantité | Français | Anglais | Explication |
|---|---|---|---|
| VP | Vrai Positif | True Positive (TP) | Le modèle prédit positif, et c'est effectivement le cas. |
| FP | Faux Positif | False Positive (FP) | Le modèle prédit positif, mais c'est faux (fausse alarme). |
| VN | Vrai Négatif | True Negative (TN) | Le modèle prédit négatif, et c'est effectivement le cas. |
| FN | Faux Négatif | False Negative (FN) | Le modèle prédit négatif, mais c'est faux (détection manquée). |

**Les métriques dérivées** :

| Métrique | Français / Anglais | Équation | Explication |
|---|---|---|---|
| Accuracy | Exactitude / Accuracy | $\dfrac{VP+VN}{VP+VN+FP+FN}$ | Proportion totale de prédictions correctes, toutes classes confondues. ⚠️ Trompeuse sur données déséquilibrées (voir §7). |
| Recall / TPR | Rappel, Taux de Vrais Positifs, Sensibilité / Recall, True Positive Rate (TPR), Sensitivity | $\dfrac{VP}{VP+FN}$ | Parmi tous les positifs réels, quelle proportion est correctement détectée ? Répond à *"je ne rate rien"*. C'est l'axe des ordonnées de la courbe ROC. |
| FPR | Taux de Faux Positifs / False Positive Rate (FPR) | $\dfrac{FP}{FP+VN}$ | Parmi tous les négatifs réels, quelle proportion est classée à tort comme positive (fausse alarme) ? C'est l'axe des abscisses de la courbe ROC. |
| Precision | Précision / Precision | $\dfrac{VP}{VP+FP}$ | Parmi toutes les prédictions positives faites par le modèle, quelle proportion est correcte ? Répond à *"quand j'annonce positif, ai-je raison ?"*. |

```python
CM = met.confusion_matrix(y_test, mod.predict(X_test))
VN, FP, FN, VP = CM.ravel()      # cas binaire uniquement ; ordre à vérifier via mod.classes_

accuracy  = (VP + VN) / (VP + VN + FP + FN)
recall    = VP / (VP + FN)        # rappel / TPR / sensibilité
fpr       = FP / (FP + VN)        # taux de faux positifs
precision = VP / (VP + FP)
```

💡 **Moyen mnémotechnique** : la **précision** répond à "combien de mes alarmes sont vraies ?", le **rappel** répond à "combien de vrais événements ai-je su détecter ?". Les deux évoluent en général en **sens opposé** selon le seuil de décision choisi — d'où l'intérêt de la courbe précision/rappel (§6) pour visualiser ce compromis.

---

## 6. Métriques d'évaluation

⚠️ **La métrique n'est pas nécessairement la fonction optimisée** par le modèle : la métrique sert à présenter des résultats de façon parlante à un expert métier, la fonction de coût sert à l'optimisation interne.

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `accuracy_score(y_true, y_pred)` | `sklearn.metrics` | — | Taux de bonne classification global. ⚠️ Trompeur sur des données déséquilibrées (voir §6). |
| `confusion_matrix(y_true, y_pred)` | idem | — | Matrice `C×C` : lignes = vraie classe, colonnes = classe prédite. Permet d'identifier VP, FP, VN, FN. |
| `ConfusionMatrixDisplay(CM, display_labels=)` puis `.plot()` | idem | `display_labels` = noms de classes (ex `mod.classes_`) | Affichage graphique direct de la matrice de confusion. |
| `precision_recall_curve(y_true, y_pred)` | idem | — | Renvoie précision, rappel pour différents seuils. |
| `precision_score(y_true, y_pred, pos_label=)` / `recall_score(...)` | idem | `pos_label` = classe considérée comme "positive" (déf. `1`) | Précision et rappel pour **une seule classe à la fois**. ⚠️ En multiclasse ou avec des labels `{-1, 1}`, il faut relancer l'appel avec `pos_label=-1` (ou l'autre classe) pour obtenir les scores de l'autre côté — un seul appel ne donne qu'une classe. |
| `PrecisionRecallDisplay(precision=, recall=, name=)` / `.from_estimator(mod, X, y)` | idem | — | Affichage direct de la courbe précision/rappel, avec comparaison possible train/test (`.plot(ax=plt.gca())` pour superposer). |
| `f1_score(y_true, y_pred, pos_label=)` | idem | `pos_label` idem | Moyenne harmonique de la précision et du rappel — pratique pour résumer les deux en un seul chiffre, à calculer **par classe**. |

```python
# précision / rappel pour chacune des deux classes {-1, 1}
print(met.precision_score(y_test, mod.predict(X_test)), met.recall_score(y_test, mod.predict(X_test)))
print(met.precision_score(y_test, mod.predict(X_test), pos_label=-1),
      met.recall_score(y_test, mod.predict(X_test), pos_label=-1))

for c in np.unique(y_train):
    print("f1 classe", c, ":", met.f1_score(y_test, mod.predict(X_test), pos_label=c))
```

```python
import sklearn.metrics as met

acc = met.accuracy_score(y_test, mod.predict(X_test))

CM = met.confusion_matrix(y_test, mod.predict(X_test))
met.ConfusionMatrixDisplay(CM, display_labels=mod.classes_).plot()
plt.show()

met.PrecisionRecallDisplay.from_estimator(mod, X_test, y_test)
```

---

## 7. Cas déséquilibré : courbe ROC et AUC

Sur des données déséquilibrées (classe minoritaire = fraude, alarme, anomalie...), l'accuracy peut être excellente en ne détectant **jamais** la classe rare — il faut donc d'autres outils.

| Concept / fonction | Explication |
|---|---|
| Vrai Positif (VP/TP), Faux Positif (FP) | VP = détection correcte de la classe d'intérêt ; FP = fausse alarme. |
| **Courbe ROC** | Trace le taux de VP en fonction du taux de FP, en faisant varier le seuil de décision (le biais ajouté au score `decision_function`). Plus la courbe passe en haut à gauche, meilleur est le classifieur. |
| `roc_curve(y_true, y_score, pos_label=)` | Import `sklearn.metrics`. Calcule directement les taux de FP/VP (`fpr`, `tpr`) pour tous les seuils, à partir des scores continus — évite de reconstruire la courbe "à la main" en balayant un biais. |
| **AUC (Area Under Curve)** | `sklearn.metrics.auc(fpr, tpr)` — aire sous la courbe ROC. Proche de 1 = excellent, proche de 0.5 = pas mieux que le hasard. |
| `mod.decision_function(X)` | Score continu (avant seuillage), nécessaire pour construire la courbe ROC "à la main" en décalant virtuellement la frontière de décision (utile pour comprendre le principe), ou en entrée de `roc_curve`. |

```python
# Construction "à la main" (pédagogique) : on décale le score par un biais b et on recompte VP/FP
yhat_score = mod.decision_function(X_test)
for b in np.linspace(-5, 5, 50):
    y_tmp = yhat_score + b
    tp = np.where((y_tmp > 0) & (y_test > 0), 1, 0).sum() / (y_test > 0).sum()
    fp = np.where((y_tmp > 0) & (y_test <= 0), 1, 0).sum() / (y_test <= 0).sum()

# Version directe scikit-learn (à privilégier en pratique)
fpr, tpr, thresholds = met.roc_curve(y_test, yhat_score, pos_label=1)
sc = met.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title(f"ROC => AUC = {sc:.2f}")
```

💡 **Interprétation opérationnelle** : si on tolère 0 fausse alerte, quelle couverture (rappel) peut-on espérer ? Si on veut détecter 100% des événements, quel taux de fausse alerte doit-on accepter ? La courbe précision/rappel (ou ROC) répond directement à ce type de question métier.

---

## 8. Validation croisée

Séparer une seule fois en train/test peut donner une estimation de performance peu fiable (dépendante du hasard du découpage). La validation croisée répète l'opération plusieurs fois.

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `KFold(n_splits=)` / `StratifiedKFold(n_splits=)` | `sklearn.model_selection` | `n_splits` = nombre de plis (folds) | Découpe les données en `n_splits` blocs, chacun servant de test à tour de rôle. **`Stratified`** garantit la même répartition des classes dans chaque pli — à privilégier en cas de déséquilibre. |
| `cross_val_score(mod, X, y, cv=, scoring=)` | idem | `cv` = nombre de plis (ou objet KFold), `scoring` = nom de métrique (ex `'accuracy'`, `'f1'`) | Effectue toute la boucle apprentissage/évaluation en une seule fonction, renvoie un score par pli. Facilement parallélisable (`n_jobs=`), mais on ne contrôle pas les données intermédiaires. |

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Solution 1 : boucle explicite (contrôle total)
skf = StratifiedKFold(n_splits=5)
allp = []
for train, test in skf.split(X, y):
    mod = SVC().fit(X[train], y[train])
    allp.append(accuracy_score(y[test], mod.predict(X[test])))

# Solution 2 : fonction toute faite
scores = cross_val_score(SVC(), X, y, cv=5, scoring='accuracy')
```

💡 **Significativité des résultats** : comparer deux modèles n'est pas trivial. Sur peu de données (<50), le problème est statistiquement difficile ; sur beaucoup de données (>10 000), la moindre amélioration est souvent significative ; dans le cas intermédiaire, un test statistique dédié comme `paired_ttest_kfold_cv` (librairie `mlxtend`) permet de trancher.

---

## 9. Sélection de modèle et d'hyperparamètres

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `GridSearchCV(estimator=, param_grid=, cv=)` | `sklearn.model_selection` | `param_grid` = dictionnaire `{nom_paramètre: [valeurs à tester]}` | Teste **toutes les combinaisons** de paramètres par validation croisée et garde le meilleur jeu (`.best_params_`, `.best_score_`, `.cv_results_`). |
| `optuna.create_study(direction=)` puis `.optimize(objective, n_trials=)` | `import optuna` | `direction` = `'maximize'`/`'minimize'` | Recherche **plus fine** qu'une grille fixe : au lieu de tester des valeurs discrètes, on définit un espace de recherche (`trial.suggest_int`, `trial.suggest_float(..., log=True)`, `trial.suggest_categorical`) et Optuna explore intelligemment cet espace (utile dès que la grille devient trop grande pour être testée exhaustivement). |

```python
from sklearn.model_selection import GridSearchCV

parameters = {'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]}
meta_mod = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=3)
meta_mod.fit(X_train, y_train)
print(meta_mod.best_params_, meta_mod.best_score_)
```

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 2, 20)
    max_depth = int(trial.suggest_float("max_depth", 1, 32, log=True))
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_val_score(clf, X, y, cv=3).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial.params)
```

---

## 10. Étendre scikit-learn : créer son propre estimateur

Grâce à l'héritage, on peut fabriquer un classifieur maison qui reste compatible avec **tous** les outils scikit-learn (validation croisée, grid search, pipelines...).

| Élément | Explication |
|---|---|
| `class MonClassifieur(BaseEstimator, ClassifierMixin):` | Hériter de `BaseEstimator` (obligatoire) et `ClassifierMixin` (ajoute `.score()` par défaut). |
| `__init__(self, ...)` | Doit **stocker tous les hyperparamètres en attributs** (nécessaire pour la sérialisation / le clonage interne de sklearn). |
| `fit(self, X, y)` | Doit renvoyer `self`. Stocker `self.classes_ = np.unique(y)` par convention. |
| `predict(self, X)` | Doit renvoyer les prédictions pour un ensemble `X`. |

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearFixClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, data_dim=2):
        self.data_dim = data_dim
        self.w = np.random.randn(data_dim)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        return np.where(X @ self.w > 0, self.y_[0], self.y_[1])

mod = LinearFixClassifier(2)
scores = cross_val_score(mod, X, y, cv=5)   # fonctionne directement !
```

---

## 📌 Récapitulatif des pièges classiques

1. **`predict` attend toujours un ensemble de points** (matrice 2D), jamais un individu isolé directement — utiliser `mod.predict([x])` pour un seul point.
2. **`predict_proba` renvoie un score par classe** : la dimension du résultat est différente de `predict` (une colonne par classe, pas juste la classe prédite).
3. **L'accuracy est trompeuse sur des données déséquilibrées** — une classe très majoritaire peut donner une accuracy élevée même en ignorant totalement la classe rare. Utiliser précision/rappel/F1/ROC-AUC selon le contexte.
4. **`StratifiedKFold`** plutôt que `KFold` en cas de classes déséquilibrées, pour garantir la même répartition dans chaque pli.
5. **Les coefficients d'un modèle linéaire ne sont interprétables directement que si les variables sont sur des échelles comparables** (voir cheat sheet Prétraitement pour la normalisation).
6. **Comparer deux modèles nécessite de la prudence statistique** : peu de données → test statistique dédié ; beaucoup de données → toute différence tend à devenir significative.
7. **`GridSearchCV`** teste **toutes** les combinaisons (coût combinatoire) — au-delà de quelques paramètres/valeurs, préférer une recherche plus fine (Optuna) qui explore intelligemment l'espace.
