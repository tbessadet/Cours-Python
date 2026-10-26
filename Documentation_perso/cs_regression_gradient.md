# 📈 Cheat Sheet Régression & Descente de gradient

> Basée sur le notebook *Régression & optimisation par descente de gradient*.
> Convention : `import numpy as np`, `import matplotlib.pyplot as plt`

## Sommaire
1. [Formulation matricielle et enrichissement](#1-formulation-matricielle-et-enrichissement)
2. [Solution analytique des moindres carrés](#2-solution-analytique-des-moindres-carrés)
3. [Évaluation d'un modèle de régression](#3-évaluation-dun-modèle-de-régression)
4. [Modèles non-linéaires par enrichissement de variables](#4-modèles-non-linéaires-par-enrichissement-de-variables)
5. [Descente de gradient](#5-descente-de-gradient)
6. [Visualiser la descente dans l'espace des paramètres](#6-visualiser-la-descente-dans-lespace-des-paramètres)
7. [Choix du pas et critère de convergence](#7-choix-du-pas-et-critère-de-convergence)
8. [Variantes : mini-batch et moment](#8-variantes--mini-batch-et-moment)
9. [Cas réel : régression sur données tabulaires](#9-cas-réel--régression-sur-données-tabulaires)
10. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Formulation matricielle et enrichissement

Pour un modèle linéaire $\hat y_i = a x_i + b$, on préfère toujours une écriture matricielle unifiée $\hat Y = X \cdot w$, en **enrichissant** $X$ d'une colonne de biais (des 1) :

$$X = \begin{pmatrix} x_0 & 1\\ \vdots & \vdots\\ x_n & 1\end{pmatrix}, \qquad w = \begin{pmatrix}a\\b\end{pmatrix}$$

```python
def make_mat_lin_biais(X):   # X : vecteur 1D
    N = len(X)
    return np.hstack((X.reshape(N, 1), np.ones((N, 1))))   # ajoute la colonne de biais

Xe = make_mat_lin_biais(X_train)
```

👉 **Intérêt de cette écriture** : le code de calcul de prédiction (`X @ w`), de coût et de descente de gradient devient **générique**, quel que soit le nombre de variables ou le degré du modèle (voir §4) — on ne change que la construction de `X`.

---

## 2. Solution analytique des moindres carrés

Minimiser $\|Xw - Y\|^2$ revient à annuler le gradient, ce qui donne un système linéaire à résoudre directement (pas besoin d'itérer) :

$$\nabla_w \|Xw-Y\|^2 = 2X^T(Xw - Y) = 0 \;\Longleftrightarrow\; \underbrace{X^TX}_{A}\, w = \underbrace{X^TY}_{B}$$

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.linalg.solve(A, B)` | `A` = matrice carrée, `B` = vecteur | Résout le système linéaire $Aw=B$ — plus stable numériquement que calculer explicitement l'inverse de `A`. |

```python
A = Xe.T @ Xe
B = Xe.T @ y_train
w = np.linalg.solve(A, B)   # solution optimale exacte, en une seule opération
```

---

## 3. Évaluation d'un modèle de régression

| Fonction | Explication |
|---|---|
| **MSE (Mean Squared Error)** | $\frac1N\sum_i (y_i-\hat y_i)^2$ — critère d'erreur le plus courant, c'est aussi la fonction de coût optimisée ci-dessus. |
| Évaluation qualitative | Sur un problème à une seule variable, tracer directement `y` observé vs `yhat` prédit permet de visuellement juger de la qualité de l'ajustement. |

```python
def erreur_mc(y, yhat):
    return ((y - yhat)**2).mean()

yhat_train = Xe @ w
yhat_test  = make_mat_lin_biais(X_test) @ w   # même enrichissement appliqué au test !

print(erreur_mc(y_train, yhat_train), erreur_mc(y_test, yhat_test))

plt.scatter(X_test, y_test, c='r', alpha=0.3)
plt.plot(X_test, yhat_test, 'g--')
```

---

## 4. Modèles non-linéaires par enrichissement de variables

Le même formalisme matriciel permet de construire des modèles **non-linéaires** (ex. polynomiaux) sans rien changer à la logique d'apprentissage : il suffit d'enrichir $X$ avec des puissances de la variable d'origine.

$$X = \begin{pmatrix} x_0^2 & x_0 & 1\\ \vdots & \vdots & \vdots\\ x_n^2 & x_n & 1\end{pmatrix}, \qquad w=\begin{pmatrix}a\\b\\c\end{pmatrix} \;\Rightarrow\; \hat y_i = a x_i^2+bx_i+c$$

```python
def make_mat_poly_biais(X, degre=2):
    return np.column_stack([X**d for d in range(degre, -1, -1)])   # [x^2, x, 1] pour degre=2

Xe   = make_mat_poly_biais(Xp_train, degre=2)
Xe_t = make_mat_poly_biais(Xp_test, degre=2)
w    = np.linalg.solve(Xe.T @ Xe, Xe.T @ yp_train)   # même code de résolution que le cas linéaire !
```

⚠️ **Ne jamais mélanger** deux jeux de données enrichis différemment (ex. `X_train`/`y_train` linéaires vs `Xp_train`/`yp_train` polynomiaux) — l'enrichissement doit être **cohérent** entre l'apprentissage, l'évaluation, et toute nouvelle prédiction.

---

## 5. Descente de gradient

Quand la solution analytique n'est pas disponible ou trop coûteuse (cas des réseaux de neurones), on optimise **itérativement** en suivant l'opposé du gradient de la fonction de coût.

$$w_{t+1} = w_t - \epsilon\, \nabla_w C(w_t), \qquad \nabla_w C = 2X^T(Xw - Y)$$

```python
def descente_grad_mc(X, y, eps=1e-4, nIterations=100):
    w = np.zeros(X.shape[1])   # initialisation (souvent à 0)
    allw = [w]                  # historique des paramètres, pour analyse/affichage
    for i in range(nIterations):
        grad = 2 * X.T @ (X @ w - y)
        w = w - eps * grad
        allw.append(w)
    return w, np.array(allw)

w, allw = descente_grad_mc(Xe, y_train, eps=1e-4, nIterations=200)
```

---

## 6. Visualiser la descente dans l'espace des paramètres

Sur un problème à **2 paramètres** (ex. régression linéaire simple, `a` et `b`), on peut représenter le coût comme une carte de niveaux et suivre la trajectoire de `w` au fil des itérations — excellent outil pédagogique pour comprendre l'algorithme.

```python
def plot_parametres(allw, X, y, opti=[], ngrid=20):
    w_min = np.minimum(opti, allw.min(0)); w_max = np.maximum(opti, allw.max(0))
    w1range = np.linspace(w_min[0], w_max[0], ngrid)
    w2range = np.linspace(w_min[1], w_max[1], ngrid)
    w1, w2 = np.meshgrid(w1range, w2range)
    cost = np.array([[np.log(((X @ np.array([w1i, w2j]) - y)**2).sum()) for w1i in w1range] for w2j in w2range])

    plt.contour(w1, w2, cost)               # carte de niveaux du coût (en log pour l'échelle)
    plt.scatter(opti[0], opti[1], c='r')    # solution optimale (calculée analytiquement, §2)
    plt.plot(allw[:,0], allw[:,1], 'b+-')   # trajectoire de la descente de gradient
```

💡 **Le coût est tracé en `log`** pour compresser les grandes variations d'échelle et mieux visualiser la trajectoire, surtout loin de l'optimum.

---

## 7. Choix du pas et critère de convergence

| Comportement observé | Cause | Diagnostic |
|---|---|---|
| Divergence (coût qui explose) | Pas (`eps`, learning rate) trop grand | Les paramètres "sautent" au-delà de l'optimum à chaque itération, de plus en plus loin. |
| Convergence trop lente | Pas trop petit, ou pas assez d'itérations | Le coût diminue mais très progressivement, il faudrait beaucoup plus d'itérations. |
| Convergence idéale | Pas et nombre d'itérations bien réglés | Le coût diminue rapidement puis se stabilise autour de l'optimum. |

> 🔑 **Le réglage du pas (learning rate) est une étape critique de tous les algorithmes de type réseaux de neurones.**

**Critère d'arrêt automatique (early stopping)** : plutôt que de fixer arbitrairement le nombre d'itérations, on arrête dès que le gradient (ou le coût) cesse d'évoluer significativement.

```python
def descente_grad_mc_cvg(X, y, eps=1e-4, cvg=1e-3, nIterations=200):
    w = np.zeros(X.shape[1])
    allw = [w]
    for i in range(nIterations):
        grad = 2 * X.T @ (X @ w - y)
        w = w - eps * grad
        allw.append(w)
        if np.sqrt((grad**2).sum()) < cvg:   # norme du gradient proche de 0 -> on arrête
            break
    return w, np.array(allw)
```

---

## 8. Variantes : mini-batch et moment

| Variante | Formule | Explication |
|---|---|---|
| Gradient **batch** (classique) | $\nabla_w C$ calculé sur **toutes** les données | Précis mais coûteux sur de grands jeux de données. |
| Gradient **stochastique** | $\nabla_w C$ calculé sur **un seul point** tiré aléatoirement | Rapide par itération, mais trajectoire bruitée. |
| Gradient **mini-batch** | $\nabla_w C$ calculé sur un **sous-ensemble** de points (le `batch_size`) | Compromis standard en pratique (deep learning notamment) : moins bruité que le stochastique, moins coûteux que le batch complet. |
| Moment (inertie) | $v_t = \gamma v_{t-1} + (1-\gamma)\nabla_w C$, puis $w_t = w_{t-1} - \epsilon v_t$ | Lisse la trajectoire en accumulant une "inertie" des gradients précédents — accélère la convergence et réduit les oscillations. `γ=0` revient exactement à la descente stochastique classique (pas de mémoire du passé). |

```python
def descente_grad_mc_mom(X, y, eps=1e-2, gamma=0.9, nIterations=200):
    w = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])          # vitesse/moment, initialisée à 0
    allw = [w]
    for i in range(nIterations):
        i_tirage = np.random.randint(len(y))            # tirage stochastique d'un point
        grad = 2 * X[i_tirage] * (X[i_tirage] @ w - y[i_tirage])
        v = gamma * v + (1 - gamma) * grad
        w = w - eps * v
        allw.append(w)
    return w, np.array(allw)
```

---

## 9. Cas réel : régression sur données tabulaires

Sur un vrai jeu de données (ex. consommation de voitures, dataset *auto-mpg*), plusieurs étapes de prétraitement sont nécessaires avant d'appliquer la même mécanique de résolution analytique.

```python
import pandas as pd

data = pd.read_csv('data/auto-mpg.data', delimiter='\s+', header=None)
data.iloc[:, [3]] = data.iloc[:, [3]].replace('?', None).astype(float)     # valeurs manquantes -> NaN
data.iloc[:, [3]] = data.iloc[:, [3]].fillna(data.iloc[:, [3]].mean())    # imputation par la moyenne

X = np.array(data.values[:, 1:-2], dtype=np.float64)
y = np.array(data.values[:, 0], dtype=np.float64)

def separation_train_test(X, y, pc_train=0.75):
    index = np.random.permutation(len(y))     # mélange aléatoire des indices
    napp = int(len(y) * pc_train)
    return X[index[:napp]], y[index[:napp]], X[index[napp:]], y[index[napp:]]

X_train, y_train, X_test, y_test = separation_train_test(X, y)

Xe = make_mat_lin_biais_multi(X_train)   # généralisation à plusieurs colonnes : ajouter juste une colonne de 1
w = np.linalg.solve(Xe.T @ Xe, Xe.T @ y_train)

plt.bar(np.arange(len(w)), w)   # interprétation des poids : quelles variables pèsent le plus ?
```

💡 **Feature engineering utile ici** : encoder l'origine du véhicule (catégorielle) en one-hot, et discrétiser l'année du modèle en quelques catégories plutôt que de la garder telle quelle — voir la cheat sheet *Scikit-learn — Prétraitement* pour `OneHotEncoder` et `KBinsDiscretizer`.

---

## 📌 Récapitulatif des pièges classiques

1. **Ne jamais mélanger deux jeux de données enrichis différemment** (ex. `Xe` linéaire vs `Xe` polynomial) — toujours vérifier quelle transformation a été appliquée avant un calcul.
2. **Le learning rate mal réglé** est la cause n°1 des soucis de descente de gradient : trop grand → divergence, trop petit → convergence trop lente.
3. **Appliquer le même enrichissement au train et au test** (`make_mat_lin_biais` sur les deux) — un oubli sur l'un des deux jeux fausse silencieusement l'évaluation (dimensions différentes, ou pire, dimensions identiques mais incohérentes).
4. **Les poids d'un modèle linéaire ne sont directement interprétables que si les variables sont sur des échelles comparables** — sur des données réelles brutes (ex. auto-mpg), penser à normaliser avant d'interpréter `w`.
5. **Le critère de convergence (`cvg`) doit être adapté à l'échelle du problème** : un seuil fixe peut être trop strict ou trop laxiste selon les données — toujours vérifier visuellement (tracé du coût) avant de se fier uniquement au critère automatique.
6. **La descente stochastique/mini-batch est bruitée par nature** : ne pas s'inquiéter d'oscillations locales dans la trajectoire, c'est la tendance générale qui compte.
