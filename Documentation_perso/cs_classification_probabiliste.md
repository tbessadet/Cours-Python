# 🎲 Cheat Sheet Classification probabiliste — Maximum de vraisemblance

> Basée sur le notebook *Apprentissage de paramètres par maximum de vraisemblance* (base USPS).
> Convention : `import numpy as np`, `import matplotlib.pyplot as plt`

## Sommaire
1. [Principe général](#1-principe-général)
2. [Modèle gaussien naïf](#2-modèle-gaussien-naïf)
3. [Classification par argmax de la vraisemblance](#3-classification-par-argmax-de-la-vraisemblance)
4. [Matrice de confusion "à la main"](#4-matrice-de-confusion-à-la-main)
5. [Modèle de Bernoulli (données binaires)](#5-modèle-de-bernoulli-données-binaires)
6. [Modèle géométrique (sur des profils)](#6-modèle-géométrique-sur-des-profils)
7. [Maximum a posteriori (prise en compte des priors)](#7-maximum-a-posteriori-prise-en-compte-des-priors)
8. [Fusion de modèles](#8-fusion-de-modèles)
9. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Principe général

Idée : plutôt que d'utiliser un modèle "boîte noire" scikit-learn, on **construit soi-même** un classifieur probabiliste : pour chaque classe, on estime les paramètres d'une loi de probabilité par **maximum de vraisemblance**, puis on classe une nouvelle donnée dans la classe qui la rend **la plus vraisemblable**.

**Hypothèse d'indépendance conditionnelle** (naïve, mais très pratique) : les variables (ici les pixels d'une image) sont supposées indépendantes **à l'intérieur d'une même classe** :
$$p(X_0,\ldots,X_{255}) = \prod_{i=0}^{255} p(X_i)$$

👉 On travaille systématiquement en **log-vraisemblance** (somme de logs plutôt que produit de probabilités) : numériquement plus stable (évite les produits de nombres minuscules qui sous-débordent), et plus simple à dériver pour l'estimation des paramètres.

---

## 2. Modèle gaussien naïf

On suppose que chaque pixel $X_i$, à l'intérieur d'une classe donnée, suit une loi normale $\mathcal N(\mu_i, \sigma_i^2)$.

**Estimation des paramètres (maximum de vraisemblance)** : la moyenne et la variance empiriques, calculées **séparément pour chaque classe et chaque pixel**.

```python
def learnML_parameters(X, Y):
    """Renvoie mu, sig : matrices (nb_classes, nb_pixels)."""
    classes = np.unique(Y)
    mu  = np.array([X[Y == c].mean(axis=0) for c in classes])
    sig = np.array([X[Y == c].std(axis=0)  for c in classes])
    return mu, sig

mu, sig = learnML_parameters(X_train, Y_train)   # shapes : (10, 256) pour USPS
```

**Log-vraisemblance d'une image** pour un jeu de paramètres (une classe) :
$$\log p(x_0,\ldots,x_{255}) = -\frac{1}{2}\sum_{i=0}^{255}\left[\log(2\pi\sigma_i^2) + \frac{(x_i-\mu_i)^2}{\sigma_i^2}\right]$$

```python
def log_likelihood(img, mu, sig, defsig=1e-5):
    sig_safe = np.maximum(sig, defsig)   # évite la division par 0 (voir piège ci-dessous)
    return -0.5 * np.sum(np.log(2*np.pi*sig_safe**2) + (img - mu)**2 / sig_safe**2)
```

⚠️ **Division par zéro** : si tous les individus d'apprentissage ont exactement la même valeur sur un pixel donné (fréquent sur les pixels de bord, toujours à 0), $\sigma_i^2 = 0$ → division par zéro dans la formule. Deux parades classiques :
- **`np.maximum(sig, defsig)`** : impose un plancher de variance (solution ci-dessus, la plus simple).
- Traiter le pixel séparément avec une vraisemblance forcée à 1 (log-vraisemblance nulle) s'il est constant.

---

## 3. Classification par argmax de la vraisemblance

Une fois les paramètres appris pour chaque classe, la prédiction consiste simplement à choisir la classe qui **maximise** la log-vraisemblance de l'image observée.

```python
def classify_image(img, mu, sig, defeps=1e-5):
    scores = [log_likelihood(img, mu[c], sig[c], defeps) for c in range(len(mu))]
    return np.argmax(scores)

def classify_all_images(X, mu, sig, defeps=1e-5):
    return np.array([classify_image(x, mu, sig, defeps) for x in X])

Y_train_hat = classify_all_images(X_train, mu, sig)
```

⚠️ **Évaluer sur les données d'apprentissage surestime toujours la performance** (le modèle a "vu" ces données pour estimer ses paramètres). Toujours évaluer sur un jeu de **test** séparé, avec les **mêmes paramètres** (`mu`, `sig` appris sur le train, jamais ré-appris sur le test).

```python
# repérer et visualiser une erreur de classification
Y_test_hat = classify_all_images(X_test, mu, sig)
erreurs = np.where(Y_test != Y_test_hat)[0]

i = erreurs[0]
plt.imshow(X_test[i].reshape(16, 16), cmap="gray")
plt.title(f"Prédit: {Y_test_hat[i]}, Réel: {Y_test[i]}")
```

---

## 4. Matrice de confusion "à la main"

Rappel : matrice $C \times C$, lignes = vraie classe, colonnes = classe prédite. À reconstruire soi-même par comptage (avant de connaître `sklearn.metrics.confusion_matrix`, ou pour bien en comprendre le mécanisme).

```python
def matrice_confusion(Y, Y_hat):
    classes = np.unique(Y)
    C = len(classes)
    M = np.zeros((C, C))
    for i in classes:
        for j in classes:
            M[i, j] = np.sum((Y == i) & (Y_hat == j))
    return M

m = matrice_confusion(Y_train, Y_train_hat)
plt.imshow(m)   # une diagonale marquée = bon classifieur
```

---

## 5. Modèle de Bernoulli (données binaires)

Pour des images **binarisées** (pixel allumé/éteint), chaque pixel $X_j$ suit une loi de Bernoulli de paramètre $p_j$ (probabilité d'illumination) :
$$p(X_j = x_{ij}) = p_j^{x_{ij}}(1-p_j)^{1-x_{ij}}$$

**Estimation par maximum de vraisemblance** : $p_j^\star$ = proportion d'images où le pixel $j$ est allumé (moyenne empirique) :

```python
Xb_train = np.where(X_train > 0, 1, 0)   # binarisation préalable

def learnBernoulli(X, Y):
    classes = np.unique(Y)
    return np.array([X[Y == c].mean(axis=0) for c in classes])   # (nb_classes, nb_pixels)

theta = learnBernoulli(Xb_train, Y_train)
```

**Log-vraisemblance** d'une image : $\sum_j x_j\log p_j + (1-x_j)\log(1-p_j)$

```python
def logpobsBernoulli(x, theta, seuil=1e-4):
    p = np.clip(theta, seuil, 1 - seuil)          # seuillage indispensable, voir piège ci-dessous
    return np.sum(x * np.log(p) + (1 - x) * np.log(1 - p), axis=1)   # un score par classe (theta a une ligne/classe)
```

⚠️ **`log(0)` n'est pas défini** : si un pixel est *toujours* allumé ($p_j=1$) ou *toujours* éteint ($p_j=0$) sur l'apprentissage, `log(p_j)` ou `log(1-p_j)` explose. Solution : **seuiller** les probabilités entre `seuil` et `1-seuil` avec `np.clip` avant de prendre le log — jamais laisser une probabilité exactement à 0 ou 1.

---

## 6. Modèle géométrique (sur des profils)

Variante : au lieu de modéliser chaque pixel indépendamment, on résume chaque ligne de l'image par la position de son premier pixel allumé (loi géométrique — "au bout de combien d'essais obtient-on un succès ?"). Cela réduit une image de 256 pixels à seulement 16 valeurs (une par ligne), avec une modélisation adaptée à ce type de variable de comptage.

```python
def learnGeom(X, Y, seuil=1e-4):
    # X : positions du 1er pixel allumé par ligne, valeurs entières dans [0,16]
    classes = np.unique(Y)
    # paramètre p de la loi géométrique = 1 / (position moyenne + 1) par ligne et par classe
    theta = np.array([1 / (X[Y == c].mean(axis=0) + 1) for c in classes])
    return np.clip(theta, seuil, 1 - seuil)

def logpobsGeom(x, theta):
    # log p(x) = (x)*log(1-p) + log(p)  pour chaque ligne, sommé sur les 16 lignes
    return np.sum(x * np.log(1 - theta) + np.log(theta), axis=1)
```

👉 **Idée à retenir** : le choix de la loi de probabilité doit être adapté à la **nature de la variable** modélisée (continue → gaussienne, binaire → Bernoulli, comptage/position → géométrique...) — ce n'est pas un détail technique mais un vrai choix de modélisation.

---

## 7. Maximum a posteriori (prise en compte des priors)

Jusqu'ici, on choisit la classe qui maximise $p(x|y)$ (vraisemblance). En toute rigueur bayésienne, on devrait maximiser $p(y|x) \propto p(x|y)\,p(y)$ — en tenant compte de la fréquence **a priori** de chaque classe $p(y)$.

```python
# estimation des probabilités a priori (fréquence de chaque classe dans le train)
p_prior, _ = np.histogram(Y_train, bins=np.arange(11) - 0.5)
p_prior = p_prior / p_prior.sum()

def classify_MAP(img, mu, sig, p_prior, defeps=1e-5):
    log_vrais = [log_likelihood(img, mu[c], sig[c], defeps) for c in range(len(mu))]
    log_post  = log_vrais + np.log(p_prior)      # somme des logs = produit des probabilités
    return np.argmax(log_post)
```

💡 **Utile surtout si les classes sont déséquilibrées** : sans prior, un modèle peut favoriser à tort une classe rare simplement parce qu'un pixel particulier colle bien à son profil, alors qu'elle est *a priori* peu probable.

---

## 8. Fusion de modèles

Plusieurs modélisations (gaussienne, Bernoulli, géométrique) peuvent être combinées pour améliorer la robustesse globale — l'idée est proche d'un "vote" en ensemble learning (cf. forêts aléatoires) :

| Stratégie | Explication |
|---|---|
| Vote simple | Chaque modèle propose une classe, on garde la classe majoritaire parmi les votes. |
| Vote pondéré | On pondère le vote de chaque modèle par sa performance en apprentissage (un modèle plus fiable "pèse" plus). |
| Fusion de vraisemblances | On **additionne directement** les log-vraisemblances des différents modèles avant l'argmax final (revient à multiplier les probabilités, en supposant les modèles indépendants). |

```python
# fusion par somme des log-vraisemblances (suppose les modèles indépendants)
score_gauss = np.array([[log_likelihood(x, mu[c], sig[c]) for c in range(10)] for x in X_test])
score_bernoulli = np.array([logpobsBernoulli(xb, theta) for xb in Xb_test])

score_total = score_gauss + score_bernoulli
Y_hat = np.argmax(score_total, axis=1)
```

---

## 📌 Récapitulatif des pièges classiques

1. **Travailler en log-vraisemblance**, jamais en probabilité brute — sinon les produits de petites probabilités sous-débordent numériquement vers 0.
2. **`σ² = 0` provoque une division par zéro** dans le modèle gaussien : toujours imposer un plancher (`np.maximum`) sur la variance avant de diviser.
3. **`log(0)` dans le modèle de Bernoulli** : toujours seuiller les probabilités estimées (`np.clip(theta, seuil, 1-seuil)`) avant de prendre le log.
4. **Ne jamais évaluer sur les données d'apprentissage** : la performance y est mécaniquement surestimée (le modèle a été calibré dessus). Toujours réserver un jeu de test distinct.
5. **Le choix de la loi de probabilité doit correspondre à la nature de la variable** (continue, binaire, comptage...) — un mauvais choix de modèle peut être pire que l'hypothèse d'indépendance elle-même.
6. **Sans prior, le modèle ignore le déséquilibre des classes** — le maximum a posteriori corrige ce biais en intégrant la fréquence de chaque classe.
7. **La fusion de vraisemblances suppose les modèles indépendants** — une hypothèse simplificatrice qu'il faut garder en tête en interprétant les résultats.
