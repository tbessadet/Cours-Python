# 🔎 Cheat Sheet Réduction de dimension & Visualisation (PCA, t-SNE)

> Basée sur le notebook *Visualisation de données en haute dimension* (correction).
> Convention : `import numpy as np`, `import matplotlib.pyplot as plt`

## Sommaire
1. [Pourquoi réduire la dimension pour visualiser ?](#1-pourquoi-réduire-la-dimension-pour-visualiser-)
2. [ACP "à la main" avec les valeurs propres](#2-acp-à-la-main-avec-les-valeurs-propres)
3. [Projection et interprétation des axes](#3-projection-et-interprétation-des-axes)
4. [PCA avec scikit-learn](#4-pca-avec-scikit-learn)
5. [t-SNE : visualisation non-linéaire](#5-t-sne--visualisation-non-linéaire)
6. [Analyse des erreurs d'un modèle via une projection 2D](#6-analyse-des-erreurs-dun-modèle-via-une-projection-2d)
7. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Pourquoi réduire la dimension pour visualiser ?

L'œil humain ne sait lire que des graphiques en 2D (ou 3D). Dès qu'un jeu de données a plus de 2-3 variables (ex : images de chiffres manuscrits USPS, 256 pixels = 256 dimensions), il faut **projeter** les données sur un espace de plus petite dimension avant de pouvoir les visualiser — quitte à perdre de l'information dans l'opération.

👉 Deux grandes familles d'outils : les méthodes **linéaires** (PCA — rapides, axes interprétables) et les méthodes **non-linéaires** (t-SNE, UMAP — capturent mieux des structures complexes, mais plus coûteuses et moins interprétables).

---

## 2. ACP "à la main" avec les valeurs propres

L'analyse en composantes principales (ACP/PCA) cherche les directions de l'espace qui **maximisent la variance** des données projetées. Ces directions sont exactement les **vecteurs propres** de la matrice de corrélation $X^TX$, associés aux plus grandes **valeurs propres**.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.linalg.eig(M)` | `M` = matrice carrée (ici `X.T @ X`) | Renvoie `(valeurs_propres, vecteurs_propres)`. Chaque colonne `V[:,i]` est le vecteur propre associé à `lam[i]`. ⚠️ Les valeurs propres ne sont **pas triées** par défaut. |
| `np.argsort(-lam)` | `lam` = vecteur de valeurs propres | Indices qui trient les valeurs propres par ordre **décroissant** (le signe `-` inverse l'ordre naturel croissant de `argsort`). |

```python
lam, V = np.linalg.eig(X.T @ X)   # travail sur la matrice de corrélation pour extraire la variance
print(lam, "\n", V)

# affichage des axes propres sur les données originales
plt.scatter(X[:,0], X[:,1], c=Y)
sc = 3   # facteur d'échelle purement visuel
for i in range(len(V)):
    plt.plot([0, V[0,i]*sc], [0, V[1,i]*sc], 'r')
    plt.text(V[0,i]*sc, V[1,i]*sc, "{:.2f}".format(lam[i]))  # annote avec la valeur propre associée
```

---

## 3. Projection et interprétation des axes

Une fois les axes principaux calculés, on **projette** les données dessus par un simple produit matriciel.

```python
# axe principal = vecteur propre associé à la plus grande valeur propre
ind = np.argmax(lam)
vp  = V[:, ind]

xp = X @ vp    # projection 1D sur l'axe principal (produit scalaire de chaque ligne avec vp)

plt.plot(xp[Y==1], [1]*np.sum(Y==1), '+')   # affichage en 1D, séparé artificiellement en hauteur pour la lisibilité
plt.plot(xp[Y==0], [-1]*np.sum(Y==0), '+')
```

Pour projeter sur **plusieurs** axes à la fois (ex. les `d` premiers) :
```python
d = 4
ind = np.argsort(-lam)
vp  = V[:, ind[:d]]     # les d vecteurs propres associés aux d plus grandes valeurs propres

xp = X @ vp             # projection sur d axes, shape (n, d)

for c in np.unique(Y):
    plt.scatter(xp[Y==c, 0], xp[Y==c, 1], alpha=0.5)   # affichage des 2 premiers axes, coloré par classe
plt.legend(np.unique(Y))
```

💡 **Attention à l'ordre d'affichage des classes** : sur un nuage de points superposés (`alpha` < 1), la dernière classe tracée peut visuellement "écraser" les précédentes. Inverser l'ordre de la boucle (`for c in np.unique(Y)[::-1]`) permet de vérifier qu'aucune classe n'est artificiellement cachée derrière une autre — un bon réflexe avant de conclure qu'une classe est "peu présente" quelque part.

---

## 4. PCA avec scikit-learn

Version prête à l'emploi, équivalente au calcul manuel ci-dessus mais gérant automatiquement le centrage des données.

| Fonction / attribut | Paramètres essentiels | Explication |
|---|---|---|
| `PCA(n_components=)` | `n_components` = nombre d'axes à garder | Objet de réduction de dimension. |
| `.fit_transform(X)` | — | Calcule les axes **et** projette en une seule étape. |
| `.components_` | — | Les vecteurs propres (axes), équivalent de `V` ci-dessus. |
| `.explained_variance_ratio_` | — | Proportion de variance expliquée par chaque axe conservé. |

```python
from sklearn.decomposition import PCA

pca = decomposition.PCA(n_components=2)
pca_X = pca.fit_transform(X)   # projection directe en 2D, prête à afficher

plt.scatter(pca_X[:,0], pca_X[:,1], c=colors, cmap='plasma')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
```

👉 Voir aussi la cheat sheet *Scikit-learn — Prétraitement* pour l'usage de la PCA en amont d'un modèle (sélection/réduction de variables), ici l'angle est purement **visualisation**.

---

## 5. t-SNE : visualisation non-linéaire

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `TSNE()` | `sklearn.manifold` | paramètres par défaut suffisent souvent pour un premier essai | Projette en 2D (ou 3D) en cherchant à préserver les **distances locales** entre points (contrairement à la PCA qui préserve la variance globale). Très utilisé pour visualiser des espaces d'embeddings (ex. sortie d'un réseau de neurones). |
| `.fit_transform(X)` | — | Calcule directement la projection 2D. |

```python
from sklearn.manifold import TSNE

visu = TSNE()             # paramètres par défaut
x2d = visu.fit_transform(X)

for y in np.unique(Y):
    plt.scatter(x2d[Y==y, 0], x2d[Y==y, 1], alpha=0.5)
plt.legend(np.unique(Y))
```

⚠️ **t-SNE n'est pas une PCA** : les axes n'ont **aucune signification** (ni direction, ni échelle interprétable), le résultat n'est **pas reproductible** d'un lancement à l'autre (initialisation aléatoire), et l'algorithme est **coûteux** sur de grands jeux de données (à limiter à quelques milliers de points, ou sous-échantillonner).

---

## 6. Analyse des erreurs d'un modèle via une projection 2D

Une projection 2D (PCA ou t-SNE) est aussi un excellent outil de **diagnostic** : en superposant les erreurs de classification sur la carte, on peut voir si elles sont concentrées dans des zones "ambiguës" (frontières entre classes) ou dispersées partout (signe d'un problème plus profond).

```python
yhat = mod.predict(X_test)
x2d = TSNE().fit_transform(X_test)

for y in np.unique(y_test):
    plt.scatter(x2d[y_test==y, 0], x2d[y_test==y, 1], alpha=0.5)

erreurs = y_test != yhat
plt.plot(x2d[erreurs, 0], x2d[erreurs, 1], "sk")   # marque les erreurs en carrés noirs, par-dessus
```

---

## 📌 Récapitulatif des pièges classiques

1. **`np.linalg.eig` ne trie pas les valeurs propres** : toujours passer par `np.argsort(-lam)` pour retrouver les axes principaux dans le bon ordre.
2. **La PCA est non supervisée** : elle maximise la variance globale, pas la séparabilité entre classes — un axe de forte variance n'est pas forcément discriminant.
3. **t-SNE n'est pas reproductible** (aléa d'initialisation) et ses axes **ne s'interprètent pas** (contrairement aux vecteurs propres de la PCA) — à réserver à l'exploration visuelle, jamais à une analyse quantitative des axes.
4. **Superposition de classes (`alpha`)** : l'ordre de tracé peut cacher visuellement une classe minoritaire — vérifier en inversant l'ordre d'affichage.
5. **t-SNE est coûteux** : sur de grands jeux de données, sous-échantillonner avant de lancer la projection.
6. **Facteur d'échelle purement visuel** (le `sc` dans les tracés d'axes) : ne pas confondre avec la valeur réelle des vecteurs propres, qui sont par convention de norme 1.
