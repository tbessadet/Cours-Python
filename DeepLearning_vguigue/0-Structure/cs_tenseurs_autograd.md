# 🔥 Cheat Sheet PyTorch — Tenseurs & Autograd

> Basée sur les notebooks `0_0-tensors-from-pytorch`, `1_1-pytorch-struct`, `1_2-pytorch-grad`.
> Convention : `import torch`, `import numpy as np`, `import matplotlib.pyplot as plt`

## Sommaire

**I. Tenseurs**
1. [Créer un tenseur](#1-créer-un-tenseur)
2. [Types (dtype) et conversions](#2-types-dtype-et-conversions)
3. [Indexation et masques](#3-indexation-et-masques)
4. [Opérations, réductions, algèbre linéaire](#4-opérations-réductions-algèbre-linéaire)
5. [Broadcasting](#5-broadcasting)
6. [Changer la forme](#6-changer-la-forme)
7. [Inplace, copies, mémoire partagée](#7-inplace-copies-mémoire-partagée)
8. [Pont NumPy / Python / matplotlib](#8-pont-numpy--python--matplotlib)

**II. Autograd & descente de gradient**

9. [Autograd : requires_grad, backward, grad](#9-autograd--requires_grad-backward-grad)
10. [Mettre à jour les paramètres à la main](#10-mettre-à-jour-les-paramètres-à-la-main)
11. [Optimiseurs `torch.optim`](#11-optimiseurs-torchoptim)
12. [Régression linéaire complète + baselines sklearn](#12-régression-linéaire-complète--baselines-sklearn)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Tenseurs

## 1. Créer un tenseur

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torch.tensor(data, dtype=None, requires_grad=False)` | `data` : liste/tuple (imbriqués) ou array ; `dtype` ; `requires_grad` | Crée un tenseur en **copiant** les données. Le type est déduit (`float32` si réels Python, `int64` si entiers). Le réflexe pour des données « à la main ». |
| `torch.zeros(*size, dtype=)` / `torch.ones(...)` | `size` : `(2,3)` ou `2, 3` | Tenseur rempli de 0 / 1. Défaut `float32`. |
| `torch.empty(*size)` | `size` | Alloue **sans initialiser** : valeurs = ce qui traînait en mémoire. Rapide mais à remplir ensuite. |
| `torch.rand(*size)` | `size`, `generator` | Uniforme sur $[0,1[$. |
| `torch.randn(*size)` | `size` | Loi normale $\mathcal N(0,1)$. Réflexe pour initialiser des poids. |
| `torch.randint(low, high, size)` | `high` exclu | Entiers uniformes. |
| `torch.linspace(start, end, steps)` | `steps` = nombre de points (bornes incluses) | Grille régulière, ex. pour tracer une courbe de régression. |
| `torch.arange(start, end, step)` | `end` exclu | Équivalent `range` / `np.arange`. |
| `torch.zeros_like(x)` / `ones_like` / `rand_like` / `empty_like` | `x` modèle ; `dtype=` pour changer | Même **forme** (et même dtype/device) que `x`. |
| `t.random_(low, high)` / `t.normal_(mean, std)` | inplace | Remplit un tenseur existant : `torch.zeros(2,3).random_(10,15)` (entiers dans [10, 15[), `.normal_(1, 0.1)`. |
| `torch.cat(tensors, dim=0)` | tuple de tenseurs, `dim` | **Concatène** le long d'une dimension existante (les autres dims doivent coïncider). `dim=0` empile des lignes, `dim=1` des colonnes. |
| `torch.stack(tensors, dim=0)` | idem | 🔎 Empile en **créant** une nouvelle dimension (tous de même forme). |
| `torch.manual_seed(n)` | entier | Fixe le générateur : mêmes tirages après chaque appel → **reproductibilité**. |

```python
torch.manual_seed(1729)
X = torch.cat((torch.tensor([[1., 2]]), torch.tensor([[2., 3]])), dim=0)  # (2, 2)
x = torch.linspace(0, 5, 100)          # 100 points pour tracer
```

> 👉 `size` peut être donné en suite d'entiers (`torch.ones(2, 3)`) ou en tuple (`torch.ones((2, 3), dtype=torch.int16)`) — le tuple est plus lisible quand on ajoute des options.

---

## 2. Types (dtype) et conversions

| Fonction / attribut | Paramètres essentiels | Explication |
|---|---|---|
| `dtype=` à la création | `torch.float` (= `float32`, défaut), `torch.double` (`float64`), `torch.half` (`float16`), `torch.bfloat16`, `torch.int`/`int32`, `torch.long` (`int64`), `torch.int16`, `torch.int8`, `torch.uint8`, `torch.bool` | Fixe le type élémentaire. **Réflexe : `dtype=torch.float` sur les données réelles** (sklearn/numpy produisent du float64). |
| `t.float()`, `t.double()`, `t.long()`, `t.int()` | — | Conversions rapides. `long` obligatoire pour des **indices / labels de classes**. |
| `t.to(dtype)` / `t.to(device)` | `torch.int32`, `"cuda"`, `"mps"` | Conversion générique (type **ou** support). `float → int` **tronque**. |
| `t.type()` | — | Nom du type (`'torch.IntTensor'`). |
| `t.dtype`, `t.shape`, `t.size(d)` | `d` optionnel | Inspection. `t.size(1)` = taille de la dim 1 (= `t.shape[1]`). |
| `torch.cuda.is_available()` / `torch.backends.mps.is_available()` | — | GPU NVIDIA / Apple Silicon disponible ? |

> ⚠️ **Pas de conversion à la volée** : `a @ [[1],[1],[1]]` ou `a * [2., 2., 2.]` (liste Python) → **erreur de type**. Seuls les **scalaires** Python passent (`a * 2`). Convertir avec `torch.tensor(...)` d'abord.

---

## 3. Indexation et masques

| Syntaxe | Exemple | Explication |
|---|---|---|
| `x[i, j]` ≡ `x[i][j]` | `x[0, 0]` | Chaque case est **elle-même un tenseur** 0-D → `.item()` pour le nombre Python. |
| Slicing `start:end:step` | `x[:, 0]`, `x[0, 1:]`, `x[0, ::2]` | Comme numpy, `end` exclu. Renvoie une **vue** (pas de copie). |
| Liste d'indices | `x[0, [0, 2]]` | Sélection de colonnes non contiguës (copie). |
| **Masque booléen** | `a[a > 0.75] = 3` | Sélection/affectation conditionnelle vectorisée. Combiner avec `&`, `|`, `~` et **parenthèses** : `a[(a > 0.5) & (a < 1)] = 1`. |
| Comptage | `(a == 0).sum()` | Compte les éléments vérifiant la condition (somme de booléens). |
| `torch.where(cond, x, y)` | 🔎 | Choix élément par élément sans modifier `a`. |

```python
a = torch.rand(6, 6)
a[a <= 0.5] = 0
a[a > 0.75] = 3              # ⚠️ l'ordre compte : seuil haut avant le seuil intermédiaire
a[(a > 0.5) & (a < 1)] = 1
print((a == 0).sum(), (a == 1).sum(), (a == 3).sum())
```

---

## 4. Opérations, réductions, algèbre linéaire

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `+ - * / **` | tenseur ou scalaire | **Terme à terme**. `twos ** torch.tensor([[1,2],[3,4]])` : puissance élément par élément. |
| `a @ b`, `torch.matmul(a, b)`, `a.mm(b)` | $(n,k)$ @ $(k,m)$ | **Produit matriciel**. `mm` = 2D strict ; `matmul`/`@` gère aussi les batchs (3D+). |
| `c.dot(c)` | vecteurs 1D | **Produit scalaire uniquement** (contrairement à `np.dot`, pas de produit matriciel). |
| `a.t()`, `a.T` | — | Transposée (2D). |
| `t.sum(dim)`, `mean`, `std`, `prod`, `max`, `min` | `dim` (alias `axis`), `keepdim=False` | Sans `dim` : réduction totale (tenseur 0-D). `dim=0` → une valeur par colonne, `dim=1` → par ligne. `keepdim=True` garde la dimension (taille 1) pour un broadcast propre ensuite. |
| `t.argmax(dim=1)` | `dim` | Indice du max par ligne → **classe prédite**. |
| `torch.max(t, dim)` | avec `dim` | Renvoie `(valeurs, indices)`. |
| `torch.unique(t)` | `return_counts=True` | Valeurs distinctes (et effectifs). |
| `torch.abs`, `ceil`, `floor`, `clamp(t, min, max)` | bornes | Fonctions usuelles ; `clamp` écrête. |
| `torch.sin`, `asin`, `exp`, `log`, … | — | Toujours utiliser les versions **`torch`** (gradient automatique). |
| `torch.eq(a, b)`, `==`, `>`… | broadcast possible | Tenseur **booléen**. |
| `torch.inverse(A)` | matrice carrée inversible | Inverse ; ex. solution analytique des moindres carrés $w = (X^\top X)^{-1}X^\top y$. 🔎 Préférer `torch.linalg.solve` / `torch.linalg.lstsq` (plus stable). |
| `torch.svd(A)` / `torch.linalg.svd` | — | Décomposition en valeurs singulières. |
| `torch.cross(u, v)` | vecteurs 3D | Produit vectoriel. |
| `t.item()` | tenseur à 1 élément | Nombre Python (pour `print`, listes de loss…). |
| `t.tolist()` | — | Tenseur → listes Python imbriquées. |

---

## 5. Broadcasting

Règles (identiques à numpy) : on compare les dimensions **de la dernière à la première** ; chaque paire doit être **égale**, ou l'une vaut **1**, ou l'une **n'existe pas**. Aucun tenseur vide.

| Forme A | Forme B | Résultat |
|---|---|---|
| `(4, 3, 2)` | `(3, 2)` | ✅ `(4, 3, 2)` — B appliqué à chaque « couche » |
| `(4, 3, 2)` | `(3, 1)` | ✅ chaque colonne de 3 multipliée |
| `(4, 3, 2)` | `(1, 2)` | ✅ chaque ligne multipliée |
| `(4, 3, 2)` | `(4, 3)` | ❌ comparaison depuis la fin : 3≠2 |
| `(4, 3, 2)` | `(3,)` | ❌ → faire `b.unsqueeze(1)` → `(3, 1)` ✅ |
| `(5, 1)` | `(5,)` | ⚠️ ✅ **mais** → `(5, 5)` ! |

```python
a = torch.ones(5, 1) * 2      # colonne (5,1)
b = torch.ones(5)             # vecteur (5,)
(a - b).shape                 # (5, 5) : broadcast silencieux !
(a.view(-1) - b).shape        # (5,)   : ce qu'on voulait
```

> ⚠️ **Piège n°1 des fonctions de coût** : `yhat` de forme `(N, 1)` et `y` de forme `(N,)` → `yhat - y` est **(N, N)** et la MSE est fausse **sans aucune erreur**. Toujours aligner : `yhat.view(-1, 1)` **et** `y.view(-1, 1)`, ou `.view(-1)` des deux côtés.

---

## 6. Changer la forme

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `t.view(*shape)` | `-1` = joker (une seule fois) | Re-forme **sans copie** (nombre total d'éléments inchangé). Exige un tenseur **contigu**. `d.view(-1)` : aplatit en vecteur ; `e.view(-1, 1)` : vecteur → colonne. |
| `t.reshape(*shape)` / `torch.reshape(t, (n,))` | tuple si 1 seul entier : `(6*20*20,)` | Comme `view`, mais **copie si nécessaire** (ex. après `transpose`). Plus sûr. |
| `t.squeeze(dim=None)` | `dim` optionnel | Supprime les dimensions de taille 1 (toutes, ou celle indiquée). Sur une dim de taille ≠ 1 : **ne fait rien**, sans erreur. |
| `t.unsqueeze(dim)` | `dim` obligatoire | Ajoute une dimension de taille 1. `img.unsqueeze(0)` : `(3,226,226)` → `(1,3,226,226)` (**batch de 1**). `v.unsqueeze(1)` : vecteur → colonne. |
| `squeeze_()`, `unsqueeze_()` | — | Versions inplace. |
| `t.flatten(start_dim=0)` | 🔎 `start_dim=1` garde le batch | Aplatit ; usage typique entre convolutions et couche linéaire. |
| `t.permute(*dims)` / `t.transpose(d0, d1)` | 🔎 nouvel ordre des axes | Réordonne les axes (vue, souvent non contiguë → `reshape` ensuite). |

```python
output3d = torch.rand(6, 20, 20)
input1d = output3d.reshape(6 * 20 * 20)          # (2400,)
e2 = torch.ones(4, 1).squeeze()                  # (4,)
e2.unsqueeze(0).shape, e2.unsqueeze(1).shape     # (1, 4), (4, 1)
```

---

## 7. Inplace, copies, mémoire partagée

| Outil | Explication |
|---|---|
| Suffixe `_` : `a.add_(b)`, `b.mul_(b)`, `torch.sin_(b)` | Modifie **l'objet appelant** au lieu de créer un tenseur. Gain mémoire. |
| `out=` : `torch.matmul(a, b, out=c)`, `torch.rand(2, 2, out=c)` | Écrit le résultat dans un tenseur **déjà alloué** (bonne forme et dtype) → pas de nouvelle allocation. |
| `b = a` | **Pas une copie** : deux noms pour le même tenseur. |
| `b = a.clone()` | **Vraie copie** des données (reste dans le graphe d'autograd). |
| `t.detach()` | 🔎 Même données, **détachées du graphe** (pas de gradient). `t.detach().clone()` pour une copie indépendante. |

> ⚠️ Les **vues** (`view`, slicing, `reshape` quand il peut) partagent la mémoire : modifier la source modifie la vue.

---

## 8. Pont NumPy / Python / matplotlib

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torch.from_numpy(arr)` | array numpy | Tenseur **partageant la mémoire** avec `arr` (garde le dtype : float64 !). |
| `t.numpy()` | tenseur CPU, sans gradient | Vue numpy **partageant la mémoire**. Si `requires_grad` : `t.detach().numpy()` ; si GPU : `t.cpu()` d'abord. |
| `torch.tensor(arr, dtype=torch.float)` | | **Copie** + typage : le réflexe pour des données sklearn. |
| `plt.imshow(t.tolist())` / `plt.imshow(t.numpy())` | | Afficher une matrice (sortir de torch d'abord). |
| Tracer sous `with torch.no_grad():` | | Évite les erreurs sur des tenseurs avec gradient (`plt.plot(x, y)`). |

```python
pytorch_tensor = torch.from_numpy(np.ones((2, 3)))   # dtype float64, mémoire partagée
X = torch.tensor(X_train, dtype=torch.float)          # copie + float32 : réflexe
```

---

# II. Autograd & descente de gradient

## 9. Autograd : `requires_grad`, `backward`, `grad`

| Élément | Paramètres essentiels | Explication |
|---|---|---|
| `requires_grad=True` (création) ou `t.requires_grad = True` | défaut `False` | Demande d'**enregistrer** dans le graphe de calcul toute opération utilisant `t`. À activer sur les **paramètres** (w, b), **pas** sur les données. |
| `loss.backward()` | `loss` **scalaire** ; `retain_graph=False` | Rétro-propage : remplit `.grad` de chaque feuille avec $\partial \text{loss}/\partial t$. Le graphe est **libéré** ensuite (recalculer le coût avant un nouvel appel). |
| `t.grad` | — | Gradient, **même forme** que `t`. `None` avant le premier `backward`. **S'accumule** (`+=`) à chaque `backward`. |
| `t.grad.zero_()` | — | Remise à zéro manuelle (à chaque itération). |
| `t.data` | — | Valeurs hors graphe (ancienne façon de faire les MAJ). |
| `with torch.no_grad():` | — | Bloc où rien n'est enregistré : MAJ des paramètres, évaluation, affichage. |
| `t.detach()` | — | Copie détachée du graphe. |

```python
y    = torch.tensor([1., 2., 1.], requires_grad=True)
yhat = torch.tensor([0., 0., 0.], requires_grad=True)

def MSE(yhat, y):                      # 1/(2N) Σ ||ŷ_i − y_i||²
    return ((yhat - y)**2).sum() / (2 * y.shape[0])

mse = MSE(yhat, y); mse.backward()
print(yhat.grad, y.grad)               # (ŷ − y)/N  et  (y − ŷ)/N
```

Hypothèses : $\hat y, y \in \mathbb R^{N\times d}$, $\hat y_i, y_i$ leurs lignes.

$$\mathrm{MSE}(\hat y, y) = \frac{1}{2N}\sum_{i=1}^{N}\lVert \hat y_i - y_i\rVert^2, \qquad \nabla_{\hat y}\,\mathrm{MSE} = \frac{1}{N}(\hat y - y), \quad \nabla_{y}\,\mathrm{MSE} = \frac{1}{N}(y - \hat y)$$

- $N$ : nombre d'exemples (lignes) ; le facteur $\tfrac12$ simplifie la dérivée.

> 💡 Le gradient d'une **distance** entre deux points donne la **direction** qui les rapproche : avec $C = \lVert A-B\rVert^2$, $-\nabla_A C$ pointe de $A$ vers $B$ (et inversement).

---

## 10. Mettre à jour les paramètres à la main

```python
EPOCHS, EPS = 5000, 1e-7
w = torch.randn(1, X.size(1), requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
loss_h = []
for i in range(EPOCHS):
    loss = ((X @ w.T + b - Y.view(-1, 1))**2).mean()   # forward + coût
    loss_h.append(loss.item())                          # .item() : pas de graphe stocké
    loss.backward()
    with torch.no_grad():                               # SOLUTION 2 (recommandée)
        w -= EPS * w.grad                               # inplace OBLIGATOIRE
        b -= EPS * b.grad
    # SOLUTION 1 équivalente : w.data -= EPS * w.grad.data
    w.grad.zero_(); b.grad.zero_()
```

> ⚠️ `w = w - EPS * w.grad` crée un **nouveau tenseur** (non feuille, sans `.grad`) : il ne se reconnecte pas au graphe → l'apprentissage casse. Toujours `-=` / `+=` sous `no_grad`.

> 💡 Si les paramètres ont des échelles très différentes (ex. coefficients d'un polynôme $ax^4+\dots+e$), un **pas différent par paramètre** peut être nécessaire en SGD pur… ou utiliser **Adam** (pas adaptatif).

> 👉 Si ça diverge (loss → `nan`/`inf`) : **baisser `EPS`**, normaliser les données ; si ça ne converge pas avec le pas donné par l'énoncé, l'erreur est presque toujours dans **le coût** (formes, voir §5).

---

## 11. Optimiseurs `torch.optim`

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `torch.optim.SGD(params, lr, momentum=0, weight_decay=0)` | `params` : liste `[w, b]` ou `model.parameters()` ; `lr` = pas | Descente de gradient classique $\theta \leftarrow \theta - \mathrm{lr}\,\nabla_\theta L$. Sensible au réglage de `lr`. `momentum=0.9` accélère. |
| `torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0)` | `lr` défaut `1e-3` | Pas **adaptatif par paramètre** : marche « out of the box » avec `1e-3`. Le réflexe par défaut. |
| `optim.zero_grad()` | — | Remet tous les `.grad` à zéro. |
| `optim.step()` | — | Applique la mise à jour (sous `no_grad` en interne). |

```python
optim = torch.optim.Adam(params=[w, b], lr=1e-3)
for i in range(EPOCHS):
    loss = MSE(flineaire(X, w, b).view(-1, 1), Y.view(-1, 1))
    optim.zero_grad()      # 1. reset
    loss.backward()        # 2. gradients
    optim.step()           # 3. MAJ
```

> 👉 Ordre canonique : `zero_grad()` → `forward/loss` → `backward()` → `step()` (le `zero_grad` peut être avant ou après le forward, mais **avant** `backward`).
> 💡 **Adoucir l'initialisation** (`with torch.no_grad(): w *= 0.1`) facilite l'optimisation.

---

## 12. Régression linéaire complète + baselines sklearn

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
housing = fetch_california_housing(data_home="./data/")
X_train, X_test, y_train, y_test = train_test_split(
    housing['data'], housing['target'], test_size=0.33, random_state=42)
Xtr = torch.tensor(X_train, dtype=torch.float); ytr = torch.tensor(y_train, dtype=torch.float)

def flineaire(x, w, b):          # x (n,d), w (1,d), b scalaire → (n,1)
    return x @ w.T + b           # marche aussi pour un seul x de forme (d,) → (1,)
```

| Baseline (sklearn) | Code | Rôle |
|---|---|---|
| Modèle **naïf** (moyenne) | `mean_squared_error(y_test, y_test.mean()*np.ones(y_test.shape))` | Plancher : tout modèle doit faire mieux. |
| `LinearRegression()` | `.fit(X_train, y_train)`, `.predict(X_test)` | Solution exacte des moindres carrés : la cible de votre descente de gradient. |
| `GradientBoostingRegressor()` | idem | Référence non linéaire forte. |

Solution analytique (moindres carrés) — hypothèses : $X \in \mathbb R^{N\times p}$ de rang plein (colonne de 1 incluse pour le biais) :

$$\mathbf w^\star = (X^\top X)^{-1} X^\top \mathbf y$$

- $\mathbf y \in \mathbb R^{N\times 1}$ : cibles ; $\mathbf w^\star$ : coefficients optimaux.

```python
X2 = torch.cat((X0**3, X0**2, X0, torch.ones(X0.size())), dim=1)  # features polynomiales
w = torch.inverse(X2.T @ X2) @ X2.T @ X1
```

> 💡 La descente de gradient n'est « que le 4×4 de l'optimisation » : pour un problème linéaire de petite taille, la solution analytique est plus rapide et exacte. Le gradient devient indispensable quand il n'y a **pas** de solution fermée (réseaux de neurones).
> 👉 Toujours évaluer sur le **test** (`housing_xT`), jamais seulement la loss d'entraînement.

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| `(N,1) - (N,)` dans le coût | MSE énorme/fausse, pas d'erreur | `.view(-1, 1)` des deux côtés |
| Oubli de `zero_grad` / `.grad.zero_()` | gradients qui grossissent, divergence | remettre à zéro à **chaque** itération |
| `w = w - lr*w.grad` | `w.grad` devient `None`, erreur au tour suivant | `w -= ...` sous `torch.no_grad()` |
| `backward()` deux fois sur le même coût | `RuntimeError: Trying to backward through the graph a second time` | recalculer le coût (ou `retain_graph=True`) |
| `loss_h.append(loss)` | mémoire qui explose (graphe gardé) | `loss.item()` |
| Données en float64 (sklearn/numpy) | `RuntimeError: expected Float but found Double` | `dtype=torch.float` / `.float()` |
| Labels de classe en float | erreur dans les losses de classification | `.long()` |
| Liste Python dans une opération | `TypeError` | convertir en tenseur |
| `torch.empty` utilisé tel quel | valeurs aléatoires absurdes | initialiser explicitement |
| `b = a` pris pour une copie | modification « fantôme » | `a.clone()` |
| `from_numpy` / `.numpy()` | mémoire partagée : modifier l'un modifie l'autre | `.clone()` / `torch.tensor(arr)` pour copier |
| `.numpy()` sur un tenseur avec gradient | `RuntimeError: Can't call numpy() on Tensor that requires grad` | `.detach().numpy()` |
| `view` après `permute`/`transpose` | `RuntimeError: view size is not compatible` | `.reshape(...)` ou `.contiguous().view(...)` |
| `math.sin` / `np.exp` dans un calcul à dériver | pas de gradient | fonctions `torch.*` |
| Masques combinés sans parenthèses | erreur de précédence | `(a > 0.5) & (a < 1)` |
