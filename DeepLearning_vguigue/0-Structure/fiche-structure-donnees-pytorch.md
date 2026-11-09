# 📘 Fiche de révision — Module 1 : Structure de données PyTorch

> Cours de Vincent Guigue (AgroParisTech – IODAA, semestre 1) · 22 diapos (`Structure.pdf`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 notes perso

---

## Sommaire

1. [Configuration générale](#1-configuration-générale)
2. [Calcul matriciel](#2-calcul-matriciel)
3. [Gradient](#3-gradient)
4. [Régression linéaire en PyTorch](#4-régression-linéaire-en-pytorch)
5. [Conclusion : ce qu'est PyTorch](#5-conclusion--ce-quest-pytorch)
6. [Pièges à éviter](#6-pièges-à-éviter)
7. [Quiz auto-test](#7-quiz-auto-test)

---

## 1. Configuration générale

### 1.1 Un environnement performant mais dynamique (diapo 2)

Le domaine de l'IA évolue vite ⇒ les librairies aussi. **L'ajout d'un paquet peut mettre à jour des dépendances** et engendrer des incompatibilités.

| Règle | Contenu |
|---|---|
| **Règle 1** | Ne **pas** utiliser la dernière version de Python : rester sur une version d'il y a 6 mois / 1 an (Python 3.12 en 2025). |
| **Règle 2** | Travailler avec des **environnements** (conda). |
| **Règle 3** | Savoir **ré-installer** sa machine en cas de problème (et/ou basculer très vite sur **Colab**). |

```bash
conda create --name nom_de_votre_environnement python=3.12
conda activate nom_de_votre_environnement
# (nom_de_votre_environnement) >>
```

> ⚠️ **ATTENTION : `numpy < v2`** (incompatibilités avec certaines librairies du cours).
>
> 💡 En cas de doute : demander à un assistant (ChatGPT ou équivalent).

### 1.2 Sauver / charger un environnement (diapo 3)

| Niveau | Commande | Usage |
|---|---|---|
| **Paquets** (le plus simple) | `pip freeze > requirements.txt` | Sauvegarde les paquets **avec leur version** |
| | `pip install -r requirements.txt` | Ré-installe tous les paquets **dans les bonnes versions** |
| **Environnement complet** | `conda env export > environment.yml` | Exporte tout l'environnement conda |

> 🔎 Pour recréer depuis le `.yml` : `conda env create -f environment.yml`.

---

## 2. Calcul matriciel

### 2.1 Matrices & philosophie générale (diapo 4)

- **Matrice = idéale pour stocker un ensemble de données** (ex. ligne = individu, colonne = descripteur).
- **Les matrices sont typées** : `torch.int16`, `torch.float32`… On cherche souvent à **gagner de la place** → toujours avoir en tête un **ordre de grandeur de la mémoire** utilisée.
- **Philosophie** = **opérateurs de haut niveau sur les matrices** ≠ boucles.
- **Toutes les structures sont `torch`** → passage plus facile sur **GPU** + **calcul de gradient**.

> 💡 Réflexions posées en cours : *Combien pèse un réel ? Quantification des LLM ? Combien de valeurs possibles sur 4 bits ?*
>
> 🔎 Réponses : un `float32` pèse **4 octets** (un `float64` 8 octets, un `float16`/`bfloat16` 2 octets). Sur $n$ bits on code $2^n$ valeurs, donc **16 valeurs sur 4 bits**. La **quantification** des LLM consiste à stocker les poids sur moins de bits (8 ou 4) pour diviser la mémoire par 4 à 8, au prix d'une perte de précision. Ex. un modèle de 7 milliards de paramètres : $7\cdot10^9 \times 4$ octets ≈ 28 Go en float32, ≈ 3,5 Go en 4 bits.

### 2.2 Parallélisation des calculs matriciels / GPU (diapo 5)

- Pas le but premier du cours, mais un **intérêt évident en tant qu'utilisateur** : chaque case $c_{ij}$ d'un produit matriciel se calcule **indépendamment** (ligne de $A$ × colonne de $B$) → parallélisable.
- La courbe montrée (produit matrice-vecteur sur GPU) : le temps de calcul GPU (cuBLAS, kernel) reste bas quand le nombre de lignes croît.

> ⚠️ **Attention au coût de transfert entre les mémoires** (CPU ↔ GPU) : déplacer les données coûte cher, il faut les transférer une fois et calculer longtemps dessus.

### 2.3 Créer des matrices (diapo 6)

```python
import torch
# créer une matrice de 4 lignes & 3 colonnes
cst = torch.tensor([[3.1, 2.7, 1.9], [1.6, 0.0, 9.8], [1.7, 8.1, 5.3]])
zeros = torch.zeros(4, 3)
ones  = torch.ones(4, 3)
x     = torch.rand(4, 3)          # uniforme sur [0, 1[
zeros_like_x = torch.zeros_like(x)  # même forme (et type) que x
```

**Représentation mémoire d'un tenseur** (schéma de la diapo, matrice `x` 4×3 d'entiers 0…11) :

| Champ | Valeur dans l'exemple | Signification |
|---|---|---|
| `data` | pointeur vers 0,1,…,11 contigus | les valeurs sont stockées **à plat**, à la suite |
| `data type` | entier 8 octets (`int64`) | type élémentaire |
| `shape` | `(4, 3)` | forme logique |
| `strides` | `(24, 8)` | nb d'octets à sauter pour avancer d'1 sur chaque axe : **8 octets par élément** (colonne suivante), **3 × 8 = 24 octets** pour descendre d'une ligne |

> 🔎 C'est ce mécanisme (`shape` + `strides` sur un même bloc mémoire) qui permet à `view`, `transpose`, `permute`, le slicing… de **ne pas copier** les données : seule la manière de lire le bloc change. (Dans PyTorch, `x.stride()` renvoie les pas en nombre d'**éléments** : `(3, 1)`.)

### 2.4 Accéder aux valeurs (diapo 7)

> 💡 **Penser aux motifs plutôt qu'aux boucles.**

```python
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
print(x[0][0], " or ", x[0, 0])   # chaque case est elle-même un tenseur
print(x[:, 0])                    # toutes les lignes, 1re colonne
print(x[0, [0, 2]])               # 1re ligne, colonnes 0 et 2
# syntaxe générale sur lignes ou colonnes : start:end(exclu):step
print(x[0, 0:2], " or ", x[0, :2])     # borne haute
print(x[0, 1:3], " or ", x[0, 1:])     # borne basse
print(x[0, 0:3:2], " or ", x[0, ::2])  # pas = 2
```

### 2.5 Calcul entre matrices (diapo 8)

Opérateurs disponibles : `+`, `-`, `*`, `/`, `**`.

```python
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
y = x + 2        # avec un scalaire : [[4, 6, 8], [5, 8, 11]]
z = x * y        # terme à terme    : [[8, 24, 48], [15, 48, 99]]
a = x @ torch.tensor([[1], [2], [3]])   # produit matriciel (2×3)@(3×1) → (2×1)
```

| Opération | Syntaxe | Forme |
|---|---|---|
| Terme à terme (Hadamard) | `x * y` | mêmes formes (ou *broadcastables*) |
| Produit matriciel | `x @ y` (ou `torch.matmul`, `x.mm(y)`) | $(n,k)\,@\,(k,m) \to (n,m)$ |

### 2.6 *Inplace* (ou pas) (diapo 9)

Selon les situations, pour **gagner de la place** :

| Mode | Code | Effet |
|---|---|---|
| Opération **classique** | `b = torch.sin(a)` ; `c = a + b` | résultat stocké dans un **nouveau** tenseur |
| Opération **inplace** | `torch.sin_(a)` ; `a.add_(b)` | **modifie l'argument** (`a`) / **l'objet** (`a`) |

> 💡 Convention : le suffixe **`_`** signale une opération *inplace*.

### 2.7 Calculs agrégatifs (diapo 10)

```python
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
li  = x.sum(1)          # ou x.sum(axis=1) → somme par ligne   : [12, 18]
co  = x.sum(0)          # somme par colonne                     : [5, 10, 15]
tot = x.sum()           # somme totale                          : 30
li  = x[:, ::2].sum(1)  # somme des colonnes 0 et 2
```

| Axe | Ce qui « disparaît » | Résultat sur l'exemple |
|---|---|---|
| `axis=0` (ou `dim=0`) | les **lignes** (on agrège verticalement) | une valeur par colonne |
| `axis=1` | les **colonnes** (on agrège horizontalement) | une valeur par ligne |

- Même logique pour **somme, moyenne, min, max**, …
- … **et on ne s'arrête pas à deux axes** (tenseurs 3D, 4D…).

### 2.8 `squeeze` / `unsqueeze` (diapo 11)

- Ligne de données = 1D ; image = 2D ; corpus d'images = 3D.
- **Donner une image en entrée du système ⇒ `unsqueeze`** : ajouter une dimension homogène à un **batch**.
- **Récupérer (& analyser) une sortie intermédiaire ⇒ `squeeze`** : passer de batch à **individu**.

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)
print(a.shape)  # torch.Size([3, 226, 226])
print(b.shape)  # torch.Size([1, 3, 226, 226])  → batch de 1 image
```

> 🔎 `squeeze()` ne supprime **que** des dimensions de taille 1 ; `squeeze(d)` sur une dimension de taille ≠ 1 ne fait rien (pas d'erreur).

---

## 3. Gradient

### 3.1 Tensor : données… et gradient (diapo 12)

La structure `tensor` intègre aussi un **champ `grad`**.

```python
a = torch.tensor(1.)
a.requires_grad = True                    # activation du champ gradient
b = torch.tensor(2., requires_grad=True)
z = 2*a + b
z.backward()                              # calcul des dérivées partielles de z
print("Dérivée de z/a : ", a.grad.item(), " z/b : ", b.grad.item())
# Dérivée de z/a :  2.0  z/b :  1.0
```

- $z$ est **connecté** à $a$ et $b$ : c'est le **graphe de calcul**.
- $z \approx$ **fonction de coût** ⇒ le gradient nous donne la **direction** pour la minimiser.

Hypothèses : $a, b \in \mathbb R$, $z = 2a + b$.

$$\frac{\partial z}{\partial a} = 2, \qquad \frac{\partial z}{\partial b} = 1$$

- $\frac{\partial z}{\partial a}$ : dérivée partielle de $z$ par rapport à $a$ (stockée dans `a.grad`) ;
- $\frac{\partial z}{\partial b}$ : idem pour $b$ (stockée dans `b.grad`).

> 🔎 Le calcul n'est **pas symbolique** : les gradients sont **évalués numériquement en un point** (les valeurs courantes de $a$ et $b$).

### 3.2 Gradient sur un vecteur / une matrice (diapo 13)

Les `tensor` sont en général des vecteurs, des matrices… (pas forcément de « vrais » tenseurs). **Le gradient a la même dimension que la variable d'origine.**

Hypothèses : $A \in \mathbb R^{n \times m}$ variable avec gradient **actif**, $L = \mathrm{fct}(A)$ **scalaire**.

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots \\ a_{21} & a_{22} & \cdots \end{pmatrix}, \qquad
\texttt{L.backward()} \;\Rightarrow\; \texttt{A.grad} = \nabla_A L = \begin{pmatrix} \frac{\partial L}{\partial a_{11}} & \frac{\partial L}{\partial a_{12}} & \cdots \\[2pt] \frac{\partial L}{\partial a_{21}} & \frac{\partial L}{\partial a_{22}} & \cdots \end{pmatrix}$$

- $L$ : la sortie (typiquement le **coût**), **doit être un scalaire** ;
- $\nabla_A L$ : gradient de $L$ par rapport à $A$, de **même forme que $A$** ;
- $A$ : variables sur lesquelles le **gradient est actif** (`requires_grad=True`).

> ⚠️ `L.backward()` sur un $L$ non scalaire lève une erreur (🔎 sauf à fournir un vecteur `gradient=` : produit Jacobienne-vecteur).

### 3.3 Syntaxe une fois le gradient activé (diapo 14)

```python
b = torch.tensor(2., requires_grad=True)
# print(b)   # ⇒ ERR : ambigu ! (selon les versions)
b.data       # ok : les valeurs, hors graphe
b.grad.data  # ok : les valeurs du gradient
```

**Plus efficace en coupant le graphe de calcul**, par ex. pour la **mise à jour des paramètres** :

```python
# Régression logistique basée sur w et b
[...]
# mise à jour des paramètres :
with torch.no_grad():
    w += -EPS * w.grad
    b += -EPS * b.grad
```

> 💡 Sous `torch.no_grad()`, les opérations ne sont **pas enregistrées** dans le graphe : la mise à jour ne devient pas elle-même une opération à dériver.

### 3.4 Accumulation du gradient (diapo 15)

- Appels **multiples** à `backward`, à partir d'une fonction de coût ou de plusieurs ⇒ **accumulation des gradients** (ils s'**additionnent** dans `.grad`).

```python
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
z = 2*a + b
z.backward()
z = 2*a + b   # il faut redéfinir le coût
z.backward()  # puis relancer le gradient = accumulation → a.grad = 4, b.grad = 2
```

**Mettre à 0 les gradients à la main :**

```python
a.grad.zero_()
b.grad.zero_()
```

> 🔎 Pourquoi « il faut redéfinir le coût » : après un `backward()`, PyTorch **libère le graphe** ; rappeler `backward()` sur le même `z` lève une erreur (sauf `retain_graph=True`).

### 3.5 Fonctions (d'activation) (diapo 16)

**Seule option pour que les gradients se calculent automatiquement dans les fonctions : utiliser des fonctions `torch` !**

```python
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
z = torch.sin(2*a + b)   # et non math.sin / np.sin
z.backward()
```

La plupart des fonctions existent dans l'univers PyTorch.

> 🔎 Ici $\frac{\partial z}{\partial a} = 2\cos(2a+b)$ et $\frac{\partial z}{\partial b} = \cos(2a+b)$, évalués en $(a,b) = (1,2)$.

---

## 4. Régression linéaire en PyTorch

### 4.1 Quels verrous pour une régression linéaire ? (diapo 17)

1. **Lire les données** ⇒ utilisation de `scikit-learn`
2. Construire l'**estimateur linéaire**
3. Appliquer la **fonction coût**
4. Trouver un **critère d'arrêt** pour la descente de gradient
5. **Évaluer** les performances

### 4.2 Accès aux données (diapo 18)

- Récupération des données → transformation en `tensor` → **vérification des dimensions** de la structure créée.

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()   ## chargement des données
# penser à typer les données pour éliminer les incertitudes
X = torch.tensor(housing['data'],   dtype=torch.float)
Y = torch.tensor(housing['target'], dtype=torch.float)
```

> ⚠️ **Tous les types sont spécifiques** :
>
> | Niveau | Types |
> |---|---|
> | Structure de données | `torch.tensor` … |
> | Typage élémentaire des données | `torch.float`, `torch.int`, … |

### 4.3 Estimateur linéaire (diapo 19)

**Déclarer les paramètres en activant le gradient :**

```python
w = torch.randn(1, X.size(1), requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
```

**Construire l'estimateur linéaire** : `(x @ w.T) + b`

Hypothèses : $\mathbf x \in \mathbb R^{N\times d}$ (N exemples, d descripteurs), $\mathbf w \in \mathbb R^{1\times d}$, $b \in \mathbb R$.

$$\hat{\mathbf y} = \mathbf x\,\mathbf w^\top + b$$

- $\hat{\mathbf y} \in \mathbb R^{N\times 1}$ : prédictions ;
- $\mathbf w$ : poids (un par descripteur) ; $b$ : biais (diffusé sur les N lignes par *broadcasting*).

Puis **construire la fonction de coût**… **et c'est (presque) tout**.

> 🔎 Coût usuel (moindres carrés), hypothèses : $y_i, \hat y_i \in \mathbb R$, $i = 1..N$ :
> $$\mathrm{MSE}(\hat{\mathbf y}, \mathbf y) = \frac{1}{N}\sum_{i=1}^{N} (\hat y_i - y_i)^2$$
> - $y_i$ : vraie valeur ; $\hat y_i$ : prédiction ; $N$ : nombre d'exemples.

### 4.4 Autograd et graphe de calcul (diapo 20)

Le schéma décompose le coût d'un exemple $(\mathbf x, y)$ :

| Nœud | Calcul |
|---|---|
| $s$ | $\langle \mathbf w, \mathbf x\rangle$ |
| $\hat y$ | $s + b$ |
| $d$ | $y - \hat y$ |
| $e$ | $d^2 = (y - (\langle \mathbf w, \mathbf x\rangle + b))^2$ |

**Passe forward** (noir) : on calcule $s \to \hat y \to d \to e$.
**Passe backward** (rouge) : on enchaîne les dérivées (**règle de dérivation en chaîne**) :

$$\frac{\partial e}{\partial d} = 2d, \quad
\frac{\partial e}{\partial d}\frac{\partial d}{\partial y} = 2d, \quad
\frac{\partial e}{\partial d}\frac{\partial d}{\partial \hat y} = -2d, \quad
\frac{\partial e}{\partial d}\frac{\partial d}{\partial \hat y}\frac{\partial \hat y}{\partial b} = -2d,$$

$$\frac{\partial e}{\partial d}\frac{\partial d}{\partial \hat y}\frac{\partial \hat y}{\partial s}\,\nabla_{\mathbf w} s = -2d\,\mathbf x, \qquad
\frac{\partial e}{\partial d}\frac{\partial d}{\partial \hat y}\frac{\partial \hat y}{\partial s}\,\nabla_{\mathbf x} s = -2d\,\mathbf w$$

- $d$ : résidu (erreur signée) ; $e$ : erreur quadratique ;
- $\nabla_{\mathbf w} s = \mathbf x$ et $\nabla_{\mathbf x} s = \mathbf w$ (gradients du produit scalaire).

> 💡 **Graphe de calcul** :
> - **graphe orienté** qui décrit l'enchaînement des opérations de calcul ;
> - *Source* = variables d'entrée + nœud de sortie : le résultat du calcul ;
> - en connaissant les **dérivées de chaque opération**, le graphe permet de calculer le **gradient de la sortie par rapport à chaque variable d'entrée**.

### 4.5 Autres verrous (diapo 21)

- **Pas de critère d'arrêt** pour l'instant (5000 itérations ?).
- **Calculer la performance.**
- **Ne pas oublier de mettre à 0 le gradient** pour éviter l'accumulation après la mise à jour.

> 🔎 Boucle complète « à la main » (ce que demande le TP) :
> ```python
> for i in range(EPOCHS):
>     loss = ((X @ w.T + b - Y.view(-1, 1))**2).mean()   # forward + coût
>     loss.backward()                                     # gradients dans w.grad, b.grad
>     with torch.no_grad():                               # MAJ hors graphe
>         w -= EPS * w.grad
>         b -= EPS * b.grad
>     w.grad.zero_(); b.grad.zero_()                      # éviter l'accumulation
> ```
> Critères d'arrêt classiques : nombre max d'itérations, variation du coût sous un seuil, ou coût de **validation** qui remonte (*early stopping*).

---

## 5. Conclusion : ce qu'est PyTorch

### 5.1 Ce que nous verrons dans ce module (diapo 22)

**PyTorch, c'est…**

- un **framework de développement + apprentissage** de réseaux *deep* sur **CPU et GPU** ;
- une **architecture modulaire** de contenants + conteneurs ⇒ architectures **flexibles** ;
- la **différenciation automatique** ⇒ **Autograd** ;
- une **couche d'abstraction pour l'optimisation** ⇒ variété de **descentes de gradient** ;
- une **gestion simplifiée des données** pour la constitution des **mini-batchs**.

### 5.2 PyTorch vs TensorFlow

| Critère | En faveur de |
|---|---|
| Intégration dans l'industrie | TensorFlow (PyTorch « un peu » moins intégré) |
| Déploiement, rapidité, processus industriel | **TensorFlow** |
| Flexibilité, prototypage, simplicité | **PyTorch** |

Les deux frameworks ont tendance à **se rapprocher** en fonctionnalités.

---

## 6. Pièges à éviter

> 🚩 **Oublier `requires_grad=True`** sur les paramètres : `backward()` lève une erreur (aucun tenseur du graphe n'exige de gradient).

> 🚩 **Oublier de remettre le gradient à 0** (`.grad.zero_()` ou `optimizer.zero_grad()`) : les gradients **s'accumulent** d'une itération à l'autre → descente faussée.

> 🚩 **Mettre à jour les paramètres hors `torch.no_grad()`** (ou sans `.data`) : la mise à jour est enregistrée dans le graphe. Et écrire `w = w - eps*w.grad` (au lieu de `w -= …`) crée un **nouveau tenseur** déconnecté (plus de `.grad`).

> 🚩 **Utiliser `math.sin` / `np.…` au lieu de `torch.sin`** dans un calcul à dériver : pas de gradient automatique.

> 🚩 **`L.backward()` sur un $L$ non scalaire** : il faut réduire (somme, moyenne) d'abord.

> ⚠️ **Confondre `*` (terme à terme) et `@` (produit matriciel).**

> ⚠️ **Axes d'agrégation** : `sum(0)` agrège les **lignes** (résultat par colonne), `sum(1)` agrège les **colonnes** (résultat par ligne).

> ⚠️ **Typage** : PyTorch ne convertit pas à la volée (float vs double vs int) ; typer explicitement à la création (`dtype=torch.float`).

> ⚠️ **Coût de transfert CPU ↔ GPU** : ne pas faire d'allers-retours inutiles.

> ⚠️ **Environnement** : pas la toute dernière version de Python, `numpy < 2`, figer les versions (`pip freeze`).

---

## 7. Quiz auto-test

<details>
<summary><b>Question 1 (3 pts) :</b> Que contient z.grad / a.grad après z = 2a + b puis z.backward(), avec a=1 et b=2 (requires_grad activé) ? Et après un deuxième z = 2a + b ; z.backward() sans remise à zéro ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Après le premier `backward()` : `a.grad = 2`, `b.grad = 1` (car $\partial z/\partial a = 2$, $\partial z/\partial b = 1$).

Après le second (coût redéfini puis `backward()`), les gradients **s'accumulent** : `a.grad = 4`, `b.grad = 2`. Il faut faire `a.grad.zero_()` et `b.grad.zero_()` entre deux itérations.
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Pourquoi faut-il faire la mise à jour des paramètres dans un bloc with torch.no_grad() ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Pour **couper le graphe de calcul** : la mise à jour $w \leftarrow w - \varepsilon \nabla_w L$ ne doit pas être enregistrée comme une opération à dériver (ce serait faux et coûteux). Alternative : passer par `w.data`. Il faut aussi faire une opération *inplace* (`-=`, `+=`) pour que `w` reste le même tenseur feuille avec son champ `.grad`.
</details>

<details>
<summary><b>Question 3 (2 pts) :</b> Quelle forme a A.grad si A est une matrice 3×4 et L = fct(A) un scalaire ? Quelle condition sur L ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

`A.grad` a **la même forme que A** : 3×4, avec $(\nabla_A L)_{ij} = \frac{\partial L}{\partial a_{ij}}$. $L$ **doit être un scalaire** pour appeler `L.backward()` sans argument.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> x = torch.tensor([[2,4,6],[3,6,9]]). Que valent x.sum(0), x.sum(1) et x[:, ::2].sum(1) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- `x.sum(0)` = `[5, 10, 15]` (somme par colonne) ;
- `x.sum(1)` = `[12, 18]` (somme par ligne) ;
- `x[:, ::2].sum(1)` = colonnes 0 et 2 → `[2+6, 3+9]` = `[8, 12]`.
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> Une image a la forme (3, 226, 226). Comment la donner à un modèle qui attend un batch ? Quelle forme obtient-on ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

`img.unsqueeze(0)` → forme `(1, 3, 226, 226)` : un batch de 1 image. À l'inverse, `squeeze` permet de revenir à l'individu quand on récupère une sortie intermédiaire.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Dans le graphe de calcul de e = (y − (⟨w,x⟩ + b))², que vaut le gradient de e par rapport à w et par rapport à b ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Avec $d = y - \hat y$ et $\hat y = \langle \mathbf w, \mathbf x\rangle + b$ :

$$\nabla_{\mathbf w} e = \frac{\partial e}{\partial d}\frac{\partial d}{\partial \hat y}\frac{\partial \hat y}{\partial s}\nabla_{\mathbf w}s = 2d \cdot (-1)\cdot 1 \cdot \mathbf x = -2d\,\mathbf x, \qquad \frac{\partial e}{\partial b} = -2d$$

Le graphe stocke la dérivée locale de chaque opération et les **enchaîne** (dérivation en chaîne) depuis la sortie vers les entrées (passe *backward*).
</details>

<details>
<summary><b>Question 7 (1 pt) :</b> Quelle différence entre torch.sin(a) et torch.sin_(a) ; entre a + b et a.add_(b) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

`torch.sin(a)` et `a + b` créent un **nouveau** tenseur. `torch.sin_(a)` et `a.add_(b)` sont **inplace** (suffixe `_`) : ils **modifient** `a` directement, ce qui économise de la mémoire.
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Citer les 5 verrous d'une régression linéaire en PyTorch selon le cours. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Lire les données (scikit-learn) ; 2. construire l'estimateur linéaire `(x @ w.T) + b` ; 3. appliquer la fonction coût ; 4. trouver un critère d'arrêt pour la descente de gradient ; 5. évaluer les performances.
</details>

<details>
<summary><b>Question 9 (1 pt) :</b> Combien de valeurs peut-on coder sur 4 bits, et combien pèse un float32 ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$2^4 = 16$ valeurs. Un float32 pèse 4 octets (32 bits). C'est l'enjeu de la **quantification** des LLM (stocker les poids sur 8 ou 4 bits).
</details>

<details>
<summary><b>Question 10 (1 pt) :</b> Quels sont les points forts respectifs de PyTorch et TensorFlow selon le cours ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

PyTorch : flexibilité, prototypage, simplicité. TensorFlow : déploiement, rapidité, processus industriel (plus intégré dans l'industrie). Les deux se rapprochent.
</details>

<details>
<summary><b>Question 11 (1 pt) :</b> Comment sauvegarder puis réinstaller les paquets d'un environnement avec leurs versions ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

`pip freeze > requirements.txt` puis `pip install -r requirements.txt`. Pour l'environnement complet : `conda env export > environment.yml`.
</details>
