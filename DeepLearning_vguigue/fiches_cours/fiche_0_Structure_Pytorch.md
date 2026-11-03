# 🧱 Fiche de cours — Module 1 : Structure de données PyTorch

> Source : diapos `0.Structure_Pytorch.pdf` + 📝 mes notes `0.Structure_Pytorch.md`. Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours) · 📝 mes notes perso.

## Sommaire
1. [Matrices et philosophie générale](#1-matrices-et-philosophie-générale)
2. [Créer et accéder aux valeurs](#2-créer-et-accéder-aux-valeurs)
3. [Calcul entre matrices, in-place, agrégats](#3-calcul-entre-matrices-in-place-agrégats)
4. [`squeeze` / `unsqueeze`](#4-squeeze--unsqueeze)
5. [Gradient et `backward`](#5-gradient-et-backward)
6. [Autograd et graphe de calcul](#6-autograd-et-graphe-de-calcul)
7. [Régression linéaire en PyTorch](#7-régression-linéaire-en-pytorch)
8. [Conclusion : ce qu'est PyTorch](#8-conclusion--ce-quest-pytorch)
9. [Quiz d'auto-test](#9-quiz-dauto-test)
10. [🚩 Pièges à éviter](#10--pièges-à-éviter)

---

## 1. Matrices et philosophie générale

- **Pourquoi des matrices ?** Une matrice est idéale pour stocker un jeu de données (ligne = individu, colonne = descripteur).
- **Les matrices sont typées** (`torch.int16`, `torch.float32`…). Le plus souvent on cherche à **gagner de la place**.
- **Philosophie** : des **opérateurs de haut niveau sur les matrices** plutôt que des boucles interprétées. Toutes les structures sont des objets `torch` → passage facile sur **GPU** et **calcul de gradient**.

> 📝 **Pourquoi PyTorch et pas numpy ?** Presque identique à numpy, mais (i) calcul sur **GPU** (numpy non), (ii) calcul de **gradient**, (iii) parallélisation des calculs matriciels. Les boucles Python « font ramer » ; on pense en **motifs** (opérations sur blocs).

> 📝 **Allocation mémoire** : passer de `torch.int64` (type entier par défaut) à `torch.int16` = **facteur 4** (64/16 bits). ✅ exact. Toujours garder en tête l'ordre de grandeur de la mémoire pour éviter les problèmes.

> 📝 « numpy.array = torch.tensor » : ⚠️ **à nuancer** — très proche dans l'usage, mais le `torch.Tensor` porte en plus un **device** (CPU/GPU) et un champ **gradient** ; les conversions de type ne sont **pas automatiques** et un tenseur ne se mélange pas avec une liste Python.

> 📝 Conseil d'environnement : créer un environnement par projet (`conda create --name nom_env python=3.12`). 🔎 La note contient une faute de frappe (`pyton =3.12`) : la syntaxe correcte est `python=3.12` (sans espace).

---

## 2. Créer et accéder aux valeurs

```python
import torch
cst   = torch.tensor([[3.1, 2.7, 1.9], [1.6, 0.0, 9.8], [1.7, 8.1, ...]])  # à partir de listes (ligne tronquée sur la diapo)
zeros = torch.zeros(4, 3)      # 4 lignes, 3 colonnes
ones  = torch.ones(4, 3)
x     = torch.rand(4, 3)       # tirage uniforme
zeros_like_x = torch.zeros_like(x)
```

**Accès (penser aux motifs plutôt qu'aux boucles)** :

```python
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
x[0][0]  ou  x[0, 0]     # chaque case est elle-même un tenseur
x[:, 0]                  # toutes les lignes, 1re colonne
x[0, [0, 2]]             # 1re ligne, colonnes 0 et 2
# syntaxe générale  début:fin(exclue):pas
x[0, 0:2] == x[0, :2]    # borne sup
x[0, 1:3] == x[0, 1:]    # borne inf
x[0, 0:3:2] == x[0, ::2] # pas de 2
```

---

## 3. Calcul entre matrices, in-place, agrégats

| Opération | Exemple | Résultat |
|---|---|---|
| Avec un scalaire | `x + 2` | ajout à chaque case : `[[4,6,8],[5,8,11]]` |
| Terme à terme | `z = x * y` | `[[8,24,48],[15,48,99]]` |
| Produit matriciel | `a = x @ torch.tensor([[1],[2],[3]])` | produit matrice × vecteur colonne |

**In-place (ou pas)** — selon les situations, pour gagner de la place :
- Opération classique + stockage dans un **nouveau** tenseur : `b = torch.sin(a)` ; `c = a + b`.
- Opération **in-place** (modifie l'argument, suffixe `_`) : `torch.sin_(a)` ; `a.add_(b)` → **modifie `a`**.

> 🔎 Dans le PDF, le caractère `_` n'apparaît pas à l'extraction (`a.add(b)`) : la version in-place est bien `add_`. Sans `_`, `add` renvoie un nouveau tenseur.

**Calculs agrégatifs** (`axis`/`dim` = 1 : on agrège les colonnes → un résultat **par ligne** ; 0 : un résultat **par colonne**) :
```python
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
li  = x.sum(1)            # ou x.sum(axis=1) : somme de chaque ligne
co  = x.sum(0)            # somme de chaque colonne
tot = x.sum()             # somme totale
li2 = x[:, ::2].sum(1)    # somme des colonnes 0 et 2
```
Autres agrégats : somme, moyenne, min, max, …

---

## 4. `squeeze` / `unsqueeze`

- Une ligne de données = tenseur 1-D ; une image = 2-D (ou 3-D avec canaux) ; un corpus d'images = 3-D (ou 4-D).
- Donner **une image** en entrée du système → `unsqueeze` : on crée une dimension homogène à un **batch** (batch de taille 1).
- Récupérer et analyser une **sortie intermédiaire** → `squeeze` : on retire la dimension batch/individu de taille 1.

```python
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)
a.shape   # torch.Size([3, 226, 226])
b.shape   # torch.Size([1, 3, 226, 226])
```

> 📝 `unsqueeze` « permet de créer un groupe » : **les réseaux de neurones attendent des groupes (batchs), même de taille 1**. `squeeze` annule l'action de `unsqueeze`. `view` = `reshape` (même nombre d'éléments, forme différente).

⚠️ `squeeze` ne retire que les dimensions **de taille 1** ; `view(-1)` laisse PyTorch **deviner** la dimension manquante.

---

## 5. Gradient et `backward`

La structure `tensor` intègre aussi un champ **`grad`** (gradient).

```python
a = torch.tensor(1.)
a.requires_grad = True                      # activation du champ gradient
b = torch.tensor(2., requires_grad=True)    # ou à la création
z = 2*a + b
z.backward()                                # dérivées partielles de z
print("Dérivée de z/a :", a.grad.item(), " z/b :", b.grad.item())   # 2.0 et 1.0
```

- $z$ est **connectée** à $a$ et $b$ (graphe de calcul). Si $z$ est une fonction de coût, **le gradient indique la direction pour la minimiser**.
- $\dfrac{\partial z}{\partial a}=2$ et $\dfrac{\partial z}{\partial b}=1$ pour $z=2a+b$.

> 📝 `backward()` calcule la dérivée partielle de la fonction par rapport aux variables ayant `requires_grad=True` ; le résultat de $z$ par rapport à $a$ se lit dans `a.grad.item()`. Les « variables de base » sont celles où l'on a activé le gradient ; la variable « fonction » **dépend** de ces variables. 🔎 Précision : `z` hérite automatiquement de `requires_grad=True` puisqu'elle dépend de feuilles qui l'ont ; mais seul le `.grad` des **feuilles** (`a`, `b`) est conservé.

### Gradient sur un vecteur / une matrice
- Les tenseurs sont en général des vecteurs, matrices… Le **gradient a la même forme que le tenseur d'origine**.
- Hypothèse : $L=\mathrm{fct}(A)$ est **un scalaire** ; `L.backward()` remplit `A.grad` :

$$A=\begin{pmatrix}a_{11}&a_{12}&\dots\\a_{21}&a_{22}&\dots\\\vdots&&\end{pmatrix},\qquad A.\mathrm{grad}=\begin{pmatrix}\frac{\partial L}{\partial a_{11}}&\frac{\partial L}{\partial a_{12}}&\dots\\\frac{\partial L}{\partial a_{21}}&\frac{\partial L}{\partial a_{22}}&\dots\\\vdots&&\end{pmatrix}$$

- $A$ : tenseur-paramètre ; $L$ : scalaire (coût) ; $a_{ij}$ : composante ; la case $(i,j)$ de `A.grad` est $\partial L/\partial a_{ij}$.

### Syntaxe une fois le gradient activé
```python
b = torch.tensor(2., requires_grad=True)
b.data          # valeur seule (ok)
b.grad.data     # valeur du gradient (ok)
```
> 📝 Si on active le gradient, la variable contient les données `b.data` et la valeur du gradient `b.grad.data`. 🔎 Le diaporama signale `print(b)` comme « ambigu » : il s'affiche en réalité `tensor(2., requires_grad=True)` (pas d'erreur) ; pour une valeur Python utiliser `.item()`.

**Mise à jour des paramètres plus efficace en coupant le graphe de calcul** :
```python
with torch.no_grad():
    w += -EPS * w.grad
    b += -EPS * b.grad
```
⚠️ In-place (`+=`) **obligatoire** ; `w = w - EPS*w.grad` crée un nouveau tenseur qui sort du graphe.

### Accumulation du gradient
Appels multiples à `backward` (à partir d'une ou plusieurs fonctions de coût) → **les gradients s'accumulent** :
```python
a = torch.tensor(1., requires_grad=True); b = torch.tensor(2., requires_grad=True)
z = 2*a + b; z.backward()
z = 2*a + b                      # il faut redéfinir le coût
z.backward()                     # puis relancer = accumulation (a.grad vaut maintenant 4)
```
🚩 D'où la nécessité de **remettre le gradient à 0** après chaque mise à jour.

---

## 6. Autograd et graphe de calcul

> 💡 **Graphe de calcul** = graphe orienté décrivant l'enchaînement des opérations. Sources = variables d'entrée ; nœud de sortie = résultat du calcul. En connaissant la **dérivée de chaque opération**, le graphe permet de calculer le gradient de la sortie par rapport à chaque variable d'entrée. Deux passes : **forward** (calcul de la sortie) puis **backward** (propagation des gradients).

---

## 7. Régression linéaire en PyTorch

**Verrous à lever** :
1. Lire les données → scikit-learn.
2. Construire l'estimateur linéaire.
3. Appliquer la fonction de coût.
4. Trouver un **critère d'arrêt** pour la descente de gradient.
5. Évaluer les performances.

**Accès aux données** (*California housing*) — récupérer, transformer en tenseur, **vérifier les dimensions** :
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
housing_x = torch.tensor(housing['data'],   dtype=torch.float)   # typer pour éliminer les incertitudes
housing_y = torch.tensor(housing['target'], dtype=torch.float)
```

**Estimateur linéaire** : déclarer les paramètres avec gradient puis construire la prédiction.
```python
w = torch.randn(1, housing_x.size(1), requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
yhat = (x @ w.T) + b          # $\hat y = x\,w^\top + b$
```
Puis la **fonction de coût** (moindres carrés) … « et c'est (presque) tout ».

**Autres verrous** :
- Pas de critère d'arrêt pour l'instant (5000 itérations ?).
- Calculer la performance.
- **Ne pas oublier de remettre le gradient à 0** après la mise à jour, pour éviter l'accumulation.

> 📝 « Définir des règles d'arrêt : trouver le bon epsilon ». 🔎 **Correction** : $\varepsilon$ (`EPS`) est le **pas d'apprentissage** (*learning rate*) de la mise à jour $w\leftarrow w-\varepsilon\nabla$, pas un critère d'arrêt. Le **critère d'arrêt** est distinct (nombre fixe d'itérations, seuil de variation de la loss, ou *early stopping* sur validation). Un mauvais $\varepsilon$ fait diverger (trop grand) ou stagner (trop petit) : c'est pourquoi on en cherche un bon.

> 📝 Étapes : importer les données en les passant en tenseur (**bien typer**) ; déclarer les paramètres $w$ et $b$ avec le gradient ; définir le coût.

---

## 8. Conclusion : ce qu'est PyTorch

**PyTorch, c'est…**
- un framework de développement et d'apprentissage de réseaux profonds sur **CPU et GPU** ;
- une **architecture modulaire** de contenants + conteneurs → architectures flexibles ;
- la **différenciation automatique** (autograd) ;
- une couche d'abstraction pour l'**optimisation** (variété de descentes de gradient) ;
- une gestion simplifiée des données pour la constitution des **mini-batchs**.

| | PyTorch | TensorFlow |
|---|---|---|
| Industrie | plus récent, moins intégré | déploiement, rapidité, processus industriel |
| Recherche | **flexibilité**, prototypage, simplicité | — |

Les deux frameworks ont tendance à se rapprocher en fonctionnalités.

---

## 9. Quiz d'auto-test

> Barème indicatif sur 20 : Q1–Q2 (3 pts chacune) · Q3–Q5 (2,5 pts) · Q6–Q9 (2 pts) · Q10 (1 pt).

<details>
<summary><b>Question 1 (3 pts) :</b> Que se passe-t-il si on appelle `backward()` deux fois de suite sur le même coût (redéfini) sans remise à zéro ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Les gradients **s'accumulent** : `.grad` est **incrémenté** (pas écrasé). Avec $z=2a+b$ appelé deux fois, `a.grad` passe de 2 à 4. Il faut donc remettre le gradient à zéro (`.grad.zero_()` ou `optim.zero_grad()`) après chaque mise à jour.
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Pourquoi met-on à jour les paramètres dans un bloc `with torch.no_grad():` avec `w += -EPS*w.grad` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Pour ne **pas enregistrer** la mise à jour dans le graphe de calcul (plus efficace) et conserver le **même tenseur-feuille** : l'opération in-place `+=` modifie `w` sans créer un nouveau tenseur. Avec `w = w - EPS*w.grad`, on crée un nouveau tenseur hors graphe, qui ne se « reconnecte » pas.
</details>

<details>
<summary><b>Question 3 (2,5 pts) :</b> Quelles sont les deux passes d'un graphe de calcul et que contient-il ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
**Forward** : calcul de la sortie à partir des entrées ; **backward** : calcul des gradients de la sortie par rapport à chaque variable d'entrée, grâce aux dérivées de chaque opération. Le graphe est orienté ; les sources sont les variables d'entrée, le nœud de sortie le résultat.
</details>

<details>
<summary><b>Question 4 (2,5 pts) :</b> Que fait `unsqueeze(0)` sur un tenseur de forme (3, 226, 226) et pourquoi ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Il ajoute une dimension de taille 1 en tête : forme (1, 3, 226, 226) = **batch de taille 1**. Les réseaux attendent des batchs, même d'un seul exemple. `squeeze` fait l'opération inverse.
</details>

<details>
<summary><b>Question 5 (2,5 pts) :</b> Quelle est la forme de `A.grad` si `A` est une matrice 2×3 et `L` un scalaire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Même forme que `A` : 2×3. Chaque case $(i,j)$ vaut $\partial L/\partial a_{ij}$. `L` doit être un scalaire pour appeler `L.backward()`.
</details>

<details>
<summary><b>Question 6 (2 pts) :</b> Différence entre `torch.sin(a)` et `torch.sin_(a)` / `a.add_(b)` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Sans `_` : l'opération **crée un nouveau tenseur** (stockage supplémentaire). Avec `_` : opération **in-place** qui modifie l'argument et économise de la mémoire.
</details>

<details>
<summary><b>Question 7 (2 pts) :</b> Quel est le rôle de $\varepsilon$ dans la régression linéaire par descente de gradient, et que n'est-il pas ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
C'est le **pas d'apprentissage** : $w\leftarrow w-\varepsilon\,\nabla_w L$. Ce n'est pas le critère d'arrêt (qui est le nombre d'itérations, un seuil sur la loss ou un *early stopping*).
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Passer de `torch.int64` à `torch.int16` : quel gain mémoire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Un facteur 4 (64 bits → 16 bits). D'où l'intérêt de typer explicitement les tenseurs.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Pourquoi `torch` plutôt que `numpy` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Calcul sur **GPU**, **calcul de gradient** automatique (autograd), opérateurs de haut niveau parallélisés (pas de boucles Python), et structures adaptées à la constitution de mini-batchs et aux modules d'un réseau.
</details>

<details>
<summary><b>Question 10 (1 pt) :</b> Que donne `x.sum(1)` pour `x = [[2,4,6],[3,6,9]]` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La somme de chaque ligne : `[12, 18]` (et `x.sum(0)` = `[5, 10, 15]`).
</details>

---

## 10. 🚩 Pièges à éviter

- **Gradient cumulé** : toujours remettre à zéro après la mise à jour.
- **`w = w - …`** au lieu d'une mise à jour in-place sous `no_grad`.
- **Types** : typer les données (`dtype=torch.float`) ; pas de conversion automatique ; liste Python × tenseur = erreur.
- **Formes** : `unsqueeze`/`squeeze`/`view(-1)` ; un `(n,1)` et un `(n,)` ne se mélangent pas (broadcasting silencieux).
- **`backward()`** exige un **scalaire** (ou un `gradient=` explicite).
- Confondre **pas d'apprentissage** ($\varepsilon$) et **critère d'arrêt**.
- Oublier que seules les **feuilles** avec `requires_grad` gardent un `.grad`.
