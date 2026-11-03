# 🧬 Fiche de cours — Apprentissage de représentation

> Source : `3-Repr.pdf` (diapos 1–24). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours) · 📝 mes notes (aucune fournie pour ce cours). Les dates et références historiques des diapos ont été omises volontairement.

## Sommaire
1. [Introduction : la chaîne de traitements et la promesse du deep learning](#1-introduction--la-chaîne-de-traitements-et-la-promesse-du-deep-learning)
2. [Frontière floue entre représentation et décision](#2-frontière-floue-entre-représentation-et-décision)
3. [Représenter des éléments discrets ou continus](#3-représenter-des-éléments-discrets-ou-continus)
4. [Representation learning et transfert](#4-representation-learning-et-transfert)
5. [Architectures d'apprentissage de représentation](#5-architectures-dapprentissage-de-représentation)
6. [Structurer l'espace appris (VAE) et explicabilité (Deep LVQ)](#6-structurer-lespace-appris-vae-et-explicabilité-deep-lvq)
7. [Applications au texte](#7-applications-au-texte)
8. [Recommandation](#8-recommandation)
9. [Sortir de la boîte noire](#9-sortir-de-la-boîte-noire)
10. [Visualisation, musique, décomposition de signaux](#10-visualisation-musique-décomposition-de-signaux)
11. [Outils pour le TP : `Embedding`, triplet loss](#11-outils-pour-le-tp--embedding-triplet-loss)
12. [Conclusion](#12-conclusion)
13. [Quiz d'auto-test](#13-quiz-dauto-test)
14. [🚩 Pièges à éviter](#14--pièges-à-éviter)

---

## 1. Introduction : la chaîne de traitements et la promesse du deep learning

**Chaîne de traitements classique** : jeu de données → **prétraitements** (normalisation) → **apprentissage** (avec supervision) → **classifieur** → **inférence** → **évaluation**. Exemple : classer des objets « verts » vs « rouges » ; ex. regarder les niveaux moyens et max de rouge par zone.

> 💡 **Facteur de performance : les features > le modèle.**

**Promesse du deep learning** (*representation learning*) :
- apprendre des **représentations continues de concepts discrets** (+ métrique continue) : sémantique des mots, profils utilisateurs ;
- apprendre des **représentations compactes de concepts continus** : représentation des images, extraction de motifs dans les signaux ;
- **optimiser les caractéristiques extraites pour une application cible**.

⚠️ Question posée : pourquoi cela marche-t-il si bien sur des données complexes/sémantiques mais moins bien sur des **données tabulaires** ? *(Étude citée : « Why do tree-based models still outperform deep learning on tabular data? » — les modèles à base d'arbres restent meilleurs sur le tabulaire.)*

---

## 2. Frontière floue entre représentation et décision

Exemple de la **détection d'objets** dans les images : la dernière couche de caractéristiques idéale = **1 si l'objet est présent / 0 s'il est absent**. Mais : *souhaite-t-on vraiment une représentation qui décide ?* (cf. ECOC, notion de distance…).

> 💡 Une bonne représentation doit garder la **notion de distance/similarité**, pas seulement coder la décision finale.

---

## 3. Représenter des éléments discrets ou continus

- En machine learning, il est **difficile de représenter des éléments discrets** : variables catégorielles, mots, utilisateurs.
- **Corpus en sac de mots** : chaque document = vecteur de comptes (ex. d1 = [1 0 0], d2 = [0 0 1], d3 = [0 1 0]). Dans cet espace one-hot, *voiture*, *automobile* et *chat* sont **à égale distance** (orthogonaux).
- **Objectif : introduire la notion de distance** (→ similarité ++, distance ++) dans un **espace vectoriel continu** où *voiture* ≈ *automobile* ≠ *chat*.

---

## 4. Representation learning et transfert

1. **Apprendre la représentation** sur **n'importe quel critère** (ex. modélisation du langage) ;
2. **Fine-tuner** = apprendre un peu **sur la tâche finale / sur peu de données**.

Schéma : *données brutes* (texte) → **étape d'apprentissage de représentation** (convolution, fully connected, …) → **représentation apprise** (embedding) → **étapes de décision** → tâches. Le **mécanisme d'apprentissage** (ex. language modeling) produit l'embedding.

---

## 5. Architectures d'apprentissage de représentation

- **Domaines** : recommandation, représentation de graphes, classification multi-labels multi-classes, séparation de sources audio, radar.
- Appris **end-to-end**, **pré-appris** et/ou **fine-tunés**.
- **Parfait quand la supervision est indirecte** : similarité par **triplet**, **contexte**, …
- Possibilité d'utiliser des **auto-encodeurs** : encodeur $X_n\to z_n\in\mathbb R^D$ puis décodeur $z\to\hat X_n$ (reconstruction).
- Exemple en recommandation : une **matrice de notes** ; en graphes : une **matrice d'affinité**.

---

## 6. Structurer l'espace appris (VAE) et explicabilité (Deep LVQ)

- **Variational Auto-Encoder (VAE)** : *structurer l'espace appris* via le **reparameterization trick**. 🔎 *Complément :* on échantillonne $z=\mu+\sigma\odot\varepsilon$ avec $\varepsilon\sim\mathcal N(0,I)$, de sorte que le gradient traverse l'échantillonnage ; le VAE impose un *a priori* sur la distribution latente.
- **Représentation & explicabilité** — hypothèse : *expliquer un processus = le décomposer sur un dictionnaire réduit et intelligible*. Chaîne : **Input** (discret/continu) → **représentation latente** (continue, avec distance) → **dictionnaire** (discret).
- **Deep LVQ** : LVQ = *Learning Vector Quantization* = idée du **plus proche voisin** (prototypes/supports) **+ deep learning** pour **apprendre la représentation des supports**.

---

## 7. Applications au texte

### 7.1 Des sacs de mots aux représentations vectorielles
**Comment apprendre un tel espace ?** Par **observations massives** (contextes de mots) : une succession de **petits mouvements** des vecteurs pour rapprocher les mots qui apparaissent dans des contextes similaires (night, stars, moon… d'un côté ; kitchen, sofa… de l'autre).

### Des mots aux tokens
**Décomposition en groupes de lettres fréquents** (ex. *Machine-Learning* → plusieurs *tokens*), chaque token ayant un vecteur dans l'espace continu. Avantages : **représentation des mots inconnus**, **résistance aux fautes d'orthographe**.

### 7.2 Agrégation des représentations de mots
Apprendre des représentations de **phrases** avec deux critères :
- **Critère 1 : régénérer la phrase** (*« Il tombe des cordes »* → représentation → décodeur qui reproduit la phrase) ;
- **Critère 2 : aligner les représentations de phrases** dans plusieurs langues (*« It's raining cats and dogs »* ≈ *« Es regnet Bindfäden »* ≈ *« Il tombe des cordes »*).

---

## 8. Recommandation

> **Apprendre des profils à partir de traces = factorisation matricielle** : *déjà un modèle de deep learning ?*

- Les **profils se ressemblent ⇒ prédisent les mêmes notes**. Matrice de notes $R$ ; profils utilisateurs $U$, profils items $I$.
- **Architecture** : un utilisateur $u$ et un item $i$ passent dans des **lookup tables** (embeddings) → **produit scalaire** ou **MLP non linéaire** → **prédiction** comparée à la **vérité terrain** par une **MSE**.
- **Variante riche** (*side information*) : texte, temps, image, musique, âge/CSP de l'utilisateur… ; critère libre (reconstruction, prédiction…) ; **un décodeur** pour produire des **explications** ; **imposer une similarité** entre représentations ; **représentation intermédiaire** ; encodage ++, décodage ++.

> 💡 **Richesse des profils = performances. Expliquer le contenu des profils = xAI** (IA explicable).

---

## 9. Sortir de la boîte noire

**Expliquer une décision** (*instance feature attribution*) : LIME, SHAP, …
**Espace latent = boîte noire… que faire ?**
1. **Structurer l'espace** : VAE, GAN (+ **prototypes**) ;
2. **Explorer l'espace** : **contrefactuels** ; **montrer et générer des exemples** ;
3. **Décoder l'espace** : approche **générative** ; **parler à l'utilisateur**.

---

## 10. Visualisation, musique, décomposition de signaux

- **Visualisation** : auto-encodeur avec un espace latent **2D** = visualisation de MNIST ; passage au **VAE** (toujours non supervisé).
- **MusiCNN** : convolution sur la **transformée temps-fréquence** ; apprentissage d'**embeddings de musique** pour la classification, la recommandation et la génération.
- **Décomposition de signaux (VQ-VAE-GAN)** : *reconstruire une entrée… en la rendant indiscernable de son originale* (discriminateur vrai/faux).

---

## 11. Outils pour le TP : `Embedding`, triplet loss

### Représentations en PyTorch
> **Représentation = vecteur + gradient activé.**
Hypothèse : en recommandation, on modélise les notes par **3 scalaires** (composante globale, utilisateur, item) :

$$r_{ui}=\mu+b_u+b_i$$

- $\mu$ : moyenne globale ; $b_u$ : biais de l'utilisateur $u$ ; $b_i$ : biais de l'item $i$.

```python
mu = torch.tensor([3.5], requires_grad=True)                       # activer le gradient pour pouvoir apprendre
bu = [torch.tensor([0.1], requires_grad=True) for _ in range(num_users)]
bi = [torch.tensor([0.1], requires_grad=True) for _ in range(num_items)]
```
Étapes : initialiser les représentations à apprendre → les utiliser pour la **prédiction** → ajouter une **fonction de coût** (MSE) → **rétropropager** l'erreur.

### Module `Embedding`
Le module existe dans PyTorch ; **idéal pour représenter des structures discrètes** :
```python
latent_size = 10; nb_users = 100
users = torch.nn.Embedding(nb_users, latent_size)   # init aléatoire
print("User 5:", users(torch.tensor(5)))            # ⚠️ appeler avec un TENSEUR, pas un int
```

### Triplet loss
> **Rapprocher ce qui se ressemble… (+ éloigner autre chose).** « À l'ancienne » : réseau **siamois** (*Siamese network*).

🔎 Pour une ancre $a$, un exemple positif $p$ et un négatif $n$ : $\mathcal L=\max\big(0,\ d(a,p)-d(a,n)+m\big)$ avec $d$ une distance et $m$ une marge. La perte est nulle si le négatif est déjà plus loin que le positif d'au moins $m$.

---

## 12. Conclusion

**Évolution du partage** : algorithmes et données → implémentations de classifieurs → **données massives + plateformes de ML** et **partage de modèles appris** (Torch, W2V…) → généralisation de GitHub, TensorFlow, PyTorch + **interfaces d'accès** (HuggingFace).

> 💡 **Le plus grand levier de performance** de la décennie ; très efficace sur les données **sémantiques disponibles massivement** ; modèles plus complexes **et** plus accessibles ; **texte et image** — mais pas encore sur la **classification de signaux**.

**Modèles pré-appris / multi-tâches / transfert / fine-tuning** — que faire maintenant ? **plus de données** (GPT-2, 3…), **plus de modalités** (*grounding*, modèles visuels-langage), **plus de transfert** (nouvelles tâches).

**Perspectives de recherche** : (1) plus de données/complexité (difficile de jouer contre les GAFAM) ; (2) **applications originales** / contraintes métiers → développement de modèles (jusqu'à de nouvelles architectures) ; (3) **niches à fort potentiel** : data-to-text, recommandation explicable ; (4) **propriétés théoriques** : optimisation, *prior* sur l'espace latent (Nesterov, VAE…).

**Niveaux d'usage des architectures deep** : **utilisateur** de modèles pré-entraînés · **affineur** sur de petits jeux de données (domaines applicatifs divers) · **entraîneur** sur données jouets (démonstration de concepts) ou massives.

---

## 13. Quiz d'auto-test

> Barème indicatif sur 25 : Q1–Q3 (4 pts) · Q4–Q6 (3 pts) · Q7–Q9 (2 pts) · Q10 (1 pt).

<details>
<summary><b>Question 1 (4 pts) :</b> Pourquoi le sac de mots one-hot est-il insuffisant et que cherche-t-on avec un embedding ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
En one-hot, tous les mots sont **orthogonaux, à égale distance** (voiture, automobile et chat identiques). On veut un **espace vectoriel continu** où la distance reflète la **similarité sémantique**.
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Décrire l'architecture neuronale de la recommandation par factorisation. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Deux **lookup tables** (embeddings utilisateurs $U$ et items $I$) ; on récupère $u$ et $i$, on les combine par **produit scalaire** (ou **MLP non linéaire**) pour prédire la note, comparée à la vérité terrain par une **MSE** ; rétropropagation sur les embeddings.
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Écrire le modèle à biais $r_{ui}=\mu+b_u+b_i$ et le mettre en œuvre en PyTorch. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\hat r_{ui}=\mu+b_u+b_i$$

$\mu$ : moyenne globale ; $b_u$ : biais utilisateur ; $b_i$ : biais item. En PyTorch : trois jeux de paramètres avec `requires_grad=True` (ou `nn.Embedding(n,1)`), prédiction = somme, coût MSE, `backward`, mise à jour.
</details>

<details>
<summary><b>Question 4 (3 pts) :</b> Quelle est la différence entre le critère 1 et le critère 2 pour apprendre une représentation de phrase ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Critère 1 : **régénérer la phrase** (auto-encodage). Critère 2 : **aligner** les représentations de phrases équivalentes entre langues (multilingue).
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Quels avantages apportent les tokens (sous-mots) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Représentation des **mots inconnus**, **résistance aux fautes d'orthographe**, adaptation à des domaines techniques, vocabulaire de taille contrôlée.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Que fait la triplet loss ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Rapproche l'ancre de l'exemple positif et l'éloigne d'un négatif :

$$\mathcal L=\max\big(0,\ d(a,p)-d(a,n)+m\big)$$

($m$ : marge.) Alternative aux réseaux siamois.
</details>

<details>
<summary><b>Question 7 (2 pts) :</b> Quelles stratégies pour « sortir de la boîte noire » de l'espace latent ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Structurer (VAE, GAN, prototypes), explorer (contrefactuels, montrer/générer des exemples), décoder (approche générative, parler à l'utilisateur). Pour une décision individuelle : LIME, SHAP.
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Pourquoi appeler `users(torch.tensor(5))` et non `users(5)` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
`nn.Embedding` attend un **tenseur d'indices entiers**, pas un entier Python.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Pourquoi le deep learning fait-il moins bien sur le tabulaire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La diapo pose la question (étude : les modèles à arbres restent meilleurs sur le tabulaire) : l'avantage du deep learning se concentre sur les données **complexes et sémantiques** (texte, image, signaux) où apprendre une représentation apporte beaucoup.
</details>

<details>
<summary><b>Question 10 (1 pt) :</b> Les trois niveaux d'usage des architectures deep ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Utilisateur de modèles pré-entraînés ; affineur (fine-tuning) sur de petits jeux de données ; entraîneur (données jouets ou massives).
</details>

---

## 14. 🚩 Pièges à éviter

- **`Embedding` appelé avec un `int`** (il faut un tenseur d'entiers).
- **Représentation qui « décide »** : conserver une vraie notion de **distance**.
- **Deep sur tabulaire** : pas toujours supérieur aux arbres.
- **Fine-tuning** : adapter un peu, sur peu de données.
- **Boîte noire** : prévoir de l'explicabilité (profils, contrefactuels, LIME/SHAP).
