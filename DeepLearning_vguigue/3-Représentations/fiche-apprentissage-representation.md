# 📘 Fiche de révision — Apprentissage de représentation

> Cours de Vincent Guigue (inspiré des supports de N. Baskiotis & B. Piwowarski) · AgroParisTech – IODAA · 29 diapos (`Représentation.pdf`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 notes perso

---

## Sommaire

1. [Introduction : pourquoi apprendre des représentations](#1-introduction--pourquoi-apprendre-des-représentations)
2. [Architectures d'apprentissage de représentation](#2-architectures-dapprentissage-de-représentation)
3. [Texte : Word2Vec et au-delà](#3-texte--word2vec-et-au-delà)
4. [Recommandation](#4-recommandation)
5. [Embeddings multi-modaux : CLIP](#5-embeddings-multi-modaux--clip)
6. [Sortir de la boîte noire, visualisation, signaux](#6-sortir-de-la-boîte-noire-visualisation-signaux)
7. [Outils connexes pour le TP](#7-outils-connexes-pour-le-tp)
8. [Conclusion](#8-conclusion)
9. [Pièges à éviter](#9-pièges-à-éviter)
10. [Quiz auto-test](#10-quiz-auto-test)

---

## 1. Introduction : pourquoi apprendre des représentations

### 1.1 Chaîne de traitements (diapo 2)

Exemple : classer des pommes (verte / rouge). Jeu de données + **supervision** → **pré-traitements** (un expert décide de « regarder les niveaux moyens et max de rouge par zone », puis **normalisation**) → **apprentissage** → **classifieur** → **évaluation**. En **inférence**, une nouvelle pomme passe par les **mêmes pré-traitements** puis le classifieur.

> 💡 **Facteur de performances : *features* > modèle.** La qualité des caractéristiques compte plus que le choix du classifieur.

### 1.2 Promesse du deep learning (diapo 3)

- **Apprendre des représentations continues de concepts discrets** (+ métrique continue) : sémantique des **mots**, profils **utilisateurs**.
- **Apprendre des représentations compactes de concepts continus** : représentation des images, extraction de motifs dans les signaux.
- (Conférence dédiée : *International Conference on Learning Representations*, ICLR.)

> 💡 **Optimiser les caractéristiques extraites pour une application cible** : ~~Deep learning~~ ⇒ ***Representation Learning***.
>
> ⚠️ Pourquoi est-ce que ça marche si bien sur des données **complexes + sémantiques** mais **moins bien sur des données tabulaires** ? (Grinsztajn et al., NeurIPS 22 : *Why do tree-based models still outperform deep learning on tabular data?*) → sur du tabulaire, les forêts / le boosting restent souvent meilleurs.

### 1.3 Une frontière floue entre représentation et décision (diapo 4)

Exemple de la détection d'objets (CNN : *feature learning* puis *classification*).
- Dernière couche de caractéristiques **idéale** = **1 / objet présent + 0 / objet absent**.
- Souhaite-t-on vraiment une représentation **qui décide** ? → ECOC, **notion de distance**…

> 🔎 Une représentation trop « décisionnelle » (un one-hot) perd l'information de similarité entre classes ; une bonne représentation garde des **distances** porteuses de sens (ECOC = *Error-Correcting Output Codes*, codage des classes par des vecteurs avec de la redondance).

### 1.4 Représenter des éléments discrets ou continus ? (diapo 5)

**Machine learning = difficile de représenter des éléments discrets** : variables catégorielles, mots, utilisateurs.

- **Corpus en sac de mots** : chaque document = vecteur creux sur le vocabulaire (mot 1, …, voiture, …, automobile, …, chat, …, mot D). « voiture », « automobile » et « chat » sont des dimensions **orthogonales** : d1 (voiture), d2 (automobile) et d3 (chat) sont **à mêmes distances** les uns des autres !
- **Espace vectoriel continu** : « voiture » et « automobile » proches (**similarité ++**), « chat » loin (**distance ++**).

> 💡 **Objectif : introduire la notion de distance ≠ orthogonalité entre les catégories.**

### 1.5 *Representation learning* & transfert (diapo 6)

1. **Apprendre la représentation** sur n'importe quel critère ;
2. **fine-tuner = apprendre un peu** sur la tâche finale / sur peu de données, etc.

Schéma : *raw textual data* → étape d'apprentissage de représentation (convolution, fully connected…) → **représentation apprise = *embedding*** → étape(s) de décision (*language modeling*…) → tâches ; le **mécanisme d'apprentissage** remonte de la tâche jusqu'à la représentation.

---

## 2. Architectures d'apprentissage de représentation

### 2.1 Apprentissage de représentation (diapo 7)

- **Domaines** : recommandation, représentation de graphes, classification multi-labels multi-classes, séparation de sources audio, radar.
- Appris en **end-to-end**, **pré-appris** et/ou **fine-tuné**.
- **Parfait pour la supervision indirecte** : similarité par triplet, contexte, …
- Possibilité d'utiliser des **auto-encodeurs**.

**Factorisation matricielle** : une matrice $X$ ($n \times d$) est approchée par le produit de $C$ ($n \times z$) et $D$ ($z \times d$) : chaque **ligne** et chaque **colonne** reçoit un vecteur de dimension $z$, et la case $X_{ij}$ ≈ produit scalaire de la ligne $i$ de $C$ et de la colonne $j$ de $D$.

$$X \approx C\,D, \qquad X_{ij} \approx \langle C_i, D_{\cdot j}\rangle$$

- $n$ : nombre de lignes (ex. films), $d$ : nombre de colonnes (ex. utilisateurs), $z \ll n, d$ : dimension latente.
- Version recommandation : **matrice de notes** (films × utilisateurs) ; la case = **affinité**.

**Neural Collaborative Filtering** : entrées one-hot de l'utilisateur $s^{user}_u$ et de l'item $s^{item}_i$ → *embedding layer* (**user latent vector** $U$, **item latent vector** $V$) → couches neuronales (Layer 1, 2, …, X) → score $\hat r_{ui}$, comparé à $r_{ui}$ (training).

**Auto-encodeur** : image 28×28 (784 neurones) → encodeur (500, 300 neurones) → **latent de dimension 2** → décodeur (300, 500) → image reconstruite (784).

### 2.2 Structurer l'espace appris (diapo 8)

- **Variational AutoEncoder (VAE)** : l'encodeur ne produit pas un point mais une **distribution** (moyennes $\mu_1, \mu_2$ et variances $\sigma_1, \sigma_2$, « *latent state distributions* ») ; on **échantillonne** $z_1, z_2$ (« *sample from distributions* ») puis le décodeur reconstruit.
- **Reparameterization trick**.

> 🔎 *Reparameterization trick* : on ne peut pas rétro-propager à travers un tirage aléatoire ; on écrit donc $z = \mu + \sigma \odot \varepsilon$ avec $\varepsilon \sim \mathcal N(0, I)$ : l'aléa est dans $\varepsilon$ et le gradient passe par $\mu$ et $\sigma$. La loss du VAE = reconstruction + $\mathrm{KL}\big(\mathcal N(\mu, \sigma^2)\,\|\,\mathcal N(0, I)\big)$, qui **régularise** l'espace latent (continu, sans trous).

### 2.3 Représentation & explicabilité (diapo 9)

**Hypothèse : expliquer un processus = décomposer sur un dictionnaire *réduit* & *intelligible*.**

| Input | → | Représentation latente | → | Dictionnaire |
|---|---|---|---|---|
| discret / continu | | continu ⇒ **distance** | | **discret** |

Schéma : encodeur → espace latent ($z_1, z_2$) → *embedding space* composé d'éléments $e_1, e_2, \dots, e_k$ (dictionnaire) → décodeur.

### 2.4 Deep LVQ (diapo 10)

- **LVQ = Learning Vector Quantization.**
- Idée ∼ **plus proche voisin** : quelques **prototypes** (*codebooks*) découpent l'espace en **cellules de Voronoï** ; chaque point est représenté par son prototype le plus proche.
- **+ deep learning = apprendre la représentation des supports** (prototypes et espace appris ensemble).

---

## 3. Texte : Word2Vec et au-delà

### 3.1 Des sacs de mots aux représentations vectorielles (diapo 11)

Même idée qu'au §1.4 : passer d'une représentation où tous les mots sont équidistants à un **espace vectoriel continu** où la similarité sémantique se traduit par la proximité.

### 3.2 Word2Vec : extraire une sémantique locale (diapo 12)

Deux architectures (fenêtre glissante de taille 5 sur « the cat **sat** on the mat ») :

| Modèle | Entrée | Sortie |
|---|---|---|
| **Skip-gram** | le mot central $w(t)$ | prédire les mots du contexte $w(t-2), w(t-1), w(t+1), w(t+2)$ |
| **CBOW** | la **somme** des mots du contexte | prédire le mot central $w(t)$ |

**Skip-Gram** (& implémentation par ***negative sampling***) : *easy learning*.
- **Analyse locale** ;
- critère **prédictif** : estimer une valeur manquante ;
- **fenêtre glissante = contexte local $C$ du mot $w$**.

Illustration (corpus de phrases contenant « moon ») : (1) **initialisation aléatoire** des positions des mots ; (2) **mouvement dans l'espace** : les mots qui apparaissent dans le contexte de « moon » (night, stars, shining, cold…) s'en rapprochent.

### 3.3 W2V : explication détaillée (diapo 13)

Étant donné un mot $w$ et ses contextes locaux $C$ :

$$\text{Idée SG : } \arg\max_\theta \prod_C \prod_{w \in C} p(C \mid w; \theta)$$

- $p(D = 1 \mid w_i, w_j; \theta)$ ⇒ probabilité que $w_i$ et $w_j$ **apparaissent dans le même contexte**.

$$\arg\max_\theta \prod_{i,j \in C} p(D = 1 \mid w_i, w_j; \theta) \;+\; \underbrace{\prod_{i,j \in \bar C} p(D = 0 \mid w_i, w_j; \theta)}_{\text{Negative Sampling}}$$

(Goldberg & Levy 2014, *word2vec explained* ; Hammer 2002, *Generalized Relevance LVQ*.)

> 🔎 Le « + » entre les deux produits est une écriture schématique : il s'agit en fait d'un **produit** des deux termes (ou d'une **somme** une fois passé au log, cf. §3.4).

### 3.4 W2V : skip-gram & negative sampling (diapo 14)

- **Fonction logistique** :

$$p(D = 1 \mid w_i, w_j) = \frac{1}{1 + \exp(-\mathbf z_i \cdot \mathbf z_j)} = \sigma(\mathbf z_i\cdot\mathbf z_j)$$

- **Log-vraisemblance globale** :

$$\arg\max_Z \left(\sum_{i,j \in C}\log\sigma(\mathbf z_i\cdot\mathbf z_j) + \sum_{i,j \in \bar C}\log\sigma(-\mathbf z_i\cdot\mathbf z_j)\right)$$

- $\sigma$ : fonction sigmoïde ; $C$ : ensemble des **cooccurrences** (paires vues ensemble dans une fenêtre) ; $\bar C$ : ensemble des **non-cooccurrences** (paires tirées au hasard = négatifs) ;
- $\mathbf z_i$ : embedding du mot $w_i$ ; $Z$ : matrice de tous les embeddings ;
- 🔎 $\sigma(-x) = 1 - \sigma(x)$ : le second terme pousse $p(D=1)$ vers 0 pour les paires négatives.

- **Descente de gradient stochastique + triplet loss** (un mot, un contexte positif, un contexte négatif).
- **Frequent word subsampling trick** : on **supprime** (pas « on garde ») le mot $w_i$ avec la probabilité

$$p(w_i) = 1 - \sqrt{\frac{t}{freq(w_i)}}, \qquad t = 10^{-5}$$

- $freq(w_i)$ : fréquence du mot dans le corpus ; plus un mot est fréquent (« the », « of »), plus il est souvent écarté de l'apprentissage.

> 🔎 Sur la diapo : « *picking words with* $p(w_i)$ » ; dans l'article de Mikolov, $p(w_i)$ est la probabilité de **rejeter** le mot.

### 3.5 Des mots aux *tokens* (diapo 15)

**Décomposition en groupes de lettres fréquents** : « Machine-Learning » → *tokens* `Machine`, `Learn`, `ing`, chacun placé dans un **espace vectoriel continu** puis combinés.
- **Représentation des mots inconnus** (on les recompose à partir de sous-mots connus) ;
- **résistance aux fautes d'orthographe**.

### 3.6 Agrégation des représentations de mots (diapo 16)

On agrège les vecteurs de mots d'une phrase en **une représentation de phrase** :
- **Critère 1 : regenerating the sentence** (la représentation doit permettre de reconstruire la phrase, ex. « Il tombe des cordes ») ;
- **Critère 2 : aligning sentence representations** (deux phrases de même sens, ex. « Il tombe des cordes » / « It's raining cats and dogs » / « Es regnet Bindfäden », doivent avoir des représentations proches) — en ajoutant des langues, le critère 1 s'applique à chacune.

> 💡 Les expressions idiomatiques montrent qu'on ne peut pas traduire mot à mot : c'est la **phrase entière** qui porte le sens.

---

## 4. Recommandation

### 4.1 Factorisation matricielle (diapo 17)

- **2001-2010 : apprendre des profils à partir de traces = factorisation matricielle.**
- **Déjà un modèle de deep learning ?** Oui : des **lookup tables** $U$ (utilisateurs) et $I$ (items) → produit scalaire → **prédiction**, comparée à la *ground truth* par une **MSE**. Variante : concaténation des deux profils → **MLP non linéaire** → prédiction.

> 💡 **Profils se ressemblent ⇒ prédisent les mêmes notes.**

> ⚠️ **ATTENTION** : réfléchir à la manière dont sont stockées ces données **parcimonieuses / *sparses*** (on ne stocke que les notes observées, en triplets (user, item, note), jamais la matrice pleine).
>
> ⚠️ **ATTENTION (2)** : combien de paramètres, combien de notes à prédire ?

🔎 Ordres de grandeur : $n_U$ utilisateurs et $n_I$ items avec $z$ dimensions → $(n_U + n_I)\times z$ paramètres, contre $n_U \times n_I$ cases à prédire (dont une infime fraction observée). Ex. MovieLens 100k : 943 utilisateurs × 1682 films = 1,6 M cases, 100 000 notes observées (6 %) ; avec $z = 25$ : ~66 000 paramètres.

### 4.2 Cas des systèmes de recommandation (diapo 18)

Architecture complète : lookup tables $U$ et $I$ → représentation intermédiaire → prédiction (MSE avec la vérité terrain), **plus** :
- des **side information** (texte, temps, image, musique, âge/CSP de l'utilisateur…) encodées par un réseau et **alignées** avec les profils (*enforce similarity*), avec un critère quelconque (reconstruction, prédiction…) ;
- un **décodeur** qui part des profils pour produire des **explications**.

> 💡 - **Richesse des profils = performances** [encodage ++]
> - **Expliquer le contenu des profils = xAI** [décodage ++]

---

## 5. Embeddings multi-modaux : CLIP

### 5.1 *Contrastive Language-Image Pre-training* (diapo 19)

Un **encodeur de texte** produit $T_1, \dots, T_N$ et un **encodeur d'images** produit $I_1, \dots, I_N$ pour un batch de $N$ paires (image, légende). On calcule la matrice des produits scalaires $I_i \cdot T_j$ : la **diagonale** (bonnes paires) doit être grande, le reste petit.

| Élément | Choix |
|---|---|
| Image encoder | ResNet-50, 101, ViT, … |
| Text encoder | DistilBERT, … |
| Données | **400 M paires images + textes** |
| Apprentissage | **contrastive learning** |

- Dans l'espace commun, « Cat building LEGO. » est **proche** (*close*) de sa photo et **loin** (*far*) de celle de « Cat and dog cuddling. ».
- Extensions : *contrastive fine-tuning* (mixup géodésique entre texte et image, *hard negatives*) pour obtenir des embeddings alignés et **uniformément répartis** sur la sphère.

> 🔎 Loss contrastive (InfoNCE, symétrique), hypothèses : embeddings normalisés, température $\tau$ :
> $$\mathcal L = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(I_i\cdot T_i/\tau)}{\sum_{j=1}^{N}\exp(I_i\cdot T_j/\tau)}\quad(+\text{ même terme en échangeant } I \text{ et } T)$$
> = une cross-entropie où la « bonne classe » de l'image $i$ est la légende $i$.

### 5.2 Exemple d'utilisation : cosine score entre $Enc(Im)\cdot Enc(Txt)$ (diapo 19)

**Classification *zero-shot*** : on encode des phrases « *a photo of a {classe}, a type of food* » pour chaque classe et on prend la plus proche de l'image (ex. Food101 : guacamole, rang 1/101, probabilité 90 % ; SUN397 : *television studio* ; FER2013 : émotion « angry » mal classée (rang 5/7) ; PatchCamelyon : tissu de ganglion).

**Robustesse** : à précision égale sur ImageNet (76,2 %), le CLIP *zero-shot* résiste beaucoup mieux aux changements de distribution qu'un ResNet-101 entraîné sur ImageNet : ImageNetV2 70,1 vs 64,3 ; ImageNet-R 88,9 vs 37,7 ; ObjectNet 72,3 vs 32,6 ; ImageNet Sketch 60,2 vs 25,2 ; ImageNet-A 77,1 vs 2,7.

> 💡 Un espace partagé texte/image permet de classer **sans apprentissage spécifique** (zero-shot), en décrivant les classes en langage naturel.

---

## 6. Sortir de la boîte noire, visualisation, signaux

### 6.1 Sortir de la boîte noire (diapo 20)

- **Expliquer une décision** (*Instance Feature Attribution*) : LIME, SHAP, …
- **Espace latent = boîte noire…** Que faire ?

| Action | Outil | ⇒ |
|---|---|---|
| 1. **Structurer** l'espace | VAE, GAN | + **prototypes** |
| 2. **Explorer** l'espace | **contrefactuel** | **montrer & générer des exemples** |
| 3. **Décoder** l'espace | approche **générative** | **parler à l'utilisateur** |

- Exemple visuel : espace latent de MNIST (clusters par chiffre) et grille de chiffres générés en parcourant l'espace (un chiffre se transforme progressivement en un autre).
- **Factual / semi-factual / counterfactual** : pour une image de test (un « 9 »), un **factuel** est un exemple similaire de même classe ; un **semi-factuel** est modifié mais reste du même côté de la **frontière de décision** (le « 9 » se rapproche d'un « 4 » mais reste classé 9) ; un **contrefactuel** franchit la frontière (devient un « 4 ») : la **plus petite modification** qui change la décision.

### 6.2 Visualisation de données (diapo 21)

- **Auto-encodeur ⇒ 2D = visualisation de MNIST** (les 10 classes se chevauchent largement) ;
- **passage au VAE** (toujours **non supervisé**) : les classes forment des groupes plus nets et l'espace est mieux occupé.

### 6.3 MusiCNN (diapo 22)

- **Convolution sur la transformée temps-fréquence** (spectrogramme) ;
- **apprentissage d'embedding de musique** : pour la classification / recommandation ; pour la génération.

### 6.4 Décomposition de signaux (diapo 23)

**VQ-VAE-GAN : reconstruire une entrée…**
- à partir d'un **ensemble de motifs finis** (*memory bank* : dictionnaire discret, cf. §2.3) ;
- en la rendant **indiscernable de son originale** ! (un **discriminateur** juge réel/faux, comme dans un GAN).

Exemple (MATS) : *Stage I* = encodeur → mémoire de motifs → décodeur + discriminateur ; *Stage II* = un **prédicteur** travaille sur les attentions mémoire pour prédire la suite (séries temporelles).

---

## 7. Outils connexes pour le TP

### 7.1 Représentations en PyTorch (diapo 24)

**Représentation = vecteur + gradient activé.**

Hypothèse : dans un problème de recommandation, je modélise les notes par **3 scalaires** : composante **globale**, composante **utilisateur**, composante **item** :

$$r_{ui} = \mu + b_u + b_i$$

- $\mu$ : moyenne globale ; $b_u$ : biais de l'utilisateur $u$ (note-t-il généreusement ?) ; $b_i$ : biais de l'item $i$ (est-il apprécié en général ?).

```python
mu = torch.tensor([3.5], requires_grad=True)   # activate gradient to be able to learn
bu = [torch.tensor([0.1], requires_grad=True) for _ in range(num_users)]
bi = [torch.tensor([0.1], requires_grad=True) for _ in range(num_items)]
```

- Utiliser ces grandeurs pour la **prédiction** ; ajouter une **fonction de coût** (ex. MSE) ; **rétro-propager** l'erreur.

### 7.2 Module de représentation (diapo 25)

Le module existe en PyTorch, il est **idéal pour représenter les structures discrètes** :

```python
latent_size = 10
nb_users = 100
users = torch.nn.Embedding(nb_users, latent_size)   # random init
# get representation of user 5:
print("User 5:", users(torch.tensor(5)))
# WARNING: call for a tensor (not an int)
```

> 💡 `nn.Embedding` = une **lookup table** : une matrice $(n, z)$ dont la ligne $k$ est le vecteur appris de l'objet d'indice $k$ (équivalent à multiplier un one-hot par la matrice, sans le calculer).

### 7.3 Triplet loss / contrastive learning (diapo 26)

- **Rapprocher ce qui se ressemble… (+ éloigner autre chose).**
- **À l'ancienne : Siamese network** : deux images passent dans **le même ConvNet (poids partagés)** → encodages $h(\text{image}_1)$, $h(\text{image}_2)$ → distance euclidienne → sigmoïde → similarité (ex. 0,98 pour deux « 0 »).
- **Triplet** : *anchor* (Obama), *positive* (Obama, autre photo), *negative* (Macron) → trois CNN **à poids partagés** → embeddings → **triplet loss**.

> 🔎 Triplet loss, hypothèses : $f$ l'encodeur, $(a, p, n)$ ancre/positif/négatif, marge $\alpha > 0$ :
> $$\mathcal L(a, p, n) = \max\big(0,\; \lVert f(a) - f(p)\rVert - \lVert f(a) - f(n)\rVert + \alpha\big)$$
> - nulle dès que le négatif est plus loin que le positif d'au moins $\alpha$ ; sinon elle rapproche $p$ de $a$ et éloigne $n$.

---

## 8. Conclusion

### 8.1 Évolution scientifique & pratique (diapo 27)

| Période | Partage |
|---|---|
| 80-90' | partage d'**algorithmes** et de **données** |
| 2000' | partage d'**implémentations** de classifieurs (SVM light, liblinear) |
| 2010' | **données massives** + plateformes de ML & **partage de modèles appris** (Torch, W2V…) |
| 2020' | généralisation de github, tensorflow, pytorch + **interfaces d'accès** (ex. HuggingFace) |

- **Le plus grand levier de performances** sur les 10 dernières années ;
- **très efficace sur les données sémantiques disponibles massivement** ;
- **des modèles + complexes ET + accessibles**.

> ⚠️ En texte, en image… **mais pas encore sur la classification de signaux.**

### 8.2 Modèles pré-appris / multi-tâches / transfert / fine-tuning (diapo 28)

**Que faire maintenant ?**
- **Plus de données** (GPT-2, 3, etc.) ;
- **plus de modalités** (*grounding*, *visual language model*) ;
- **plus de transfert** (nouvelles tâches).

Exemples : **instruction finetuning** (« Please answer the following question… »), **chain-of-thought finetuning** (raisonner étape par étape), **multi-task instruction finetuning** (1,8 k tâches) ⇒ **généralisation à des tâches non vues**. Modèles **vision-langage** à *prompt* mixte image + texte (ex. compléter « This is a flamingo. They are found in… », lire un panneau, résoudre « 3×6 » écrit à la main).

### 8.3 Quelles perspectives pour notre recherche ? (diapo 29)

1. **+ de données, + de complexité** : difficile de jouer contre les GAFAM ;
2. **applications originales / implémentation de contraintes métiers** (développement de modèles : +/- de développement selon les cas d'usage, jusqu'à des architectures nouvelles : CNN, LSTM, Transformer, ReLU…) ;
3. **recherche de niches à fort potentiel** : nouvelles problématiques (data-to-text, recommandation explicable) ;
4. **recherche de propriétés théoriques** : optimisation, prior sur l'espace latent (Nesterov, VAE…) ; complémentarité intéressante mais complexe entre théorie et pratique.

> 💡 **Niveaux d'usage des architectures deep :**
> - **Utilisateur** de modèles pré-entraînés ;
> - **Affineur** sur de petits jeux de données, domaines applicatifs divers ;
> - **Entraîneur** sur des données jouets (démonstration de concepts) / données massives ;
> - **Développeur** pour jouer sur la plasticité.

---

## 9. Pièges à éviter

> 🚩 **One-hot / sac de mots = tous les objets équidistants** : aucune notion de similarité. C'est précisément ce que les embeddings corrigent.

> 🚩 **Stocker la matrice de notes pleine** : elle est très creuse ; travailler en triplets (user, item, note) et ne prédire que les cases observées pour l'apprentissage.

> 🚩 **Sous-estimer le nombre de paramètres** : $(n_U + n_I)\times z$ ; trop de dimensions latentes pour trop peu de notes → sur-apprentissage (d'où la régularisation L2 sur les profils).

> ⚠️ **Negative sampling** : sans exemples négatifs, la solution triviale est de rapprocher **tous** les mots (tous les produits scalaires grands).

> ⚠️ **Subsampling** : $p(w_i) = 1 - \sqrt{t/freq(w_i)}$ est la probabilité d'**écarter** un mot fréquent.

> ⚠️ **Le deep n'est pas toujours le meilleur** : sur données tabulaires, les modèles à base d'arbres l'emportent souvent.

> ⚠️ **Représentation qui « décide »** (one-hot de la dernière couche) : elle perd les distances utiles au transfert.

> ⚠️ **Triplet loss** : les poids de l'encodeur sont **partagés** entre ancre, positif et négatif ; la marge empêche l'effondrement de tous les embeddings en un point.

> ⚠️ **`nn.Embedding` s'appelle avec un tenseur d'indices** (`long`), pas un entier Python.

---

## 10. Quiz auto-test

<details>
<summary><b>Question 1 (3 pts) :</b> Pourquoi le sac de mots / le one-hot pose-t-il problème, et quel est l'objectif de l'apprentissage de représentation ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

En one-hot, chaque catégorie est une dimension orthogonale : « voiture », « automobile » et « chat » sont tous à la même distance. Objectif : apprendre un **espace vectoriel continu** où la distance a un sens (similarité ++ pour voiture/automobile, distance ++ pour chat) : distance ≠ orthogonalité.
</details>

<details>
<summary><b>Question 2 (2 pts) :</b> Que signifie « features > modèle » ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le principal facteur de performance est la qualité des caractéristiques extraites, plus que le classifieur. Le deep learning apprend ces caractéristiques en les optimisant pour la tâche cible (*representation learning*).
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Écrire la log-vraisemblance du skip-gram avec negative sampling et définir chaque terme. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\arg\max_Z\left(\sum_{i,j\in C}\log\sigma(\mathbf z_i\cdot\mathbf z_j) + \sum_{i,j\in\bar C}\log\sigma(-\mathbf z_i\cdot\mathbf z_j)\right)$$

$\sigma$ : sigmoïde, $p(D=1\mid w_i,w_j) = \sigma(\mathbf z_i\cdot\mathbf z_j)$ ; $C$ : paires cooccurrentes (même fenêtre) ; $\bar C$ : paires non cooccurrentes tirées au hasard (négatifs) ; $\mathbf z_i$ : embedding du mot $i$. On rapproche les mots vus ensemble et on éloigne les paires aléatoires.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> Différence entre skip-gram et CBOW ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Skip-gram : à partir du mot central, prédire les mots du contexte. CBOW : à partir de la somme des mots du contexte, prédire le mot central. Les deux exploitent une fenêtre glissante (contexte local).
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> À quoi sert le subsampling des mots fréquents ? Donner la formule. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

À écarter souvent les mots très fréquents (« the », « of ») qui apportent peu d'information et dominent l'apprentissage : on rejette $w_i$ avec la probabilité $p(w_i) = 1 - \sqrt{t/freq(w_i)}$, $t = 10^{-5}$.
</details>

<details>
<summary><b>Question 6 (2 pts) :</b> Pourquoi travailler avec des tokens (sous-mots) plutôt que des mots ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Décomposer en groupes de lettres fréquents permet de représenter des **mots inconnus** (recomposés à partir de sous-mots) et donne une **résistance aux fautes d'orthographe**.
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Écrire le modèle de base à 3 scalaires en recommandation, puis la factorisation matricielle. Que représentent les termes ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$r_{ui} = \mu + b_u + b_i \qquad\text{puis}\qquad \hat r_{ui} = \mu + \langle U_u, I_i\rangle$$

$\mu$ : moyenne globale, $b_u$ / $b_i$ : biais utilisateur / item ; $U_u$, $I_i$ : profils (embeddings) de dimension $z$. Des profils proches prédisent des notes proches. Les paramètres sont appris par descente de gradient sur la MSE (+ régularisation L2).
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Quelles sont les deux mises en garde du cours sur les données de recommandation ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Les données sont **parcimonieuses (sparse)** : réfléchir à leur stockage (triplets, pas de matrice pleine). 2. **Combien de paramètres, combien de notes à prédire ?** $(n_U+n_I)\times z$ paramètres pour $n_U\times n_I$ cases dont très peu sont observées.
</details>

<details>
<summary><b>Question 9 (3 pts) :</b> Comment fonctionne CLIP et comment l'utiliser en zero-shot ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Un encodeur d'images (ResNet, ViT) et un encodeur de texte (DistilBERT…) sont entraînés par **apprentissage contrastif** sur 400 M paires image+texte : les bonnes paires (diagonale de la matrice $I_i\cdot T_j$) doivent avoir un grand produit scalaire, les autres un petit. Zero-shot : encoder « a photo of a {classe} » pour chaque classe et choisir celle de plus grand score cosinus avec l'image. Il est beaucoup plus robuste aux changements de distribution.
</details>

<details>
<summary><b>Question 10 (3 pts) :</b> Écrire la triplet loss. Quel est le rôle des poids partagés et de la marge ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\mathcal L(a,p,n) = \max\big(0,\ \lVert f(a)-f(p)\rVert - \lVert f(a)-f(n)\rVert + \alpha\big)$$

Le même encodeur $f$ (poids partagés) projette ancre, positif et négatif dans le même espace. La marge $\alpha$ impose que le négatif soit plus loin que le positif d'au moins $\alpha$ et empêche la solution triviale (tout au même point).
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Qu'apporte un VAE par rapport à un auto-encodeur, et qu'est-ce que le reparameterization trick ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

L'encodeur produit une distribution ($\mu$, $\sigma$) plutôt qu'un point : l'espace latent est **structuré** et continu (meilleure visualisation de MNIST, génération possible). Trick : $z = \mu + \sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$, pour que le gradient traverse l'échantillonnage.
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Qu'est-ce qu'un contrefactuel ? Et un semi-factuel ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Contrefactuel : la plus petite modification de l'exemple qui **franchit la frontière de décision** (un « 9 » modifié en « 4 »). Semi-factuel : un exemple modifié qui se rapproche de la frontière mais **reste** dans la même classe. Ils servent à explorer l'espace latent et à expliquer.
</details>

<details>
<summary><b>Question 13 (1 pt) :</b> Quels sont les 4 niveaux d'usage des architectures deep ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Utilisateur (modèles pré-entraînés), affineur (fine-tuning sur petits jeux), entraîneur (données jouets ou massives), développeur (jouer sur la plasticité des architectures).
</details>

<details>
<summary><b>Question 14 (1 pt) :</b> Qu'est-ce que le Deep LVQ ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Learning Vector Quantization : représenter l'espace par des prototypes (codebooks, cellules de Voronoï), idée proche du plus proche voisin ; la version deep apprend en plus la représentation dans laquelle vivent ces prototypes.
</details>
