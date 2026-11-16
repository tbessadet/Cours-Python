# La classification non supervisée (clustering) : partitionnement et hiérarchique

> ⚠️ **Note de rédaction** : les diapos **21 à 26, 31 à 36, 38-39 et 46 à 50** sont essentiellement des images (formules, schémas) et n'ont pas été lues en image. Leur contenu a été reconstitué à partir des titres et de mes connaissances : ces passages sont balisés 🔎. **À vérifier sur les diapos.**

## Sommaire

1. [Classification vs clustering](#s1)
2. [Objectif et difficultés du clustering](#s2)
3. [Similarité et distances](#s3)
4. [Qu'est-ce qu'une bonne partition ?](#s4)
5. [Étapes d'un système de clustering et taxonomie](#s5)
6. [Partitionnement : les k-moyennes (k-means)](#s6)
7. [Clustering hiérarchique (CAH / CHD)](#s7)
8. [Quiz d'auto-test](#s8)
9. [Pièges à éviter](#s9)

---

<a id="s1"></a>
## 1. Classification vs clustering

**La classification** (au sens large) : organiser un ensemble de « formes » en **groupes contrastés**, pour **comprimer et structurer** les données et permettre des **prédictions**.

| | **Clustering** | **Classification** |
|---|---|---|
| Type d'apprentissage | **Non supervisé** | **Supervisé** |
| Étiquettes | Inconnues : on **découvre** les classes | Connues : on cherche à **retrouver** les étiquettes des exemples |
| Nombre de classes | Peut être fourni en entrée (ex. k-means) | Fixé par les étiquettes |

Vocabulaire (diapo 4) : un **cluster** = un groupe de points ; chaque cluster a un **centre / prototype / référent** (le point représentatif du groupe).

> 💡 Synonymes de clustering : **catégorisation, partitionnement, segmentation**.

---

<a id="s2"></a>
## 2. Objectif et difficultés du clustering

### 2.1 Objectif
- Trouver des **regroupements « naturels »** (clusters) des exemples.
- Revient à **inventer un nouvel attribut** $C$ dont les valeurs $c_1, c_2, \dots, c_k$ indiquent le cluster de chaque exemple.
- **Critère** :
  - **maximiser la distance inter-cluster** (clusters bien séparés) ;
  - **minimiser la distance intra-cluster** (clusters compacts).

Les classes peuvent être :
- **bien définies** (nuages globulaires séparés, ou deux losanges qui se touchent) ;
- **imbriquées**, de **formes bizarres** (anneaux, formes allongées, lettres…), voire pire (bruit autour).

> 💡 Usage principal : **analyse exploratoire** de données (sciences sociales, biologie, médecine, astronomie, diagnostic, images, marketing…).

### 2.2 Il n'y a pas UNE bonne catégorisation (diapos 8-9)
- **Exemple des animaux** (mouton, chat, chien, requin, vipère, lézard, grenouille, moineau, mouette, rouget, poisson rouge) : selon le critère choisi, la partition change :
  - **port de la progéniture** : {mouton, chat, chien, requin} vs le reste ;
  - **existence de poumons** : {poisson rouge, rouget, requin} vs le reste ;
  - **type d'environnement** : terrestres+aériens / {grenouille} / aquatiques (3 groupes) ;
  - **progéniture + poumons** : {lézard, vipère, grenouille, moineau, mouette} vs {mouton, chat, chien, poisson rouge, rouget, requin}.
- **Exemple géométrique** : les mêmes points alignés peuvent être regroupés par **proximité** (« groupes ») ou par **alignement** (« linéarité »).

> 💡 La « bonne » partition dépend des **attributs** et de la **notion de ressemblance** choisis.

### 2.3 Sources de difficultés
- **Énorme espace de recherche** : fléau de la dimensionnalité, sélection d'attributs, descripteurs de types variés.
- **Choix du critère** : mesure de similarité/proximité, **nombre de groupes**.
- **Choix de l'algorithme**.
- **Validation / interprétation** des résultats.

### 2.4 Les problèmes
- **Types de données variés** : symbolique, numérique, matrice de similarité, mixte → comment les comparer ?
- **Clusters de formes arbitraires** : beaucoup de méthodes favorisent les **sphères**.
- **Paramètres** : besoin de connaissances minimales pour les fixer.
- **Sensibilité à l'ordre** de présentation des exemples.

### 2.5 La méthode idéale
| Propriété | Sens |
|---|---|
| **Résistante au bruit et aux anomalies** | (la détection de points isolés est un thème de recherche en soi) |
| **Grande dimension** | applicable avec beaucoup d'attributs |
| **Passage à l'échelle** | beaucoup d'exemples |
| **Incrémentale** | adaptée aux flux de données |
| **Stable** | mêmes structures à partir d'échantillons différents de la même distribution |

### 2.6 Les grandes approches (diapo 14)
- **Fondées sur les distances** : partition « plate » (k-moyennes, k-médoïdes…), hiérarchique (ascendante, descendante).
- **Fondées sur des modèles probabilistes** : mélanges de modèles, hiérarchie de modèles.
- **Fondées sur les grilles ou graphes** : Kohonen.

---

<a id="s3"></a>
## 3. Similarité et distances

> 💡 Il faut **choisir de manière avisée** la mesure de similarité : elle est **subjective** et **dépendante du contexte** (exemple : une femme aux cheveux bouclés et un cocker — semblables par la couleur et la texture, pas par l'espèce).

### 3.1 Vocabulaire
| Mesure | Interprétation |
|---|---|
| **Dissimilarité** $DM$ (ex. distance) | **plus elle est faible**, plus les points sont similaires |
| **Similarité** $SM$ | **plus elle est grande**, plus les points sont similaires |

$$DM = \text{borne} - SM$$

- $\text{borne}$ : valeur maximale de la similarité (ex. 1 si $SM \in [0,1]$).

### 3.2 Distance entre 2 points numériques $d(x_1, x_2)$

Hypothèses : $x_1, x_2 \in \mathbb{R}^p$ (vecteurs **ligne**), $i$ parcourt les $p$ attributs.

**Distance euclidienne**
$$d^2(x_1,x_2) = \sum_i \left(x_{1_i} - x_{2_i}\right)^2 = (x_1 - x_2)(x_1 - x_2)'$$

**Distance de Manhattan**
$$d(x_1,x_2) = \sum_i \left|x_{1_i} - x_{2_i}\right|$$

**Distance de Sebestyen**
$$d^2(x_1,x_2) = (x_1 - x_2)\,W\,(x_1 - x_2)'$$

**Distance de Mahalanobis**
$$d^2(x_1,x_2) = (x_1 - x_2)\,C^{-1}\,(x_1 - x_2)'$$

- $x_{1_i}$ : $i$-ème attribut de $x_1$
- $'$ : transposée (le produit ligne × colonne donne un scalaire)
- $W$ : **matrice diagonale de pondération** (un poids par attribut)
- $C$ : **matrice de covariance** des données

> 🔎 **À quoi servent-elles ?**
> - Euclidienne : la distance « à vol d'oiseau », la plus courante ; sensible aux **échelles** des attributs (→ standardiser).
> - Manhattan : somme des écarts par axe, moins sensible aux valeurs extrêmes.
> - Sebestyen = euclidienne **pondérée** : on donne plus ou moins d'importance à chaque attribut.
> - Mahalanobis : tient compte des **corrélations** et des variances ; si $C = I$, on retrouve l'euclidienne.

### 3.3 Distance entre 2 points à valeurs discrètes

On construit la **matrice de contingence** $A(x_1, x_2) = [a_{ij}]$ :
- $a_{ij}$ = nombre de positions où $x_1$ vaut $i$ **et** $x_2$ vaut $j$.

**Exemple du cours** : $x_1 = [0;1;2;1;2;1]$, $x_2 = [1;0;2;1;0;1]$

Paires (position par position) : (0,1), (1,0), (2,2), (1,1), (2,0), (1,1).

$$A(x_1,x_2) = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 2 & 0 \\ 1 & 0 & 1 \end{bmatrix}$$

**Distance de Hamming** : nombre de places où les 2 vecteurs **diffèrent** = somme des termes **hors diagonale**.

Hypothèses : les attributs prennent $k$ valeurs possibles $\{0, \dots, k-1\}$.
$$d(x_1,x_2) = \sum_{i=0}^{k-1} \; \sum_{\substack{j=0 \\ j \neq i}}^{k-1} a_{ij}$$

- $k$ : nombre de valeurs possibles (ici $k = 3$)
- la diagonale $a_{ii}$ compte les positions **identiques**, d'où $j \neq i$

> 💡 Sur l'exemple : $d = 1 + 1 + 1 = 3$ (les vecteurs diffèrent aux positions 1, 2 et 5).

**Distance de Tanimoto**, … (citée sans formule).

> 🔎 Pour des vecteurs binaires, la similarité de Tanimoto (≈ Jaccard) vaut $S_T = \dfrac{n_{11}}{n_{11} + n_{10} + n_{01}}$ ($n_{11}$ : nb de positions où les deux valent 1, etc.) et la distance $d_T = 1 - S_T$. Forme générale : $S_T(x_1,x_2) = \dfrac{x_1 \cdot x_2}{\|x_1\|^2 + \|x_2\|^2 - x_1 \cdot x_2}$.

### 3.4 Distance entre 2 classes $D(C_1, C_2)$ (diapos 21-23)

> 🔎 Les mesures classiques (détaillées en §7.3) :
> - **saut minimal** : distance entre les 2 points les plus proches ;
> - **saut maximal** : distance entre les 2 points les plus éloignés ;
> - **saut moyen** : moyenne des distances entre toutes les paires ;
> - **barycentres** : distance entre les centres de gravité ;
> - **Ward** : augmentation de l'inertie intra due à la fusion.

Questions posées par le cours (diapos 22-23) :
- **Un ou deux clusters ?** Deux groupes proches peuvent être vus comme un seul selon la mesure.
- **Cluster ou bruit ?** Un petit groupe isolé de points peut être un vrai cluster… ou du bruit.

---

<a id="s4"></a>
## 4. Qu'est-ce qu'une bonne partition ?

- **Inertie intra-cluster** = variance des points d'un **même cluster**.
- **Inertie inter-cluster** = variance des **centres** des clusters.

> 💡 Il faut **minimiser l'inertie intra** (clusters compacts) et **maximiser l'inertie inter** (clusters séparés).

> 🔎 Formules (notations usuelles). Hypothèses : $n$ points $x_i$, $K$ clusters $C_k$ d'effectif $n_k$, de centre $g_k$ ; $g$ = centre de gravité global.
> $$I_{\text{intra}} = \sum_{k=1}^{K} \sum_{x_i \in C_k} d^2(x_i, g_k) \qquad I_{\text{inter}} = \sum_{k=1}^{K} n_k \, d^2(g_k, g)$$
> **Théorème de Huygens** : $I_{\text{totale}} = I_{\text{intra}} + I_{\text{inter}}$.
> L'inertie totale étant fixe pour un jeu de données, **minimiser l'intra ⇔ maximiser l'inter**.

---

<a id="s5"></a>
## 5. Étapes d'un système de clustering et taxonomie

### 5.1 Les 5 étapes
1. **Représentation des données** : nombre de classes, nombre de données, nombre/type/échelle des attributs.
2. **Mesure de proximité** adaptée au domaine : fonction de distance entre paires.
3. **Regroupement** :
   - **Hard** : partition en groupes **distincts** ;
   - **Fuzzy** (flou) : chaque donnée a un **degré d'appartenance** à chaque cluster.
4. **Abstraction des données** (si nécessaire) : représentation compacte (ex. description de chaque cluster).
5. **Évaluation de la sortie** (si nécessaire) :

| Évaluation | Principe |
|---|---|
| **Externe** | comparer à une structure **a priori** (ex. vraies étiquettes) |
| **Interne** | la structure est-elle **intrinsèquement** adaptée aux données ? |
| **Relative** | **comparer** plusieurs structures possibles (ex. différents $k$) |

### 5.2 Taxonomie des méthodes
| Famille | Exemple |
|---|---|
| **Partitionnement** | k-means |
| **Hiérarchiques** | CAH |
| **À base de densité** | DBSCAN, OPTICS |
| **Modèles de mélange** | EM |
| **À base de grille** | cartes de Kohonen |

---

<a id="s6"></a>
## 6. Partitionnement : les k-moyennes (k-means)

### 6.1 Principe
- Cherche **$k$ partitions** des exemples, $k$ **fourni par l'utilisateur**.
- Données **numériques** uniquement.
- Hypothèse implicite : les données proviennent de **$k$ gaussiennes sphériques**.
- Répartir les $N$ points en $K$ ensembles **disjoints**, en regroupant les points proches.
- Minimise la **distorsion** : somme des distances entre les exemples et le **centre de gravité** de leur cluster.

🔎 Problème de minimisation. Hypothèses : $N$ points $x_i$, $K$ clusters $C_k$ disjoints de centres $\mu_k$.
$$\min_{C_1,\dots,C_K} \; J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \left\| x_i - \mu_k \right\|^2 \qquad \text{avec} \qquad \mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$$

- $J$ : distorsion (= inertie intra-cluster)
- $\mu_k$ : centroïde (moyenne) du cluster $k$
- $|C_k|$ : nombre de points du cluster $k$

> ⚠️ Problème **NP-difficile** : on ne peut trouver qu'un **minimum local**.

### 6.2 Algorithme (🔎 diapos 32-36 non lues en image)
1. **Initialiser** $K$ centroïdes $\mu_1, \dots, \mu_K$.
2. **Affectation** : chaque point est affecté au cluster du centroïde **le plus proche**.
3. **Mise à jour** : chaque centroïde est recalculé comme la **moyenne** des points de son cluster.
4. **Répéter** 2-3 jusqu'à ce que les affectations ne changent plus (convergence).

> 💡 Chaque étape fait diminuer (ou laisse égale) la distorsion $J$ → l'algorithme converge forcément, mais vers un minimum **local**.

### 6.3 Initialisation et formes fortes
Initialisation des centroïdes :
- **aléatoirement dans l'intervalle** de définition des $x_i$ ;
- **aléatoirement parmi les $x_i$**.

⚠️ Des initialisations différentes ⇒ des clusters différents (**minima locaux**).

**Formes fortes** (clusters « stables ») :
1. on répète l'algorithme **$n$ fois** ;
2. on regroupe les $x_i$ qui se retrouvent **toujours dans les mêmes clusters** ;
3. on choisit les centroïdes parmi ces formes fortes.

### 6.4 K-means séquentiels (diapo 39)
Adaptation quand les exemples **arrivent au fur et à mesure** (flux).

> 🔎 Version classique : initialiser $\mu_1,\dots,\mu_K$ et des compteurs $n_k = 0$. Pour chaque nouvel exemple $x$ :
> - trouver le centroïde le plus proche $\mu_k$ ;
> - $n_k \leftarrow n_k + 1$ ;
> - $\mu_k \leftarrow \mu_k + \dfrac{1}{n_k}\,(x - \mu_k)$
>
> ($\mu_k$ reste égal à la moyenne des points affectés au cluster $k$, mise à jour sans stocker les points.)

### 6.5 Avantages, limites, analyse
| Avantages | Limites |
|---|---|
| Simple, efficace en **temps et mémoire** | **Sensible aux moyennes initiales** |
| Grandes bases (milliers d'objets) | Sensible à la **taille et la densité** des clusters |
| Convergence rapide (typiquement $T < 10$) | Sensible au **bruit** |
| | L'utilisateur doit **deviner $k$** |
| | Distance euclidienne par défaut : bien pour classes **globulaires**, mal pour classes **allongées** |

**Complexité** : $O(n\,d\,k\,T)$
- $n$ : nombre d'exemples
- $d$ : nombre d'attributs
- $k$ : nombre de clusters
- $T$ : nombre d'itérations

**À utiliser quand** :
- on a des **informations a priori** sur les centres ;
- en **essayant plusieurs initialisations** ;
- en **post-traitement d'un clustering hiérarchique** (la CAH fournit $k$ et les centres initiaux).

---

<a id="s7"></a>
## 7. Clustering hiérarchique (CAH / CHD)

### 7.1 Objectif et approches
Trouver une **hiérarchie** de catégories.

| **Descendante** (divisive, CHD) | **Ascendante** (agglomérative, CAH) |
|---|---|
| Partir du cluster « tout » | Partir des exemples (clusters élémentaires) |
| Diviser chaque cluster | Regrouper les 2 clusters les plus « proches » |
| Jusqu'à un critère d'arrêt | Jusqu'à obtenir le cluster « tout » |

### 7.2 CAH : algorithme
Chaque point/cluster est progressivement **« absorbé » par le cluster le plus proche**.

- **Initialisation** :
  - chaque individu est dans **son propre cluster** ;
  - calcul de la **matrice de ressemblance $M$** entre chaque couple de clusters.
- **Répéter** :
  - sélectionner dans $M$ les 2 clusters les plus proches $C_I$ et $C_J$ ;
  - les **fusionner** en un cluster plus général $C_G$ ;
  - **mettre à jour $M$** : ressemblance entre $C_G$ et les clusters existants.
- **Jusqu'à** la fusion des 2 derniers clusters.

**Dendrogramme** : arbre représentant les **fusions successives**. La **hauteur** d'un nœud = **dissimilarité entre les 2 clusters avant fusion** (sauf exception avec certaines mesures).

> 🔎 Pour obtenir une partition, on **coupe le dendrogramme** à une hauteur donnée : un grand saut de hauteur entre deux fusions suggère un bon nombre de clusters. L'« exception » : avec la méthode des **barycentres**, une fusion peut se faire à une hauteur **plus basse** que la précédente (**inversion**), et le dendrogramme n'est plus monotone.

### 7.3 Distance entre clusters (« métrique » de la CAH)

Problème : trouver l'**ultramétrique** (distance entre clusters) **la plus proche** de la métrique utilisée pour les individus.

> 🔎 Une **ultramétrique** vérifie $d(x,z) \le \max\big(d(x,y), d(y,z)\big)$. C'est exactement la distance « lue » sur un dendrogramme (hauteur de la première fusion qui réunit deux points).

🔎 Formules usuelles. Hypothèses : $C_1, C_2$ clusters d'effectifs $n_1, n_2$, de barycentres $g_1, g_2$ ; $d$ distance entre individus.

| Méthode | Formule $D(C_1,C_2)$ | Comportement (cours) |
|---|---|---|
| **Saut minimal** (single linkage, plus proche voisin) | $\min_{x \in C_1, y \in C_2} d(x,y)$ | classes **générales** (effet de **chaînage**) ; sensible au bruit |
| **Saut maximal** (complete linkage, diamètre max) | $\max_{x \in C_1, y \in C_2} d(x,y)$ | classes **spécifiques** (on ne regroupe que des classes très proches) ; sensible au bruit |
| **Saut moyen** | $\dfrac{1}{n_1 n_2}\sum_{x \in C_1}\sum_{y \in C_2} d(x,y)$ | classes de **variance proche** |
| **Barycentre** | $d(g_1, g_2)$ | **bonne résistance au bruit** |
| **Ward** | $\dfrac{n_1 n_2}{n_1 + n_2}\, d^2(g_1, g_2)$ | prend **en plus en compte les effectifs** |

> 🔎 Ward = **augmentation de l'inertie intra** causée par la fusion : à chaque étape, on fait la fusion qui dégrade le moins la compacité. C'est le lien naturel avec k-means (même critère d'inertie).

### 7.4 Illustration (diapos 49-50)
🔎 Exemple de 8 points numérotés 1 à 8 fusionnés pas à pas, avec le dendrogramme correspondant (à revoir sur les diapos ; même principe que le TD CAH).

### 7.5 Complexité
- **CAH** : $O(n^3)$
- Variantes plus rapides :
  - **CURE** : $O(n^2)$ en petite dimension, $O(n^2 + n\,m \log n)$ sinon (en partant de $m$ clusters) ;
  - **CHAMELEON** : graphe des k-ppv en $O(n \log n)$ en petite dimension, sinon $O(n^2)$ ; clustering en $O(nm + n\log n + m^2 \log m)$.

> 🔎 $n^3$ : $n$ fusions, et à chaque fusion on cherche le minimum dans une matrice $n \times n$. Impraticable sur de très grands jeux de données → d'où l'intérêt de k-means.

### 7.6 Classification hiérarchique descendante (CHD)
- **Principe** : sélectionner le cluster **le moins cohérent**, puis le **subdiviser**.
- **Problème** : pour savoir comment séparer au mieux, il faut souvent… faire une **CAH** sur ce cluster.
- ⇒ Algorithme **beaucoup moins attractif** que la CAH.

---

<a id="s8"></a>
## 8. Quiz d'auto-test

<details>
<summary><b>Question 1 (2 pts) :</b> Quelle différence entre clustering et classification ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le **clustering** est **non supervisé** : on découvre des groupes sans étiquettes (le nombre de groupes peut être donné). La **classification** est **supervisée** : on apprend à retrouver des étiquettes connues.
</details>

<details>
<summary><b>Question 2 (2 pts) :</b> Quel critère cherche-t-on à optimiser en clustering ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**Minimiser** l'inertie (distance) **intra**-cluster et **maximiser** l'inertie **inter**-cluster. Par Huygens ($I_{\text{tot}} = I_{\text{intra}} + I_{\text{inter}}$) 🔎, les deux sont équivalents.
</details>

<details>
<summary><b>Question 3 (1 pt) :</b> Pourquoi dit-on qu'il n'existe pas une unique « bonne » catégorisation ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Parce qu'elle dépend des **attributs** et de la **mesure de similarité** choisis : les animaux se regroupent différemment selon le port de la progéniture, les poumons ou l'environnement ; des points alignés se regroupent par proximité ou par linéarité.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> Différence entre distance euclidienne, Sebestyen et Mahalanobis ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Toutes s'écrivent $d^2 = (x_1 - x_2)\,M\,(x_1 - x_2)'$ :
- euclidienne : $M = I$ ;
- Sebestyen : $M = W$, **diagonale de pondération** (un poids par attribut) ;
- Mahalanobis : $M = C^{-1}$, inverse de la **covariance** (tient compte des corrélations).
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> Calculer la distance de Hamming entre x₁ = [0;1;2;1;2;1] et x₂ = [1;0;2;1;0;1] avec la matrice de contingence. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$A = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 2 & 0 \\ 1 & 0 & 1 \end{bmatrix}$$

Hamming = somme hors diagonale = $1 + 1 + 1 = 3$.
</details>

<details>
<summary><b>Question 6 (1 pt) :</b> Relation entre une mesure de similarité SM et de dissimilarité DM ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$DM = \text{borne} - SM$. Une dissimilarité faible ⇔ points similaires ; une similarité grande ⇔ points similaires.
</details>

<details>
<summary><b>Question 7 (1 pt) :</b> Quels sont les 3 types d'évaluation d'un clustering ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **Externe** : comparaison à une structure a priori ;
- **Interne** : la structure est-elle intrinsèquement adaptée aux données ;
- **Relative** : comparaison de plusieurs structures.
</details>

<details>
<summary><b>Question 8 (1 pt) :</b> Clustering hard vs fuzzy ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**Hard** : chaque donnée appartient à un seul groupe. **Fuzzy** : chaque donnée a un **degré d'appartenance** à chacun des clusters.
</details>

<details>
<summary><b>Question 9 (3 pts) :</b> Décrire l'algorithme des k-means et la quantité qu'il minimise. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Minimise la **distorsion** $J = \sum_{k} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$.
1. Initialiser $k$ centroïdes ;
2. affecter chaque point au centroïde le plus proche ;
3. recalculer chaque centroïde comme moyenne de son cluster ;
4. répéter jusqu'à stabilité.
</details>

<details>
<summary><b>Question 10 (2 pts) :</b> Pourquoi k-means peut-il donner des résultats différents d'une exécution à l'autre, et que faire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le problème est **NP-difficile** ; l'algorithme converge vers un **minimum local** qui dépend de l'**initialisation**. Solutions : répéter avec plusieurs initialisations, chercher les **formes fortes** (points toujours regroupés ensemble), utiliser des infos a priori, ou initialiser avec une CAH.
</details>

<details>
<summary><b>Question 11 (1 pt) :</b> Quelle est la complexité de k-means ? Celle de la CAH ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

k-means : $O(n\,d\,k\,T)$ (avec typiquement $T < 10$). CAH : $O(n^3)$.
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Quelles hypothèses/limites viennent de la distance euclidienne dans k-means ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

k-means suppose des données issues de **k gaussiennes sphériques** : la distance euclidienne détecte bien les classes **globulaires**, mal les classes **allongées** ou de formes arbitraires. Il est aussi sensible à la taille/densité des clusters et au bruit.
</details>

<details>
<summary><b>Question 13 (1 pt) :</b> À quoi sert le k-means séquentiel ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

À traiter des exemples qui **arrivent au fur et à mesure** (flux) : chaque nouvel exemple met à jour le centroïde le plus proche, 🔎 $\mu_k \leftarrow \mu_k + \frac{1}{n_k}(x - \mu_k)$.
</details>

<details>
<summary><b>Question 14 (3 pts) :</b> Décrire l'algorithme de la CAH. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Chaque individu forme son propre cluster ; calcul de la matrice de ressemblance $M$.
2. Répéter : trouver les 2 clusters les plus proches $C_I, C_J$ ; les fusionner en $C_G$ ; mettre à jour $M$ (distance entre $C_G$ et les autres).
3. Jusqu'à la fusion des 2 derniers clusters.

Le résultat se lit sur un **dendrogramme**.
</details>

<details>
<summary><b>Question 15 (1 pt) :</b> Que représente la hauteur d'un nœud dans un dendrogramme ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

La **dissimilarité entre les 2 clusters** juste avant leur fusion (sauf exception, ex. inversions avec la méthode des barycentres 🔎).
</details>

<details>
<summary><b>Question 16 (3 pts) :</b> Comparer saut minimal, saut maximal, saut moyen, barycentre et Ward. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **Saut minimal** : classes générales (**chaînage**), sensible au bruit.
- **Saut maximal** : classes spécifiques, sensible au bruit.
- **Saut moyen** : classes de variance proche.
- **Barycentre** : bonne résistance au bruit.
- **Ward** : tient compte des **effectifs** ; 🔎 $D = \frac{n_1 n_2}{n_1 + n_2} d^2(g_1, g_2)$, = hausse d'inertie intra.
</details>

<details>
<summary><b>Question 17 (1 pt) :</b> Pourquoi la classification hiérarchique descendante est-elle peu utilisée ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Pour subdiviser le cluster le moins cohérent, il faut souvent faire une **CAH** pour trouver la meilleure séparation : c'est plus coûteux et moins attractif.
</details>

<details>
<summary><b>Question 18 (1 pt) :</b> Citer les 5 familles de méthodes de clustering avec un exemple. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Partitionnement (k-means), hiérarchique (CAH), densité (DBSCAN, OPTICS), modèles de mélange (EM), grille (Kohonen).
</details>

---

<a id="s9"></a>
## 9. Pièges à éviter

> 🚩 **Confondre clustering et classification** : le clustering n'a **pas d'étiquettes**.

> 🚩 **k-means trouve « la » solution** : non, seulement un **minimum local** dépendant de l'initialisation → toujours tester plusieurs initialisations.

> 🚩 **Oublier que $k$ est un paramètre** : k-means ne trouve pas $k$ tout seul ; c'est à l'utilisateur de le fixer (ou de comparer plusieurs $k$ : évaluation **relative**).

> 🚩 **Appliquer k-means à des données non numériques** ou à des clusters **allongés/non sphériques** : la moyenne et la distance euclidienne ne conviennent pas.

> 🚩 **Ne pas normaliser les attributs** 🔎 : avec la distance euclidienne, un attribut à grande échelle écrase les autres.

> 🚩 **Inverser intra et inter** : intra = **à minimiser** (compacité), inter = **à maximiser** (séparation).

> 🚩 **Hamming** : on somme les termes **hors diagonale** de la matrice de contingence (la diagonale = positions identiques).

> 🚩 **Saut minimal ≠ « meilleur »** : il provoque un **effet de chaînage** (clusters très étendus) ; saut minimal et maximal sont tous deux **sensibles au bruit**.

> 🚩 **CAH sur de gros jeux de données** : complexité $O(n^3)$ → coûteux ; k-means est en $O(ndkT)$.

> 🚩 **Mahalanobis** utilise $C^{-1}$ (inverse de la covariance), pas $C$.
