# 📘 Fiche de révision — Réseaux convolutionnels pour l'image

> Cours de Vincent Guigue (inspiré des supports de N. Baskiotis & B. Piwowarski) · AgroParisTech – IODAA · 27 diapos (`Convolutions.pdf`), complété par tes notes (`Convolutions.md`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 tes notes

---

## Sommaire

1. [Introduction : tâches et limites des MLP](#1-introduction--tâches-et-limites-des-mlp)
2. [La convolution](#2-la-convolution)
3. [Pooling et couche convolutionnelle usuelle](#3-pooling-et-couche-convolutionnelle-usuelle)
4. [Paramètres et dimensions](#4-paramètres-et-dimensions)
5. [Fine-tuning](#5-fine-tuning)
6. [Architectures en vision](#6-architectures-en-vision)
7. [Problèmes, modèles et métriques](#7-problèmes-modèles-et-métriques)
8. [Introspection d'un CNN](#8-introspection-dun-cnn)
9. [Graph Neural Networks](#9-graph-neural-networks)
10. [Pièges à éviter](#10-pièges-à-éviter)
11. [Quiz auto-test](#11-quiz-auto-test)

---

## 1. Introduction : tâches et limites des MLP

### 1.1 Tâches en *computer vision* (diapos 2–3)

| Tâche | Sortie |
|---|---|
| **Object recognition / classification** | une étiquette (« Cat ») |
| **Classification + localisation** | une étiquette + **une** boîte |
| **Object detection** | **plusieurs** objets : étiquette + boîte pour chacun (cat, ball / cat, dog, duck) |
| **Semantic segmentation** | une classe **par pixel** (cat / grass / tree), sans distinguer les individus (« no object, just pixels ») |
| **Instance segmentation** | une classe par pixel **et** séparation des individus (cat, cat, dog, duck) |
| **Image captioning** | une phrase (« A cat is playing a ball ») |
| **Image question answering** | réponse à une question (« How many balls? » → « One ») |
| **Image generation** | une image à partir d'un texte |
| Logique (LiLi) | apprentissage d'opérations logiques sur des images (ex. image binaire → règle R → image binaire) |

Le domaine du *computer vision* couvre aussi : *object tracking*, estimation de pose, reconnaissance d'action, navigation autonome, reconstruction 3D, compréhension de foules / de scènes urbaines et intérieures, collaboration multi-agents, surveillance aérienne…

### 1.2 Pourquoi les MLP ne sont pas suffisants ? (diapo 4)

- **Image = grille de pixels** ⇒ sémantique **très pauvre** (un pixel seul ne veut rien dire).
- **Fully-connected (MLP) ⇔ une entrée = un rôle propre** ⇒ **dépendance à la position du pixel**.
- ≠ Dans une image, **l'absolu n'a pas d'importance, le relatif est bien plus important** (mais pas seulement).

> 💡 **Les invariants sont très importants en image** car un objet peut prendre de multiples formes : angle de vue, pose de l'objet, éclairage, occlusion, échelle, environnement.

> 📝 **Tes notes** — « Le MLP n'a pas la propriété d'invariance par translation, rotation… qui permet la robustesse et l'adaptation au changement de contexte. »
>
> ✏️ **Correction / précision** : c'est juste pour la **translation**, mais il faut nuancer : la **convolution** est **équivariante** par translation (un motif décalé produit la même réponse, décalée) et c'est le **pooling** qui apporte une **invariance locale** aux petites translations. En revanche, un CNN **n'est pas invariant par rotation** ni par changement d'échelle de façon native : on l'obtient (approximativement) par la **data augmentation** (rotations, recadrages…) vue au §6 du module précédent et dans le TP.

### 1.3 Avant le *deep* (diapo 5)

1. **Extraire un dictionnaire de *features*** (*keypoints* détectés ou grille, puis descripteurs = vecteurs de caractéristiques) ;
2. **agréger les features** (représentation type ***Bag of Words*** : clustering des descripteurs → **vocabulaire** de « mots visuels ») ;
3. **utiliser un classifieur** sur cette représentation pour classer.

La « matrice des données » a une ligne par **image** et une colonne par **mot visuel** (on compte les occurrences), et on prédit la **classe**.

> 📝 **Tes notes** — « On crée des descripteurs clefs, on regarde quel descripteur tique où et on compte, on fait du clustering et on trouve des paquets "vocabulary", puis on utilise un SVM et on fait de la classification sur les descripteurs. » ✅ Conforme au cours (le classifieur était typiquement un SVM). Petite précision d'ordre : on **clusterise d'abord** les descripteurs pour construire le vocabulaire, **puis** on compte, pour chaque image, combien de ses descripteurs tombent dans chaque cluster.

### 1.4 Réseaux convolutifs (diapo 6)

Architecture type : `INPUT → [CONVOLUTION + RELU → POOLING] × n → FLATTEN → FULLY CONNECTED → SOFTMAX`, découpée en **feature learning** (convolutions) puis **classification** (couches denses).

> 💡 **Convolution = filtre d'analyse glissant sur l'image**
> - **peu de paramètres** (le même filtre est réutilisé partout) ;
> - **apprentissage des motifs à extraire** (les poids du filtre sont appris) ;
> - **agrégation progressive des échelles** (chaque couche voit une zone plus large).

---

## 2. La convolution

### 2.1 Convolution 1D (diapo 7)

*Commençons par un exemple 1D pour bien comprendre* (un signal temporel) :
- un **filtre glissant** (*sliding filter*) parcourt le signal ; ses coefficients sont appris par **rétro-propagation de l'erreur** depuis une couche *fully connected* finale ;
- **plusieurs filtres** (*sliding filters*) en parallèle → plusieurs signaux de sortie ;
- **max / average pooling** pour résumer chaque sortie sur des fenêtres ;
- on peut **empiler** convolution → pooling → convolution… avant la couche *fully connected*.

| Propriétés | |
|---|---|
| Fenêtre glissante | Apprentissage des motifs à détecter |
| Peu de paramètres | Invariance / dépendance à la localisation |
| Largeur / *stride* / *padding* | |

**Dilatation et champ récepteur** (empilement de 3 couches à noyau de taille 3) :

| Couche | Dilatation | Champ récepteur |
|---|---|---|
| 1 | 1 | 3 |
| 2 | 2 | 7 |
| 3 | 4 | 15 |

> 💡 En **doublant la dilatation** à chaque couche, le **champ récepteur** (la portion d'entrée qui influence un neurone) croît **exponentiellement** avec la profondeur, sans ajouter de paramètres.

### 2.2 Convolution 2D (diapo 8)

Exemple travaillé : image binaire **5×5** convoluée par un filtre **3×3** :

$$I = \begin{pmatrix} 1&1&1&0&0\\0&1&1&1&0\\0&0&1&1&1\\0&0&1&1&0\\0&1&1&0&0\end{pmatrix}, \qquad K = \begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix} \quad\Longrightarrow\quad I * K = \begin{pmatrix}4&3&4\\2&4&3\\2&3&4\end{pmatrix}$$

Le filtre glisse de gauche à droite puis de haut en bas ; chaque case de sortie (*convolved feature*) = **somme des produits terme à terme** entre le filtre et la fenêtre de l'image.

🔎 Formule (hypothèses : image $I$, noyau $K$ de taille $k\times k$, *stride* 1, sans *padding*) :

$$(I * K)_{i,j} = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} I_{i+u,\, j+v}\; K_{u,v} \;+\; b$$

- $(i,j)$ : position de la sortie ; $(u,v)$ : position dans le filtre ; $b$ : biais.
- Ex. première case : $1\cdot1 + 1\cdot0 + 1\cdot1 + 0\cdot0 + 1\cdot1 + 1\cdot0 + 0\cdot1 + 0\cdot0 + 1\cdot1 = 4$.

> 🔎 En deep learning, l'opération calculée est en fait une **corrélation croisée** (pas de retournement du filtre) ; on garde le nom de « convolution ». Sur une image à $C$ canaux (RGB), le filtre a la taille $C\times k\times k$ et somme sur tous les canaux.

---

## 3. Pooling et couche convolutionnelle usuelle

### 3.1 Couche de *pooling* (diapo 9)

> 💡 **Pooling (ou *subsampling*) : réduire la dimensionnalité de sortie.**
> - **Max pooling** : on prend le **max** sur une fenêtre ;
> - **Average pooling** : on fait la **moyenne** ;
> - **Sum pooling** : la **somme**…

Exemple (max pooling avec filtres 2×2 et *stride* 2) :

$$\begin{pmatrix}1&1&2&4\\5&6&7&8\\3&2&1&0\\1&2&3&4\end{pmatrix} \xrightarrow{\text{max pool } 2\times2,\ \text{stride } 2} \begin{pmatrix}6&8\\3&4\end{pmatrix}$$

🔎 Le pooling **n'a pas de paramètres** à apprendre et agit **canal par canal**.

### 3.2 Couche convolutionnelle usuelle (diapo 10)

Exemple de dimensions : entrée **11×11** → filtre **3×3** → **9×9** → **max pooling** → **3×3**.

Une « couche » de CNN enchaîne trois étages :
1. **Convolution layer** : transformation **affine** ;
2. **Detector layer** : **non-linéarité** (ex. *rectified linear*, ReLU) ;
3. **Pooling layer**.

Exemple d'architecture profonde (VGG, reconnaissance de visages) : $224\times224\times3 \to 224\times224\times64 \to 112\times112\times128 \to 56\times56\times256 \to 28\times28\times512 \to 14\times14\times512 \to 7\times7\times512 \to$ FC 4096 → FC 4096 → softmax (2622 classes). À chaque *maxpool*, la **résolution spatiale est divisée par 2** et la **profondeur** (nombre de filtres) augmente : 64 → 128 → 256 → 512.

### 3.3 Exemple : reconnaissance de caractères — LeNet-5 (diapo 11)

`INPUT 32×32 → C1 : 6 feature maps 28×28 → S2 : 6 maps 14×14 (subsampling) → C3 : 16 maps 10×10 → S4 : 16 maps 5×5 → C5 : couche de 120 → F6 : couche de 84 → OUTPUT 10` (convolutions, sous-échantillonnages, puis connexions complètes).

> 🔎 Vérification avec la formule du §4.2 : $(32 - 5)/1 + 1 = 28$ (noyau 5) ; pooling 2×2 → 14 ; $(14-5)+1 = 10$ ; pooling → 5.

---

## 4. Paramètres et dimensions

### 4.1 Récapitulatif des paramètres (diapo 12)

| Paramètre | Rôle |
|---|---|
| `kernel_size` | taille du filtre (fenêtre) |
| `stride` | pas de déplacement du filtre |
| `padding` | bordure ajoutée autour de l'entrée (souvent des zéros) |
| `in_channels` | nombre de canaux d'entrée (profondeur de l'entrée) |
| `out_channels` | nombre de **filtres** = nombre de canaux de sortie |
| `dilation` | espacement entre les éléments du filtre (filtre « à trous ») |

```python
class ConvNet(nn.Module):
    # argument sur le choix du pooling
    def __init__(self, pooling=nn.MaxPool2d):
        super().__init__()
        # Convolution 5*5, 16 filtres
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=5, stride=1, padding=0, dilation=1)
        # Max pooling 3x3
        self.pool1 = pooling(kernel_size=3, stride=1)
        ...
```

### 4.2 Dimensions d'entrée / sortie (diapo 13)

*This formula should give you the output size per channel* — hypothèses : une dimension spatiale (hauteur ou largeur), *dilation* = 1 :

$$\text{sortie} = \frac{W - K + 2P}{S} + 1$$

- $W$ : taille de l'entrée (*input volume*) ;
- $K$ : taille du noyau (*kernel size*) ;
- $P$ : *padding* ;
- $S$ : *stride* (la division est **entière**, arrondie vers le bas).

> 🔎 Avec dilatation $D$ : $\left\lfloor \frac{W + 2P - D(K-1) - 1}{S}\right\rfloor + 1$. Le nombre de canaux de sortie vaut `out_channels`, indépendamment de la formule. Nombre de **paramètres** d'une `Conv2d` : $C_{out}\times(C_{in}\times K\times K + 1)$.

Exemples (TP) : entrée 6×6 → noyau 3 : **4×4** ; noyau 2, *stride* 2 : **3×3** ; noyau 2, *padding* 1 : **7×7**.

---

## 5. Fine-tuning

### 5.1 L'approche *fine-tuning* (diapo 14)

- **Learning a network from scratch** ⇒ OK, bons descripteurs… **mais** : nécessite **beaucoup de données** ; performance **(souvent) non optimale**.
- ⇒ **Fine-tuner un réseau** (pré-entraîné) :
  - garder les couches **originales** (convolutions) et remplacer les **dernières couches FC** par de **nouvelles** couches (adaptées aux nouvelles classes) ;
  - **How many layers to keep / re-train?** Les couches basses extraient des *features* **bas niveau** (bords, couleurs), puis **moyen niveau**, puis **haut niveau** (parties d'objets), avant le **classifieur entraînable**.

| Stratégie | Ce qu'on entraîne |
|---|---|
| **Freeze early layers in network** | **seulement les couches FC** (les convolutions sont gelées) |
| **Unfreeze early layers & train all** | tout le réseau (avec les poids pré-entraînés comme initialisation) |

- **Do we need to fine-tune all?**
- **Catastrophic forgetting** : en optimisant pour la tâche B, les paramètres $\theta$ quittent la zone de faible erreur pour la tâche A ($\theta^\star_A \to \theta^\star_B$) : le réseau **oublie** la tâche initiale.

> 💡 Plus le nouveau jeu de données est **petit** et **proche** du jeu d'origine, plus on **gèle** de couches. Avec beaucoup de données ou un domaine éloigné, on ré-entraîne davantage (avec un petit learning rate).

---

## 6. Architectures en vision

### 6.1 AlexNet (diapo 15)

Entrée $224\times224\times3$ → convolutions **11×11** (*stride* 4), **5×5**, **3×3**, *max pooling*, puis 3 couches denses (4096, 4096, 1000).
- **11×11, 5×5, 3×3 convolutions** ;
- **max-pooling, ReLU activations** ;
- **Dropout et data-augmentation**.

### 6.2 Deep CNN / VGG / ResNet (diapo 16)

- **VGG** : empilement **uniquement** de convolutions **3×3** et de *pooling* /2 (64 → 128 → 256 → 512 filtres), puis FC 4096, 4096 → softmax.
- **ResNet** (« 34-layer residual » vs « 34-layer plain ») : même empilement, mais avec des **connexions résiduelles** (les flèches qui sautent deux couches).

> 📝 **Tes notes** — « À expliquer : residual layer. »
>
> 🔎 **Bloc résiduel** : au lieu d'apprendre directement $H(\mathbf x)$, deux couches apprennent un **résidu** $F(\mathbf x)$ et on **ajoute l'entrée** à la sortie (*skip connection*) :
> $$\mathbf y = F(\mathbf x) + \mathbf x$$
> - $\mathbf x$ : entrée du bloc ; $F$ : deux convolutions 3×3 (+ BatchNorm, ReLU) ; $\mathbf y$ : sortie.
>
> **Pourquoi ça marche** : (1) le gradient passe **directement** par le raccourci ($\partial\mathbf y/\partial\mathbf x = \partial F/\partial \mathbf x + I$), ce qui combat le **vanishing gradient** et permet d'entraîner des réseaux de 50, 100, 150 couches ; (2) si une couche est inutile, il suffit d'apprendre $F = 0$ (identité), donc ajouter des couches **ne dégrade pas** le réseau, alors qu'un réseau « plain » très profond s'entraîne moins bien qu'un moins profond. Quand les dimensions changent (pointillés sur le schéma), le raccourci passe par une projection (convolution 1×1).

### 6.3 Un grand nombre d'architectures (diapo 17)

Comparatif (ImageNet) : précision top-1 de ~55 % (AlexNet) à ~80 % (Inception-v4), en passant par BN-NIN, GoogLeNet, VGG-16/19, ResNet-18/34/50/101/152, Inception-v3. Le second graphique croise **précision**, **nombre d'opérations** (G-Ops) et **nombre de paramètres** (taille des bulles) : VGG est très coûteux (~140 M paramètres) pour une précision inférieure aux ResNet/Inception.

> 💡 Plus de paramètres ≠ meilleur : l'architecture compte (ResNet-50 bat VGG-19 avec beaucoup moins de paramètres).

### 6.4 *Image reconstruction* et transfert de style (diapo 18)

- **Générer des images en combinant contenu et style** ; utilise un **CNN entraîné de façon discriminante** ; génération d'image = **problème inverse sur le CNN** (on optimise **l'image d'entrée**, pas les poids).

Formules du schéma (Gatys et al.) — hypothèses : $F^l$ = activations (*feature maps*) de l'image générée $\vec x$ à la couche $l$, $P^l$ celles de l'image de **contenu** $\vec p$, $A^l$ la matrice de Gram de l'image de **style** $\vec a$ :

$$\mathcal L_{content} = \sum (F^l - P^l)^2, \qquad G^l_{ij} = \sum_k F^l_{ik}F^l_{jk}, \qquad E_l = \sum (G^l - A^l)^2, \qquad \mathcal L_{style} = \sum_l w_l E_l$$

$$\mathcal L_{total} = \alpha\,\mathcal L_{content} + \beta\,\mathcal L_{style}, \qquad \vec x \leftarrow \vec x - \lambda\,\frac{\partial \mathcal L_{total}}{\partial \vec x}$$

- $G^l$ : **matrice de Gram** (corrélations entre filtres $i$ et $j$ sur toutes les positions $k$) = « texture », indépendante de la position → capte le **style** ;
- $w_l$ : poids de chaque couche de style ; $\alpha, \beta$ : compromis contenu/style ; $\lambda$ : pas de gradient ;
- on part d'un **bruit blanc** $\vec x$ et on descend le gradient **par rapport à l'image**.

---

## 7. Problèmes, modèles et métriques

### 7.1 Classification (diapo 19)

| | |
|---|---|
| Problème | $\mathcal I \to \mathcal C$, $X \mapsto f(X)$, avec $\mathcal I = RGB^{d\times d}$, **1000 classes** (ImageNet) |
| Modèles | VGG, ResNet, etc. |
| **Métriques** | accuracy, précision, rappel |

- *How to switch to multi-class?* Les classes d'ImageNet sont **hiérarchiques** (mammal → placental → carnivore → canine → dog → working dog → husky ; vehicle → craft → watercraft → sailing vessel → sailboat → trimaran).
- ⇒ ***Think about transfer learning!***

### 7.2 Détection (diapo 19)

| | |
|---|---|
| Problème | $\mathcal I \to \mathcal C \times \mathcal B$, avec $\mathcal B = (pos_x, pos_y, width, height)$ |
| Nature | ⇒ **un problème de régression (!)** pour la boîte |
| Modèles | **Yolo, RCNN, fast-RCNN** |
| **Métriques** | **intersection géométrique des aires** : *Intersection over Union* (IoU) |

$$\mathrm{IoU}(A, B) = \frac{\text{aire}(A \cap B)}{\text{aire}(A \cup B)}$$

- $A$ : boîte prédite ; $B$ : boîte réelle ; $\mathrm{IoU} \in [0, 1]$ (🔎 une détection est souvent comptée correcte si $\mathrm{IoU} \ge 0{,}5$).

> 📝 **Tes notes** — « Pour trouver les boîtes qui contournent les objets : former un carré = 2 coordonnées (x, y) + 2 dimensions (hauteur, largeur) ⇒ quadruple régression pour entourer un objet, puis classifier pour dire ce qu'il y a dans la boîte. » ✅ Conforme ($\mathcal B = (pos_x, pos_y, width, height)$). Précision : c'est un **rectangle** (pas forcément un carré), et dans les détecteurs modernes (YOLO) la régression de la boîte et la classification se font **en même temps**, pour de nombreuses boîtes candidates.
>
> 📝 **Tes notes** — « IoU : métrique pour savoir si on a bien détecté l'objet ; *box level annotation* : contour de l'objet en boîte ; *pixel level annotation* : contour au pixel près (zone coloriée), passe par convolution, maxpooling, upscaling. » ✅ Conforme (voir §7.3).

### 7.3 Segmentation (diapo 19)

| | |
|---|---|
| Problème | $\mathcal I \to \mathcal I'$ : **pixel level annotation** ⇒ *how to output an image?* |
| Modèle | **U-Net** (encodeur-décodeur convolutionnel) |
| Métrique | IoU aussi (par classe, sur les pixels) |

- Le réseau **encode** (convolutions + *pooling* : la résolution diminue, la profondeur augmente) puis **décode** (*upsampling* : on remonte à la résolution d'origine) pour produire une carte de segmentation de même taille que l'image.
- ⇒ *Specificities wrt auto-encoders?*

> 🔎 Différence avec un auto-encodeur : la sortie n'est pas l'image d'entrée reconstruite mais une **carte de classes** (apprentissage supervisé), et U-Net ajoute des **connexions directes** (*skip connections*, « copy and crop ») entre les étages de même résolution de l'encodeur et du décodeur, pour récupérer les détails fins perdus au *pooling*.

### 7.4 *The Transformer wave* (diapo 20)

**Vision Transformer (ViT)** : l'image est découpée en **patchs**, chaque patch est aplati puis projeté linéairement (*Linear Projection of Flattened Patches*), on ajoute un **position embedding** et un *token* `[class]` appris ; le tout passe dans un **Transformer encoder**, puis une **MLP head** prédit la classe.
- ⇒ *Which differences / complementarities with CNN?*

> 🔎 Le CNN a un **biais inductif** fort (localité, partage des poids) : efficace avec peu de données. Le ViT, via l'attention, relie **tous les patchs entre eux dès la première couche** (contexte global) mais a besoin de **beaucoup plus de données** (ou d'un pré-entraînement massif).

### 7.5 CNN & représentation temps-fréquence (diapo 21)

- Exemple de la **séparation de sources** (qui a fait de grands progrès ces 5 dernières années).
- **Problème original** : ICA (*independent component analysis*) ; algorithme **SVD (non supervisé)** dans le domaine temporel ou temps-fréquence.
- **Nouveau problème** : un **problème de classification supervisée dans le domaine temps-fréquence** : le **spectrogramme** est traité comme une image par un CNN (on prédit, pour chaque case temps-fréquence, à quelle source elle appartient).

---

## 8. Introspection d'un CNN

### 8.1 Introspection d'un CNN (diapo 22)

- **Comprendre la classification** : quelle région de l'image **active** la classification, quels filtres sont les plus importants…
- Les filtres sont **de plus en plus abstraits**, produisant des « **mots élémentaires visuels** » (ex. visages : bords → yeux/nez → visages entiers ; voitures : bords → roues/vitres → voitures).
- Les couches **conservent les informations topologiques** (la position dans la *feature map* correspond à la position dans l'image).

### 8.2 Occlusion (diapo 23)

**Sensibilité de la classe détectée aux occlusions** : on masque une zone de l'image par un carré gris et on mesure la probabilité de la vraie classe ; la carte obtenue (bleu = probabilité qui chute) montre la zone **décisive** (ex. la tête du Poméranien, la roue de la voiture, le visage du lévrier afghan).

> 📝 **Tes notes** — « Technique par occlusion : on prend l'image, on la modifie en occultant une partie avec un carré gris/noir, on crée $n$ images avec une occlusion qui se déplace, on regarde si le modèle classe bien selon la position de l'occlusion → mapping des scores par pixel = l'endroit important pour la classification. » ✅ Conforme. Précision : on trace en général la **probabilité de la vraie classe** (ou sa baisse) en fonction de la position du carré, plutôt qu'un simple « bien / mal classé ».

### 8.3 Carte de saillance (diapo 24)

*Original image → saliency map → proto-objects.*

> 💡 Elles sont calculées en prenant le **gradient de la sortie par rapport à l'image d'entrée**. Elles permettent de mettre en avant les **pixels auxquels la classification est la plus sensible**.

🔎 Hypothèses : $S_c(I)$ score (ou probabilité) de la classe $c$ pour l'image $I$ :

$$M_{ij} = \max_{\text{canaux } ch}\left|\frac{\partial S_c(I)}{\partial I_{ch,i,j}}\right|$$

- $M_{ij}$ : saillance du pixel $(i,j)$ ; un gradient fort = un petit changement du pixel modifie fortement le score.

### 8.4 Class Activation Map (CAM) (diapo 25)

Schéma : `CONV → … → CONV → GAP → couche linéaire (poids w₁, w₂, …, wₙ) → classe (« Australian terrier »)` ; la carte = $w_1\cdot f_1 + w_2\cdot f_2 + \dots + w_n\cdot f_n$.

**Class Activation Maps :**
- **Global Average Pooling** (1 filtre = 1 *feature*) + **pondération** ;
- **combinaison linéaire** ;
- **agrégation pondérée des filtres** pour indiquer les **régions d'intérêt**.

🔎 Hypothèses : $f_k(x,y)$ sortie du filtre $k$ de la dernière couche convolutive à la position $(x,y)$, $w^c_k$ poids de la couche linéaire entre le filtre $k$ et la classe $c$ :

$$\mathrm{CAM}_c(x, y) = \sum_k w^c_k\, f_k(x, y)$$

- Le score de la classe est $S_c = \sum_k w^c_k \cdot \mathrm{GAP}(f_k)$ : la CAM « redéplie » ce score dans l'espace.

> 📝 **Tes notes** — « GradCam : on supprime la tête (réseau de neurones…), on la remplace par une décision linéaire et on fait une saillance map. »
>
> ✏️ **Correction** : ce que tu décris est la **CAM** (diapo 25) : on remplace la tête *fully-connected* par **Global Average Pooling + une couche linéaire**, on ré-entraîne cette tête, et la carte est la **somme des cartes de filtres pondérées par les poids linéaires** (ce n'est pas une carte de saillance au sens du §8.3, qui utilise le gradient par rapport aux pixels). 🔎 **Grad-CAM** (non vu en diapo) généralise la CAM **sans modifier l'architecture** : les poids $w^c_k$ sont remplacés par la **moyenne spatiale du gradient** du score de la classe par rapport à chaque carte $f_k$, puis on applique un ReLU : $\mathrm{GradCAM}_c = \mathrm{ReLU}\big(\sum_k \alpha^c_k f_k\big)$ avec $\alpha^c_k = \frac{1}{Z}\sum_{x,y}\frac{\partial S_c}{\partial f_k(x,y)}$.

### 8.5 LIME : *Local Interpretable Model-Agnostic Explanations* (diapo 26)

- **Sampling** ($P \approx$ sim) : on tire des perturbations proches de l'exemple ;
- **local linear model** (on weighted data) : on ajuste un modèle linéaire sur ces perturbations, pondérées par leur proximité ;
- **linear coefficients interpretation**.

Global : frontière **complexe non linéaire** ; local (autour de l'exemple) : une frontière **simple linéaire** suffit.

Exemple image : l'image est découpée en **super-pixels** ; chaque perturbation est un vecteur binaire (`[1,0,1,1,0,0,1,0,…]`) indiquant quels super-pixels sont gardés (les autres sont noircis) ; les coefficients du modèle linéaire désignent les super-pixels qui expliquent la décision (ici la tête du chien).

> 💡 **Model-agnostic** : LIME n'a besoin que des **prédictions** du modèle (boîte noire), contrairement à la saillance ou la CAM qui utilisent ses gradients / son architecture.

---

## 9. Graph Neural Networks

### 9.1 Passer des convolutions à une logique de graphe (diapo 27)

**Convolution = outil sur une grille régulière ⇒ passage à des liens quelconques** (un graphe : nœuds $n_i$, arêtes $e_{ji}$, chacun porteur d'un vecteur de caractéristiques).

Hypothèses : $\mathcal N(n_i)$ = voisins du nœud $n_i$, $e_{ji}$ = caractéristiques de l'arête de $n_j$ vers $n_i$ :

$$n_i^{(2)} = \sum_{n_j \in \mathcal N(n_i)} \psi(n_i, n_j, e_{ji})$$

- $n_i^{(2)}$ : nouvelle représentation du nœud $i$ (sa dimension = « nb filtres ») ;
- $\psi$ : fonction de message (un **réseau de neurones**, ex. PMC).

> 💡 **Message passing**
> - $\psi$ = réseau de neurones (ex. PMC) ;
> - nb couches ++ ⇒ ↗ **receptive field** (on agrège des voisins de plus en plus lointains) ;
> - possibilité de brancher des ***losses*** sur les **nœuds**, les **arcs/edges** ou les **graphes entiers** [trouver un agrégateur].

> 🔎 La convolution 2D est un cas particulier : chaque pixel est un nœud relié à ses 8 voisins, et $\psi$ dépend de la position relative (le poids du filtre).

---

## 10. Pièges à éviter

> 🚩 **Croire qu'un CNN est invariant par rotation / échelle** : seule la translation est bien gérée (équivariance des convolutions + invariance locale du pooling) ; le reste s'apprend par **data augmentation**.

> 🚩 **Mal calculer la taille de sortie** : $\lfloor (W - K + 2P)/S\rfloor + 1$, à appliquer à **chaque** convolution **et** chaque pooling, avant de dimensionner la première couche linéaire.

> 🚩 **Confondre CAM, Grad-CAM et carte de saillance** : CAM = GAP + linéaire (poids appris, architecture modifiée) ; Grad-CAM = poids issus des gradients sur les cartes de filtres ; saillance = gradient par rapport aux **pixels**.

> ⚠️ **Fine-tuning** : tout ré-entraîner sur peu de données → sur-apprentissage et **oubli catastrophique** ; geler les premières couches.

> ⚠️ **Détection = classification + régression** (4 valeurs de boîte) ; métrique **IoU**, pas l'accuracy.

> ⚠️ **Le pooling n'a pas de paramètres** et agit canal par canal ; le nombre de canaux de sortie d'une convolution = nombre de filtres (`out_channels`).

> ⚠️ **Transfert de style** : on optimise **l'image**, pas les poids ; le style passe par la **matrice de Gram** (corrélations entre filtres), le contenu par les activations brutes.

> ⚠️ **ViT vs CNN** : le ViT demande beaucoup de données (moins de biais inductif).

---

## 11. Quiz auto-test

<details>
<summary><b>Question 1 (3 pts) :</b> Pourquoi un MLP est-il mal adapté aux images, et qu'apporte la convolution ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Dans un MLP, chaque entrée (pixel) a un rôle propre : le modèle dépend de la **position absolue** des pixels, alors qu'en image c'est le **relatif** qui compte. Il a aussi énormément de paramètres. La convolution est un **filtre glissant** : peu de paramètres (partage des poids), motifs appris, équivariance par translation, agrégation progressive des échelles (avec le pooling, invariance locale).
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Quelle est la taille de sortie d'une convolution de noyau K, padding P, stride S sur une entrée de taille W ? Application : W=32, K=5, P=0, S=1, puis pooling 2×2 stride 2. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\text{sortie} = \left\lfloor\frac{W - K + 2P}{S}\right\rfloor + 1$$

$(32 - 5 + 0)/1 + 1 = 28$, puis pooling : $(28 - 2)/2 + 1 = 14$ (c'est le début de LeNet-5).
</details>

<details>
<summary><b>Question 3 (2 pts) :</b> Calculer la première case de la convolution de l'image 5×5 du cours par le filtre [[1,0,1],[0,1,0],[1,0,1]]. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Fenêtre en haut à gauche : $\begin{pmatrix}1&1&1\\0&1&1\\0&0&1\end{pmatrix}$. Somme des produits terme à terme avec le filtre : $1+0+1+0+1+0+0+0+1 = 4$. Sortie complète : $\begin{pmatrix}4&3&4\\2&4&3\\2&3&4\end{pmatrix}$.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> Que fait un max pooling 2×2 stride 2 sur [[1,1,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]] ? Combien a-t-il de paramètres ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$\begin{pmatrix}6&8\\3&4\end{pmatrix}$ (max de chaque bloc 2×2). Aucun paramètre à apprendre ; il réduit la dimension et apporte une invariance locale.
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> Comment évolue le champ récepteur en empilant des convolutions de taille 3 avec des dilatations 1, 2, 4 ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

3, puis 7, puis 15 : il croît exponentiellement avec la profondeur, sans paramètres supplémentaires.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Qu'est-ce que le fine-tuning, quelles stratégies existent, et quel risque mentionne le cours ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Partir d'un réseau **pré-entraîné** (ex. sur ImageNet) plutôt que de zéro (qui demande beaucoup de données et donne souvent une performance non optimale), et remplacer les dernières couches FC par une nouvelle tête. Stratégies : **geler** les premières couches et n'entraîner que les FC, ou **tout dégeler** et ré-entraîner. Risque : l'**oubli catastrophique** (les paramètres quittent la zone de faible erreur de la tâche A).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Qu'est-ce qu'une connexion résiduelle (ResNet) et pourquoi aide-t-elle ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le bloc calcule $\mathbf y = F(\mathbf x) + \mathbf x$ : les couches apprennent un **résidu** et l'entrée est ajoutée par un raccourci. Le gradient traverse directement le raccourci (lutte contre le vanishing gradient), et une couche inutile peut se réduire à l'identité ($F=0$) : on peut entraîner des réseaux très profonds (34, 50, 152 couches) sans dégradation.
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Pour chaque problème (classification, détection, segmentation), donner l'espace de sortie, un modèle et une métrique. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- Classification : $\mathcal I \to \mathcal C$ ; VGG, ResNet ; accuracy, précision, rappel.
- Détection : $\mathcal I \to \mathcal C \times \mathcal B$ avec $\mathcal B = (pos_x, pos_y, width, height)$ (régression) ; YOLO, RCNN, fast-RCNN ; IoU.
- Segmentation : $\mathcal I \to \mathcal I'$ (une classe par pixel) ; U-Net ; IoU.

$$\mathrm{IoU} = \frac{\text{aire}(A\cap B)}{\text{aire}(A\cup B)}$$
</details>

<details>
<summary><b>Question 9 (3 pts) :</b> Écrire les coûts du transfert de style. Pourquoi utiliser une matrice de Gram ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\mathcal L_{content} = \sum (F^l - P^l)^2, \quad G^l_{ij} = \sum_k F^l_{ik}F^l_{jk}, \quad \mathcal L_{style} = \sum_l w_l \sum (G^l - A^l)^2, \quad \mathcal L_{total} = \alpha\mathcal L_{content} + \beta\mathcal L_{style}$$

On part d'un bruit et on descend le gradient par rapport à l'**image**. La matrice de Gram mesure les **corrélations entre filtres** sur toutes les positions : elle capture la texture (le style) indépendamment de la disposition spatiale.
</details>

<details>
<summary><b>Question 10 (3 pts) :</b> Différence entre carte de saillance, occlusion, CAM et LIME ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **Saillance** : gradient de la sortie par rapport aux **pixels** d'entrée.
- **Occlusion** : on masque des zones et on mesure la chute de la probabilité de la classe.
- **CAM** : tête remplacée par GAP + linéaire ; carte $= \sum_k w^c_k f_k$ (filtres pondérés).
- **LIME** : perturbations locales (super-pixels), modèle linéaire local pondéré, lecture des coefficients ; **agnostique** au modèle.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Écrire la mise à jour d'un nœud dans un GNN par message passing. Quel est l'effet du nombre de couches ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$n_i^{(2)} = \sum_{n_j\in\mathcal N(n_i)}\psi(n_i, n_j, e_{ji})$$

avec $\psi$ un réseau de neurones. Plus de couches ⇒ champ récepteur plus large (voisins de voisins…). Les losses peuvent porter sur les nœuds, les arêtes ou le graphe entier (avec un agrégateur).
</details>

<details>
<summary><b>Question 12 (1 pt) :</b> Quelles étaient les 3 étapes de la classification d'images avant le deep learning ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Extraire un dictionnaire de features (keypoints + descripteurs) ; 2. les agréger (Bag of Words : clustering → vocabulaire de mots visuels, histogrammes) ; 3. un classifieur (ex. SVM) sur cette représentation.
</details>

<details>
<summary><b>Question 13 (1 pt) :</b> Comment un CNN peut-il traiter un signal audio (séparation de sources) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

En passant dans le domaine **temps-fréquence** (spectrogramme), traité comme une image : la séparation devient un problème de **classification supervisée** de chaque case temps-fréquence (au lieu de l'ICA / SVD non supervisée).
</details>
