# 🖼️ Fiche de cours — Réseaux convolutionnels pour l'image

> Source : `2-cnn.pdf` (diapos 1–25). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours) · 📝 mes notes (aucune fournie pour ce cours).

## Sommaire
1. [Introduction : pourquoi les MLP ne suffisent pas](#1-introduction--pourquoi-les-mlp-ne-suffisent-pas)
2. [Avant le deep learning](#2-avant-le-deep-learning)
3. [Convolution 1D puis 2D](#3-convolution-1d-puis-2d)
4. [Pooling](#4-pooling)
5. [Couche convolutionnelle usuelle et paramètres](#5-couche-convolutionnelle-usuelle-et-paramètres)
6. [Dimensions d'entrée/sortie](#6-dimensions-dentréesortie)
7. [Fine-tuning](#7-fine-tuning)
8. [Architectures en vision](#8-architectures-en-vision)
9. [Autres usages : reconstruction, problèmes et métriques, transformers, signal](#9-autres-usages--reconstruction-problèmes-et-métriques-transformers-signal)
10. [Introspection d'un CNN](#10-introspection-dun-cnn)
11. [Quiz d'auto-test](#11-quiz-dauto-test)
12. [🚩 Pièges à éviter](#12--pièges-à-éviter)

---

## 1. Introduction : pourquoi les MLP ne suffisent pas

- Une image = **grille de pixels** → sémantique très pauvre.
- **Fully-connected (MLP)** : *une entrée = un rôle propre* → dépendance à la **position du pixel**.
- > 💡 Dans une image, **l'absolu n'a pas d'importance, le relatif est bien plus important** (mais pas seulement). Les **invariants** sont très importants car un objet peut prendre de multiples formes (position, taille, orientation…).

---

## 2. Avant le deep learning

Chaîne en trois temps :
1. **Extraire un dictionnaire de features** (points-clés/*keypoints*, descripteurs, sur une grille ou par détection) ;
2. **Agréger** les features (représentation type **Bag of Words** : clustering → vocabulaire → « mots visuels ») ;
3. **Classifieur** sur cette représentation.

> 💡 Le deep learning (CNN) apprend **conjointement** les descripteurs et le classifieur.

---

## 3. Convolution 1D puis 2D

**Réseaux convolutifs** : un **filtre d'analyse glissant** sur l'image :
- **peu de paramètres** (poids partagés) ;
- **apprentissage des motifs** à extraire (par rétropropagation, avec les coefficients) ;
- **agrégation progressive des échelles**.

### Convolution 1D (pour bien comprendre)
Schéma : **fenêtre glissante** (*sliding filter*) → pooling max/average → *fully connected*.
- Fenêtre glissante, peu de paramètres ;
- apprentissage des **motifs à détecter** ;
- **invariance / dépendance à la localisation** (selon l'agrégation) ;
- paramètres : **largeur (kernel) / stride / padding**.
On empile plusieurs filtres (*sliding filters*), puis pooling, puis nouveaux filtres…

**Convolutions dilatées** (*dilated convolutions*, utiles en segmentation) : la dilatation écarte les points du noyau → le **champ réceptif** croît exponentiellement avec la profondeur. Exemple de la diapo (noyau de taille 3) : dilatation 1 → champ réceptif **3** ; dilatation 2 → **7** ; dilatation 4 → **15**.

### Convolution 2D
Le filtre est une petite fenêtre (ex. 5×5) qui glisse en 2D ; sa sortie en chaque position est une somme pondérée locale ; un filtre = une carte de caractéristiques.

---

## 4. Pooling

**Pooling** (ou *subsampling*) : **réduire la dimensionnalité de sortie**.
| Type | Opération |
|---|---|
| **Max pooling** | prend le **max** sur une fenêtre |
| **Average pooling** | fait la **moyenne** |
| **Sum pooling** | la **somme** |

---

## 5. Couche convolutionnelle usuelle et paramètres

**Couche usuelle** : *sliding filters* → activation → *max/average pooling* → (répétition) → *fully connected*.

*Exemple de la diapo* : reconnaissance de caractères.

### Récapitulatif des paramètres (`nn.Conv2d`)
| Paramètre | Rôle |
|---|---|
| `in_channels` | nombre de canaux d'entrée |
| `out_channels` | nombre de filtres (= canaux de sortie) |
| `kernel_size` | taille du noyau |
| `stride` | pas de déplacement de la fenêtre |
| `padding` | bord de zéros ajouté |
| `dilation` | écart entre les points du noyau |

```python
class ConvNet(nn.Module):
    def __init__(self, pooling=nn.MaxPool2d):       # argument sur le choix du pooling
        super().__init__()
        # Convolution 5x5, 16 filtres
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=5, stride=1, padding=0, dilation=1)
        # Max pooling 3x3
        self.pool1 = pooling(kernel_size=3, stride=1)
        ...
```

---

## 6. Dimensions d'entrée/sortie

Formule donnée (taille de sortie **par canal**), hypothèse : entrée carrée de côté $W$, noyau $K$, padding $P$, stride $S$ :

$$\text{sortie}=\left\lfloor\frac{W-K+2P}{S}\right\rfloor+1$$

- $W$ : taille du volume d'entrée ; $K$ : taille du noyau ; $P$ : padding ; $S$ : stride.
- 🔎 Avec dilatation $D$ : remplacer $K$ par $D(K-1)+1$. La partie entière (`⌊·⌋`) vaut quand la division ne tombe pas juste.
- Le nombre de canaux de sortie = `out_channels`.

🔎 *Exemple* (MNIST 28×28, noyau 5, pooling 3 stride 1, deux fois) : $28\to24\to22\to18\to16$, soit $16\times16\times16$ valeurs à aplatir avant le classifieur.

---

## 7. Fine-tuning

| Approche | Remarques |
|---|---|
| **Apprendre un réseau from scratch** | OK, bons descripteurs… mais **nécessite beaucoup de données** et la performance n'est (souvent) pas optimale. |
| **Fine-tuner un réseau** | Question : **combien de couches garder / ré-entraîner ?** Faut-il tout fine-tuner ? Risque : **oubli catastrophique** (*catastrophic forgetting*). |

> 💡 Réflexe : partir d'un réseau pré-entraîné (ImageNet) et n'ajuster que les dernières couches (ou tout, avec un petit *learning rate*).

---

## 8. Architectures en vision

- **AlexNet** : convolutions 11×11, 5×5, 3×3 ; **max-pooling**, activations **ReLU** ; **Dropout** et **data augmentation**.
- **Deep CNN / VGG / ResNet** : réseaux plus profonds ; **ResNet** ajoute des connexions résiduelles (la sortie d'un bloc s'ajoute à son entrée) pour entraîner de très profonds réseaux.
- Il existe **un grand nombre d'architectures** disponibles (`torchvision.models`).

---

## 9. Autres usages : reconstruction, problèmes et métriques, transformers, signal

### Reconstruction d'image / génération
- **Transfert de style** (*A neural algorithm of artistic style*) : générer des images en combinant **contenu** et **style** à partir d'un CNN entraîné de façon discriminative. La génération = **problème inverse sur le CNN**. 🔎 *Complément :* on part d'une image de bruit blanc $x$ et on fait une **descente de gradient sur l'image** pour que ses features de **contenu** (une couche) et de **style** (statistiques sur plusieurs couches) correspondent à celles des images de référence.

### Différents problèmes, modèles et métriques
| Problème | Entrée → sortie | Modèles | Métriques |
|---|---|---|---|
| **Classification** | $I=\mathrm{RGB}^{d\times d}\to C$ (1000 classes ImageNet) | VGG, ResNet… | accuracy, précision, rappel |
| **Détection** | $I\to C\times B$ avec $B=(pos_x,pos_y,\text{largeur},\text{hauteur})$ : **un problème de régression (!)** | YOLO, R-CNN, fast-R-CNN | **aire géométrique d'intersection** (IoU) |
| **Segmentation** (au pixel) | $I\to I$ | U-Net | (comment encoder/décoder l'image ?) |

À retenir : « *think about transfer learning!* » pour passer à une classification multi-classe sur son propre jeu ; l'U-Net a des spécificités par rapport aux auto-encodeurs.

### La vague Transformer
Utiliser l'architecture **Transformer** pour l'image : quelles différences / complémentarités avec les CNN ? (cf. fiche Transformers).

### CNN et représentation temps-fréquence
Exemple de la **séparation de sources** (grands progrès depuis 5 ans). Problème d'origine : ICA (analyse en composantes indépendantes), SVD (non supervisé) dans le domaine temporel ou temps-fréquence. **Nouveau problème supervisé** : un problème de classification dans le domaine temps-fréquence ; une architecture CNN dépasse les modèles précédents (BLSTM) sur des mélanges simulés et réels.

---

## 10. Introspection d'un CNN

**But** : comprendre la classification — quelle **région** de l'image active la classification, quels **filtres** sont les plus importants…
- Les filtres sont de **plus en plus abstraits** (mots visuels élémentaires) ;
- les couches **conservent l'information topologique**.

### Occlusion
Mesurer la **sensibilité de la classe détectée aux occlusions** : on masque une zone de l'image et on observe la baisse du score de la classe.

### Carte de saillance (*saliency map*)
Calculée en prenant le **gradient de la sortie par rapport à l'image d'entrée** : elle met en avant les **pixels auxquels la classification est la plus sensible**.

### Class Activation Map (CAM)
- **Global Average Pooling** (1 filtre = 1 feature) + **pondération** ;
- **combinaison linéaire** ;
- **agrégation pondérée des filtres** pour indiquer les **régions d'intérêt**.

$$\mathrm{CAM}_c(i,j)=\sum_k w^c_k\,A_k(i,j)$$

- 🔎 $A_k(i,j)$ : sortie du filtre $k$ de la dernière couche conv. en $(i,j)$ ; $w^c_k$ : poids du classifieur linéaire (après GAP) reliant le filtre $k$ à la classe $c$.

### LIME (*Local Interpretable Model-Agnostic Explanations*)
1. **Échantillonnage** (perturbations de l'entrée, pondérées par la similarité avec l'exemple) ;
2. **modèle linéaire local** (sur les données pondérées) ;
3. **interprétation des coefficients linéaires**.

---

## 11. Quiz d'auto-test

> Barème indicatif sur 25 : Q1, Q2, Q4 (4 pts) · Q3, Q5, Q6 (3 pts) · Q7, Q8, Q9 (2 pts) · Q10 (1 pt).

<details>
<summary><b>Question 1 (4 pts) :</b> Pourquoi un MLP est-il mal adapté aux images, et que apporte la convolution ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Dans un MLP chaque pixel a un rôle propre (dépendance à la position) et le nombre de paramètres explose. La convolution utilise un **filtre glissant à poids partagés** : peu de paramètres, motifs appris, invariance à la translation, agrégation progressive des échelles.
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Donner la formule de la taille de sortie d'une convolution et l'appliquer à W=28, K=5, P=0, S=1. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\left\lfloor\frac{W-K+2P}{S}\right\rfloor+1=\left\lfloor\frac{28-5+0}{1}\right\rfloor+1=24$$

($W$ : entrée, $K$ : noyau, $P$ : padding, $S$ : stride.)
</details>

<details>
<summary><b>Question 3 (3 pts) :</b> Rôle des paramètres `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
`in_channels` : canaux d'entrée ; `out_channels` : nombre de filtres ; `kernel_size` : taille du noyau ; `stride` : pas de glissement ; `padding` : zéros ajoutés autour ; `dilation` : espacement des points du noyau (élargit le champ réceptif sans paramètres en plus).
</details>

<details>
<summary><b>Question 4 (4 pts) :</b> Comment fonctionne une CAM et que faut-il changer dans le réseau ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
On remplace le classifieur par **GAP → couche linéaire** (le réseau est ré-entraîné). La carte d'une classe $c$ est la somme pondérée des sorties des filtres de la dernière couche conv. : $\sum_k w^c_k A_k(i,j)$. Elle indique les **régions** qui ont fait réagir les filtres utiles à la classe.
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Saillance vs occlusion ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La **saillance** utilise le gradient de la sortie par rapport à l'image (pixels sensibles). L'**occlusion** masque physiquement une zone et mesure la chute du score de la classe.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Fine-tuning : quelles questions pratiques et quel risque ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Combien de couches garder/ré-entraîner ? Faut-il tout fine-tuner ? Risque d'**oubli catastrophique**. Un réseau appris de zéro demande beaucoup de données et n'est souvent pas optimal.
</details>

<details>
<summary><b>Question 7 (2 pts) :</b> Pourquoi la détection d'objets est-elle « un problème de régression » ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Parce que la sortie inclut une boîte $B=(pos_x,pos_y,\text{largeur},\text{hauteur})$ de valeurs continues ; la métrique est l'aire géométrique d'intersection.
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Types de pooling ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Max (max sur la fenêtre), average (moyenne), sum (somme). Ils réduisent la dimension de sortie.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Quel est l'intérêt des convolutions dilatées ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Augmenter le **champ réceptif** (3 → 7 → 15 avec dilatations 1, 2, 4) sans augmenter le nombre de paramètres ; utile en segmentation.
</details>

<details>
<summary><b>Question 10 (1 pt) :</b> Que fait LIME ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Échantillonne autour d'un exemple, ajuste un modèle linéaire local (pondéré par la proximité) et interprète ses coefficients.
</details>

---

## 12. 🚩 Pièges à éviter

- **Taille de sortie** mal calculée (oublier le padding, la partie entière, le stride).
- **Aplatir** les cartes avant le classifieur (dimension d'entrée du premier `Linear`).
- **Fine-tuner sans précaution** : petit *learning rate*, oubli catastrophique.
- **Saillance** : calculée sur la probabilité (softmax) de la classe, pas sur la loss.
- Les **filtres** seuls ne s'interprètent pas : combiner avec saillance/CAM/occlusion.
