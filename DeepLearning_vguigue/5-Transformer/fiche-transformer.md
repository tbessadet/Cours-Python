# 📘 Fiche de révision — Architecture Transformer

> Cours de Vincent Guigue (inspiré des supports de B. Piwowarski & T. Gérald) · AgroParisTech – IODAA · 30 diapos (`5-Transf.pdf`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 notes perso

---

## Sommaire

1. [Introduction : le problème de l'agrégation](#1-introduction--le-problème-de-lagrégation)
2. [Architecture : représentation en sous-mots](#2-architecture--représentation-en-sous-mots)
3. [Le bloc d'attention (self-attention)](#3-le-bloc-dattention-self-attention)
4. [Multi-têtes, couches, position, masques](#4-multi-têtes-couches-position-masques)
5. [Trois familles d'architectures](#5-trois-familles-darchitectures)
6. [Pré-entraînement et tâches d'apprentissage](#6-pré-entraînement-et-tâches-dapprentissage)
7. [Exploitation : BERT, fine-tuning, nouvelles modalités](#7-exploitation--bert-fine-tuning-nouvelles-modalités)
8. [Pièges à éviter](#8-pièges-à-éviter)
9. [Quiz auto-test](#9-quiz-auto-test)

---

## 1. Introduction : le problème de l'agrégation

### 1.1 Problème d'agrégation (diapo 2)

- Séquence : $\mathbf s = \{s_1, \dots, s_L\}$ ;
- apprentissage de représentation : $s_i \to \mathbf e_i \in \mathbb R^Z$ ;
- décision sur $\mathbf s$ ⇒ représentation / *mapping* $R(\mathbf s) : \mathbb R^{L\times Z} \mapsto \mathbb R^d$ ;
- \+ fonction de décision $f(R(\mathbf s)) : \mathbb R^d \mapsto \mathcal Y$.

| Agrégation | Principe |
|---|---|
| RNN + dernière couche cachée (ext. : Bi-RNN) | $R(\mathbf s) = h_L$ |
| CNN/RNN + **pooling** | moyenne / max des états |
| CNN/RNN + **attention** | **somme pondérée apprise** |

- $L$ : longueur de la séquence ; $Z$ : dimension des embeddings ; $d$ : dimension de la représentation de la séquence.

### 1.2 Attention & agrégation vs processus de diffusion (diapo 3)

$e_s = \sum_i \alpha_i h_i$, les poids $\alpha_i$ étant calculés à partir d'un **vecteur d'attention** $a_s$.
- **Apprendre à pondérer une représentation localisée** ;
- de nombreuses applications efficaces [2014-2018] ;
- **peu de paramètres / parcimonie** (*sparsity*) ;
- **vecteur d'attention = vecteur *query*** ;
- comment le coder en PyTorch ?

Séance pratique : Q1 tests préliminaires (attention uniforme) ; Q2 attention globale classique ; **Q3 ⇒ reconstruction de *tous* les embeddings par recombinaison = processus de diffusion** (au lieu d'un seul vecteur $e_s$, chaque position reçoit une nouvelle représentation = c'est l'idée du Transformer).

### 1.3 Code pour un *attention pooling* (diapo 4)

*Imaginons chaîner un nouveau module après un RNN :*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, h):                 # h : (B, T, Z) Batch (first) x Time x LatentDim
        scores = self.attn(h)             # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (weights * h).sum(dim=1)   # (B, Z)
        return pooled
```

- Code **très simple**… et différentiation automatique & compatibilité batch (grâce au module `Linear`).

> 🔎 `nn.Linear(dim, 1, bias=False)` **est** le vecteur d'attention $a_s$ : `self.attn(h)` calcule $\langle a_s, h_t\rangle$ pour chaque $t$. Avec du padding, il faudrait masquer les scores (cf. §4.5).

### 1.4 Inspiration : les *memory networks* (question answering) (diapo 5)

Tâche (jeu bAbI) : « Sam walks into the kitchen. Sam picks up an apple. Sam walks into the bedroom. Sam drops the apple. *Q : Where is the apple ?* A : Bedroom ».

- *Comment ancrer le question answering dans des bases de connaissances ?*
- Architecture : les phrases $\{x_i\}$ sont plongées deux fois (Embedding A → mémoires $m_i$ ; Embedding C → sorties $c_i$) ; la question est plongée (Embedding B → $u$) ; **poids** $p_i = \mathrm{Softmax}(u^\top m_i)$ (produit scalaire) ; **somme pondérée** $o = \sum_i p_i c_i$ ; réponse $\hat a = \mathrm{Softmax}(W(o + u))$. Version multi-sauts (*hops*) : on recommence avec $u^{k+1} = u^k + o^k$.
- Résultats : sur « Where is John? », l'attention se concentre (0,98) sur la phrase support « John travelled to the bathroom » ; certaines questions demandent **2 faits** (plusieurs *hops*).

> 💡 C'est déjà le schéma **query** ($u$) / **keys** ($m_i$) / **values** ($c_i$) du Transformer.

### 1.5 Conclusion de l'introduction (diapo 6)

**Limites actuelles des RNN :**
- **dépendances longues** ;
- **représentation globale du message** (indexation continue) ;
- **dépendances ciblées** (résolution de co-référence : à quoi renvoie « it » ?).

---

## 2. Architecture : représentation en sous-mots

### 2.1 Généralités (diapo 7)

Sources pour comprendre (plus simples que l'article original) : J. Alammar (*illustrated transformer*), A. Rush (*annotated transformer*), P. Bloem.

Schéma de référence : **encodeur** (×N : *Multi-Head Attention* → *Add & Norm* → *Feed Forward* → *Add & Norm*) et **décodeur** (×N : *Masked Multi-Head Attention* → *Add & Norm* → *Multi-Head Attention* sur la sortie de l'encodeur → *Add & Norm* → *Feed Forward* → *Add & Norm*) → *Linear* → *Softmax* ; entrées et sorties (décalées à droite) passent par un **embedding + positional encoding**.

> 💡 **Dépasser les images classiques ⇒ ré-implémenter un bloc transformer.**

### 2.2 Représentation en sous-mots (diapo 8)

**Idée générale : diffusion de la représentation + apprentissage de la diffusion** (chaque token « Pierre mange un radis . ll » met à jour sa représentation en regardant les autres, couche après couche).

**Choix de l'échelle :**

| Échelle | + | − |
|---|---|---|
| **Caractère** | (très) petit vocabulaire | pas de sémantique |
| **Mot** | sémantique ++ | vocabulaire énorme ; mots inconnus |
| **Sous-mot** (Sennrich et al. 2016) | compromis | |

Exemple : « les chats sont des félidés » est transformé en `[CLS] _les _chats _sont _des _f éli dés [SEP]`.

---

## 3. Le bloc d'attention (self-attention)

### 3.1 Transformer : agrégation de l'état de l'art (diapo 9)

| RNN | Transformer |
|---|---|
| $h_{t+1} = h_tW_1 + x_{t+1}W_2$ : lecture **séquentielle**, le contexte passe d'état en état | chaque token embedding $v_i$ est recombiné avec **tous** les autres via une **matrice de self-attention** $\alpha_{ij}$ : $v'_i = \sum_j \alpha_{ij}v_j$, puis une couche **fully connected** → $v''_i$ |

- Le bloc reçoit $x_i$ (+ *positional embedding*), calcule $k_i = W^K x_i$, $q_i = W^Q x_i$, $v_i = W^V x_i$, la matrice de self-attention, puis $v'_i = \sum_j \alpha_{ij}v_j$ et une couche dense → $x'_i$.
- On **empile** des blocs (*nb transf. blocks*) : chaque bloc contient plusieurs têtes d'attention (*word cross-attn head*) et produit des $\tilde x_i$ de même dimension.
- Exemple de visualisation (couche 5) : dans « The animal didn't cross the street because it was too tired », le token « it » porte son attention sur « The animal ».

(Vaswani et al. NeurIPS 2017, *Attention is all you need* ; Sutskever et al. 2014.)

### 3.2 Bloc d'attention : idée générale (diapo 10)

1. **Calculer la self-attention entre les tokens** ;
2. **normaliser l'attention** (`softmax`) ;
3. **exploiter l'attention pour recombiner la représentation des tokens**.

⇒ Comment calculer les interactions (= ***self attention***) ⇒ nouvelle représentation de token (= **représentation contextuelle**).

Exemple (matrice d'attention de « [CLS] it looks small outside but quite big inside nice styling too ») : les mots s'attendent surtout à leurs voisins proches.

### 3.3 Bloc d'attention : détail (diapo 11)

**Interactions à travers des vecteurs dérivés de l'embedding :**
1. calculer ***Keys*, *Queries*, *Values*** par **projection linéaire** ($W^Q$, $W^K$, $W^V$).

> 💡 **Note : ce sont les principaux paramètres de l'architecture transformer.**

**Calculer l'attention :**
1. **produit scalaire** entre keys et queries ;
2. **division par la taille des clés/requêtes** : le softmax est trop sensible aux grandes valeurs (**température**) ;
3. **softmax** ⇒ sur la dimension pertinente.

**Exemple travaillé** (« Thinking Machines », $d_k = 64$, $\sqrt{d_k} = 8$), pour le token « Thinking » :

| Étape | « Thinking » ($j=1$) | « Machines » ($j=2$) |
|---|---|---|
| Score $q_1 \cdot k_j$ | 112 | 96 |
| Divisé par 8 ($\sqrt{d_k}$) | 14 | 12 |
| Softmax | 0,88 | 0,12 |
| Softmax × valeur | $0{,}88\,v_1$ | $0{,}12\,v_2$ |
| **Somme** | $z_1 = 0{,}88\,v_1 + 0{,}12\,v_2$ | |

**Nouvelles valeurs** : $v_i^{\ell+1} = \sum_k \alpha_k v_k^\ell$. *Comment revenir à l'espace des embeddings ?* value ↦ embedding : **utiliser une couche fully connected** pour revenir dans l'espace des embeddings (encodeur : *Self-Attention* → $z_1, z_2$ → *Feed Forward Neural Network* (appliqué à chaque position) → $r_1, r_2$ → encodeur suivant).

### 3.4 Parallélisme ⇒ batch de données (diapo 12)

**Tous les tokens à la fois !**

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

$$Z = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- $X \in \mathbb R^{L\times d}$ : embeddings des $L$ tokens (une ligne par token) ;
- $W^Q, W^K \in \mathbb R^{d\times d_k}$, $W^V \in \mathbb R^{d\times d_v}$ : projections apprises ;
- $QK^\top \in \mathbb R^{L\times L}$ : scores de toutes les paires ; softmax **ligne par ligne** (chaque requête distribue 1 sur les clés) ;
- $d_k$ : dimension des clés (la division par $\sqrt{d_k}$ évite la saturation du softmax) ; $Z \in \mathbb R^{L\times d_v}$ : nouvelles représentations.

> 🔎 Pourquoi $\sqrt{d_k}$ : si les composantes de $q$ et $k$ sont de variance 1, $q\cdot k$ a une variance $d_k$ ; diviser par $\sqrt{d_k}$ la ramène à 1, sinon le softmax devient quasi one-hot (gradients nuls).
>
> 💡 Contrairement au RNN, aucun calcul n'attend le précédent : tout est **matriciel et parallélisable** sur GPU (coût en $O(L^2)$ en mémoire).

---

## 4. Multi-têtes, couches, position, masques

### 4.1 Têtes multiples : défi de compatibilité des dimensions (diapo 13)

**Pourquoi une seule interaction ?** On utilise plusieurs **têtes** (*attention head #0, #1…*), chacune avec ses propres $W_h^Q, W_h^K, W_h^V$.

Procédure :
1. phrase d'entrée ; 2. **embedding** de chaque mot ($X$ ; dans les encodeurs suivants, on part directement de la sortie $R$ de l'encodeur précédent) ; 3. **découper en 8 têtes** : multiplier $X$ (ou $R$) par les matrices de poids de chaque tête ; 4. **calculer l'attention** avec les $Q/K/V$ obtenus → $Z_0, \dots, Z_7$ ; 5. **concaténer** les $Z$ puis multiplier par une matrice $W^O$ pour produire la sortie de la couche.

> 💡 Typiquement, les valeurs sont **concaténées** puis projetées pour **reconstruire les embeddings** (chaque tête projette vers une dimension plus faible).

| Modèle | Embedding | Dim. key/query/values | Têtes | Vérification |
|---|---|---|---|---|
| **BERT** | 768 | 64 | 12 | $12 \times 64 = 768$ |
| **ChatGPT** | 12 288 | 128 | 96 | $96 \times 128 = 12\,288$ |

🔎 Formule :

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(Z_0, \dots, Z_{h-1})\,W^O, \qquad Z_i = \mathrm{softmax}\!\left(\frac{XW_i^Q\,(XW_i^K)^\top}{\sqrt{d_k}}\right)XW_i^V$$

- $h$ : nombre de têtes ; $d_k = d/h$ ; $W^O \in \mathbb R^{h\,d_v\times d}$.

### 4.2 Couches multiples (diapo 14)

- **Regarder la connexion résiduelle** (*skip connection*) : chaque sous-bloc (self-attention, feed forward) est suivi de **Add & Normalize** ;
- **la diffusion se fait pas à pas** (couche après couche) ;
- la ***Bertology*** montre que **différentes relations sont extraites à différentes échelles** (couches) :
  - têtes qui « attendent largement », à « next token », à [SEP], aux ponctuations ;
  - l'attention moyenne vers [CLS] est forte dans les premières couches, vers [SEP] au milieu (couches 6–10), vers « . , » dans les dernières ;
  - **tête 8-10** : les **objets directs** attendent leur verbe (86,8 % de précision sur la relation `dobj`) ; **tête 8-11** : les **modificateurs du nom** (déterminants) attendent leur nom (94,3 % sur `det`).

🔎 Sous-bloc encodeur (Post-LN, article original) :

$$X' = \mathrm{LayerNorm}\big(X + \mathrm{MultiHead}(X)\big), \qquad X'' = \mathrm{LayerNorm}\big(X' + \mathrm{FFN}(X')\big), \qquad \mathrm{FFN}(x) = W_2\,\mathrm{ReLU}(W_1x + b_1) + b_2$$

### 4.3 Couches multiples (& langues) (diapo 15)

Wendler et al. 2024, *Do Llamas work in English?* Tâches : traduire du français vers le chinois ; répéter du chinois ; compléter une phrase chinoise (*cloze*). En lisant les couches intermédiaires :
- **identifier la langue** ; **trouver un embedding existant proche** ;
- les couches intermédiaires passent par l'**anglais** (probabilité de « en » élevée vers les couches 20–30) avant que le chinois n'apparaisse dans les dernières couches.

### 4.4 Codage de la position (diapo 16)

**Jusqu'ici, le Transformer ne prend pas la position en compte !!** (l'attention est une somme : permuter les tokens ne change rien). On **ajoute** un *positional encoding* à l'embedding : $x_i \leftarrow x_i \oplus$ (ou $+$) $PE_i$.

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/size}}\right), \qquad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/size}}\right)$$

- $pos$ : position du token ; $i$ : indice dans la dimension latente ; $size$ : dimension latente.

Exemple : « Je suis étudiant » : $PE_1 = (0, 0, 1, 1)$, $PE_2 = (0{,}84;\ 0{,}0001;\ 0{,}54;\ 1)$, $PE_3 = (0{,}91;\ 0{,}0002;\ -0{,}42;\ 1)$, **ajoutés** aux embeddings $x_1, x_2, x_3$.

> 💡 La carte (positions × dimensions) montre des sinusoïdes de fréquences décroissantes : les premières dimensions oscillent vite (position fine), les dernières lentement (position grossière). 🔎 Alternative : **embeddings de position appris** (BERT, ViT).

### 4.5 Masquer le calcul de l'attention (diapos 17 et 20)

**Penser aux batchs** : séquences paddées de longueurs différentes (4, 1, 6, 3).
- les longueurs sont différentes ;
- **attention vers un token PAD $= -\infty \Rightarrow \alpha_{PAD} = 0$** : le padding ne doit pas influencer les autres tokens.

**Contrainte sur l'attention pour améliorer la prédiction** (*causal mask*) : *raw attention weights* × **masque triangulaire** ; la sortie $y_t$ ne peut attendre que $x_1, \dots, x_t$ (pas le futur).

🔎 En pratique, on ajoute au score $QK^\top/\sqrt{d_k}$ une matrice $M$ avec $M_{ij} = 0$ si autorisé, $-\infty$ sinon, **avant** le softmax.

### 4.6 Tokens spécifiques (diapo 18)

*Comment découper les textes ou agréger toute l'information ?* ⇒ tokens spécifiques :
- **CLS** : son vecteur de sortie (*Vec*) agrège la phrase (il attend tous les autres tokens, ex. poids 0,2 / 0,14 / 0,1 / 0,05 / 0,25 / 0,25 / 0,01 sur « [CLS] He has a cute dog [SEP] ») ;
- **SEP** : séparateur de phrases / segments.

---

## 5. Trois familles d'architectures

### 5.1 Différentes architectures (et différents usages) (diapo 19)

| Famille | Exemples | Attention | Usage typique |
|---|---|---|---|
| **Encoder-Decoder** | article original, **T5** | encodeur : tout voit tout ; décodeur : causal + attention croisée vers l'encodeur | traduction, résumé (seq2seq) |
| **Encoder only** | **BERT** | **bidirectionnelle** (non causale) | classification, NER, QA extractive, embeddings |
| **Decoder only** | **GPT** | **causale** (triangulaire) | génération de texte |

Masques (schéma « I am a causal decoder ») : *causal decoder* = triangulaire ; *non-causal decoder* (préfixe) = le préfixe se voit entièrement, la suite est causale ; *encoder-decoder* = encodeur complet, décodeur causal.

---

## 6. Pré-entraînement et tâches d'apprentissage

### 6.1 De Word2Vec au Masked-LM (MLM) (diapo 21)

- **Masquage de tokens** (+ masquage de l'attention pour que le masque n'ait pas d'influence) : BERT ;
- **prédiction du token suivant** : GPT.

Schéma : « the cat [mask] on the mat » + CLS → BERT (encodeur) → **embeddings contextualisés** → MLP sur un token : **prédiction du mot** (classification) ou étiquetage **IOBES** ; MLP sur CLS : **classification au niveau de la phrase**.

### 6.2 *Next sentence* (diapo 22)

- **Apprendre à agréger… à moindre coût** : on cherche à savoir **si une phrase en suit une autre**.

```
Input = [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]
Label = IsNext
Input = [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = NotNext
```

### 6.3 T5 : généraliser l'IA générative (diapo 23)

- **Affiner un modèle génératif pour répondre à différentes tâches… T5 :)** (Colin Raffel).
- Courbe (SQuAD, *exact match*) : progression lente « sans transfer learning », puis forte accélération **avec transfer learning** (2018-2019).
- **Tout est texte → texte**, la tâche est indiquée dans l'entrée : `"translate English to German: That is good."` → `"Das ist gut."` ; `"cola sentence: The course is jumping well."` → `"not acceptable"` ; `"stsb sentence1: … sentence2: …"` → `"3.8"` ; `"summarize: …"` → résumé.

### 6.4 Les ingrédients de ChatGPT (diapos 24-25)

> 🔎 La diapo numérote à partir de 2 ; l'ingrédient 1 est implicitement le **pré-entraînement génératif massif** (GPT / T5, §6.1–6.3).

**2. Suivi de dialogue (*dialogue tracking*)** : **entraînement spécifique** de GPT sur un **corpus de dialogues** → suivi du dialogue, **résolution de co-référence**, **manière de parler**.
- **Données très propres** : générées / validées / classées **par des humains**.

**3. Fine-tuning sur différentes tâches de raisonnement (±) complexes** :
- ***instruction finetuning*** (« Please answer the following question. What is the boiling point of Nitrogen? » → « −320.4F ») ;
- ***chain-of-thought finetuning*** (« Answer the following question by reasoning step-by-step. The cafeteria had 23 apples… » → « …23 − 20 = 3… 3 + 6 = 9 ») ;
- ***multi-task instruction finetuning*** (1,8 k tâches) ⇒ **inférence : généralisation à des tâches non vues** (« Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering. » → raisonnement → « no »).
- Collections de tâches : T0-SF, Muffin, CoT (*Reasoning*), Natural Instructions v2, et évaluation sur MMLU, BBH, TyDiQA, MGSM. Plus le **nombre de tâches de fine-tuning** augmente (9 → 1 836) et plus le modèle est grand (8B → 62B → 540B), plus la performance moyenne sur les tâches réservées augmente ; le CoT aide surtout les grands modèles (Chung et al., JMLR 2024).

---

## 7. Exploitation : BERT, fine-tuning, nouvelles modalités

### 7.1 Visualisation des couches (diapo 26)

**BERT viz** (`github.com/jessevig/bertviz`) : vue de toutes les têtes de toutes les couches.

### 7.2 Exemples d'usage de BERT (diapo 27)

| Tâche | Sortie utilisée | Exemple |
|---|---|---|
| Classification de phrase | MLP sur le **CLS** → +1/−1 | « An iPhone, what a scam! » |
| Étiquetage de tokens (NER, relations) | MLP sur **chaque** token | « JFK died in 1963, he was assassinated in Dallas » → Person, RELATION, date, place |
| Question answering extractif | MLP sur chaque token (début/fin de la réponse) | « How many … have? <SEP> BERT-large … » → empan de la réponse |
| Paire de phrases (même sens ?) | MLP sur le **CLS** de « Phrase 1 <SEP> Phrase 2 » → +1/−1 | paraphrase, implication |

### 7.3 Différentes stratégies de fine-tuning (diapo 28)

| Stratégie | Ce qui est appris | Coût |
|---|---|---|
| **Fine-tuning complet** | tout le réseau (le gradient traverse tous les blocs) | élevé |
| **Adapters** | petits modules **insérés** entre les blocs (le reste est gelé) | faible |
| **Soft-prompting** | des **embeddings de prompt** $x_0$ ajoutés en entrée (le modèle est gelé) | très faible |
| **LoRA** (*Low Rank Approximation*) | une mise à jour de **faible rang** $\Delta W$ des matrices K, Q, V, W (MLP) | faible |

> 🔎 LoRA : $W' = W + \Delta W$ avec $\Delta W = BA$, $B \in \mathbb R^{d\times r}$, $A \in \mathbb R^{r\times d}$, $r \ll d$ : on apprend $2dr$ paramètres au lieu de $d^2$ ; $W$ reste gelée.

### 7.4 Attention à la nouveauté : les améliorations ne sont pas garanties (diapo 29)

- **Nouvelles façons de voir l'architecture transformer** (Anthropic, *transformer circuits*) : un **flux résiduel** (*residual stream*) traverse le modèle (tokens → *embed* → $x_0$ → … → *unembed* → *logits*) ; **chaque couche « lit »** le flux par une projection linéaire et **« écrit »** en ajoutant une projection de ses résultats (les têtes $h_0, h_1, \dots$ et le MLP $m$ s'additionnent : $x_{i+1} = x_i + \sum_h h(x_i)$, $x_{i+2} = x_{i+1} + m(x_{i+1})$).
- **Séries temporelles** : *Are Transformers effective for time series forecasting?* (Zeng et al. 2022) : des modèles linéaires simples font souvent aussi bien.

### 7.5 Apprendre des tokens… pour toutes les modalités (diapo 30)

- **Séries temporelles** : découper la série en **tokens** (pas de temps / fenêtres, éventuellement multivariés) → attention sur les tokens temporels → FFN.
- **Images** : **Vision Transformer (ViT)** : *patchs* aplatis → projection linéaire → **patch + position embedding** (+ token `[class]` appris) → encodeur Transformer (*L ×* : Norm → Multi-Head Attention → + ; Norm → MLP → +) → MLP head → classe. Les **embeddings de position appris** retrouvent la structure 2D (similarité cosinus élevée entre patchs de même ligne / colonne).
- …

> 💡 ⇒ **pré-entraînement, masquage, data-augmentation…** : la recette se transpose à toutes les modalités dès qu'on sait **tokeniser**.

---

## 8. Pièges à éviter

> 🚩 **Oublier le positional encoding** : sans lui, le Transformer traite un **sac** de tokens (permutation invariant).

> 🚩 **Oublier le masque de padding** : les tokens PAD reçoivent de l'attention et polluent les représentations ; score à $-\infty$ ⇒ $\alpha_{PAD} = 0$.

> 🚩 **Oublier le masque causal en génération** (decoder) : le modèle « triche » en regardant le futur à l'entraînement et s'effondre à l'inférence.

> ⚠️ **Oublier la division par $\sqrt{d_k}$** : softmax saturé (température trop basse), gradients quasi nuls.

> ⚠️ **Softmax sur la mauvaise dimension** : chaque **requête** doit distribuer 1 sur les **clés** (softmax par ligne de $QK^\top$).

> ⚠️ **Dimensions multi-têtes** : $h \times d_k = d$ (BERT : $12\times64 = 768$) ; la concaténation est suivie d'une projection $W^O$.

> ⚠️ **Coût quadratique** en la longueur ($L\times L$ scores) : tronquer les textes, limiter la longueur max.

> ⚠️ **Encoder vs decoder** : BERT (bidirectionnel) pour comprendre/classer, GPT (causal) pour générer ; ne pas générer avec un encodeur.

> ⚠️ **Les améliorations ne sont pas garanties** (ex. séries temporelles) : comparer à des baselines simples.

---

## 9. Quiz auto-test

<details>
<summary><b>Question 1 (2 pts) :</b> Quelles sont les trois façons d'agréger une séquence en un vecteur, et quelles limites des RNN motivent le Transformer ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

RNN + dernier état ; CNN/RNN + pooling ; CNN/RNN + attention (somme pondérée apprise). Limites des RNN : dépendances longues, représentation globale du message, dépendances ciblées (co-référence), et calcul séquentiel non parallélisable.
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Écrire la formule de la self-attention en version matricielle et définir Q, K, V. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V,\qquad Z = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$X$ : embeddings des tokens (une ligne par token) ; $W^Q, W^K, W^V$ : projections apprises (les principaux paramètres) ; le softmax est appliqué ligne par ligne ; chaque ligne de $Z$ est la nouvelle représentation contextuelle d'un token.
</details>

<details>
<summary><b>Question 3 (2 pts) :</b> Exemple « Thinking Machines » : scores 112 et 96, d_k = 64. Calculer les poids d'attention. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Division par $\sqrt{64} = 8$ : 14 et 12. Softmax : $e^{14}/(e^{14}+e^{12}) = 1/(1+e^{-2}) \approx 0{,}88$ et $0{,}12$. Sortie : $z_1 = 0{,}88\,v_1 + 0{,}12\,v_2$.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> Pourquoi diviser par √d_k ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le produit scalaire de vecteurs de dimension $d_k$ a une variance qui croît avec $d_k$ ; de grandes valeurs rendent le softmax trop « piqué » (quasi one-hot, gradients nuls). Diviser par $\sqrt{d_k}$ joue le rôle d'une température qui ramène les scores à une échelle raisonnable.
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Qu'est-ce que l'attention multi-têtes ? Vérifier la cohérence des dimensions de BERT et de ChatGPT. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Plusieurs têtes calculent chacune leur attention avec leurs propres $W_h^Q, W_h^K, W_h^V$ (projetant vers une dimension plus petite), ce qui capte plusieurs types de relations. Les sorties sont concaténées puis projetées par $W^O$. BERT : 12 têtes × 64 = 768 ; ChatGPT : 96 têtes × 128 = 12 288 (= dimension des embeddings).
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Pourquoi faut-il un codage de position ? Donner la formule sinusoïdale. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

L'attention est une somme pondérée : sans information de position, permuter les tokens ne change rien. On ajoute à chaque embedding :

$$PE(pos,2i) = \sin\!\left(\frac{pos}{10000^{2i/size}}\right),\qquad PE(pos,2i+1) = \cos\!\left(\frac{pos}{10000^{2i/size}}\right)$$

(ou des embeddings de position appris).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Quels sont les deux types de masques d'attention et leur rôle ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Masque de **padding** : score $-\infty$ vers les tokens PAD ⇒ $\alpha_{PAD} = 0$, le padding n'influence pas les autres tokens. Masque **causal** (triangulaire) : un token ne peut attendre que les positions précédentes, indispensable pour la génération (decoder / GPT).
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Comparer encoder-decoder, encoder only et decoder only (exemples, attention, usage). <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Encoder-decoder (article original, T5) : encodeur bidirectionnel + décodeur causal avec attention croisée ; seq2seq (traduction, résumé). Encoder only (BERT) : attention bidirectionnelle ; classification, NER, QA, embeddings. Decoder only (GPT) : attention causale ; génération.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Quelles sont les tâches de pré-entraînement de BERT ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**Masked LM** : prédire des tokens masqués à partir du contexte des deux côtés ; **Next Sentence Prediction** : à partir du CLS, dire si la phrase B suit la phrase A (IsNext / NotNext). GPT est pré-entraîné par prédiction du token suivant.
</details>

<details>
<summary><b>Question 10 (2 pts) :</b> Quel est le principe de T5 ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Toutes les tâches sont formulées en **texte → texte**, la tâche étant décrite dans l'entrée (« translate English to German: … », « summarize: … », « stsb sentence1: … » → « 3.8 ») : un seul modèle génératif encodeur-décodeur, fine-tuné/transféré sur de nombreuses tâches.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Quels ingrédients au-delà du pré-entraînement le cours cite-t-il pour ChatGPT ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Entraînement spécifique sur des **dialogues** (données très propres, générées/validées/classées par des humains) pour le suivi du dialogue et la co-référence ; **fine-tuning d'instructions**, **chain-of-thought** et **multi-tâches** (1,8 k tâches) qui permettent de généraliser à des tâches non vues.
</details>

<details>
<summary><b>Question 12 (3 pts) :</b> Citer quatre stratégies de fine-tuning et ce qu'elles apprennent. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Fine-tuning complet (tous les poids) ; **adapters** (petits modules insérés entre les blocs, reste gelé) ; **soft-prompting** (embeddings de prompt appris en entrée, modèle gelé) ; **LoRA** (mise à jour de faible rang $\Delta W = BA$ des matrices K, Q, V, MLP).
</details>

<details>
<summary><b>Question 13 (2 pts) :</b> Comment utiliser BERT pour classer une phrase, pour la NER et pour juger si deux phrases ont le même sens ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Classification : MLP sur la représentation contextuelle du **CLS**. NER : MLP sur la représentation de **chaque** token. Paire de phrases : entrée « Phrase 1 [SEP] Phrase 2 », MLP sur le CLS → +1/−1.
</details>

<details>
<summary><b>Question 14 (1 pt) :</b> Comment le ViT applique-t-il le Transformer aux images ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

L'image est découpée en patchs aplatis, projetés linéairement (= tokens), auxquels on ajoute un embedding de position appris et un token [class] ; le tout passe dans un encodeur Transformer et une MLP head classe à partir du token [class].
</details>
