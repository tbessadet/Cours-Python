# 🧠 Fiche de cours — Architecture Transformer

> Source : `5-Transf.pdf` (51 pages, 24 diapos numérotées). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours, pour l'autonomie) · 📝 mes notes (aucune fournie pour ce cours). Les références historiques ont été omises.

## Sommaire
1. [Introduction : le problème d'agrégation](#1-introduction--le-problème-dagrégation)
2. [Attention et agrégation vs processus de diffusion](#2-attention-et-agrégation-vs-processus-de-diffusion)
3. [Inspiration : réseaux à mémoire (question-réponse)](#3-inspiration--réseaux-à-mémoire-question-réponse)
4. [Limites des RNN](#4-limites-des-rnn)
5. [Architecture : vue générale et sous-mots](#5-architecture--vue-générale-et-sous-mots)
6. [Le bloc d'attention](#6-le-bloc-dattention)
7. [Parallélisme, multi-têtes, multi-couches](#7-parallélisme-multi-têtes-multi-couches)
8. [Codage de position](#8-codage-de-position)
9. [Masques, tokens spéciaux, trois familles d'architectures](#9-masques-tokens-spéciaux-trois-familles-darchitectures)
10. [Pré-entraînement et tâches d'entraînement](#10-pré-entraînement-et-tâches-dentraînement)
11. [Exploitation : visualisation, exemples BERT, stratégies de fine-tuning](#11-exploitation--visualisation-exemples-bert-stratégies-de-fine-tuning)
12. [Prudence face à la nouveauté](#12-prudence-face-à-la-nouveauté)
13. [Quiz d'auto-test](#13-quiz-dauto-test)
14. [🚩 Pièges à éviter](#14--pièges-à-éviter)

---

## 1. Introduction : le problème d'agrégation

**Séquence** $s=\{s_1,\dots,s_L\}$. **Apprentissage de représentation** : $s_i\mapsto e_i\in\mathbb R^Z$.
**Décision sur $s$** : une **représentation / application** $R(s):\mathbb R^{L\times Z}\to\mathbb R^d$ puis une **fonction de décision** $f(R(s)):\mathbb R^d\to\mathcal Y$.

Choix possibles pour agréger :
- **RNN + dernier état caché** (extension : Bi-RNN) ;
- **CNN/RNN + pooling** ;
- **CNN/RNN + attention = somme pondérée apprise**.

---

## 2. Attention et agrégation vs processus de diffusion

- **Attention** : *apprendre à pondérer des représentations localisées* ; beaucoup d'applications efficaces ; **peu de paramètres** ; **vecteur d'attention = vecteur de requête** ($h_1,\dots,h_m\to a_s\to$ somme pondérée).
- **TP** : Q1 attention uniforme ; Q2 attention globale classique ; **Q3 : reconstruction de tous les embeddings par recombinaison = processus de diffusion** (chaque élément est réécrit comme combinaison pondérée des autres).

> 💡 L'idée du Transformer : appliquer cette recombinaison **à tous les tokens, entre eux** (self-attention).

---

## 3. Inspiration : réseaux à mémoire (question-réponse)

- Tâches synthétiques de question-réponse (bAbI) : une série d'**énoncés**, puis une **question** dont la réponse est typiquement un mot (ex. *Sam picks up an apple… Where is the apple? → bedroom*). Seule une partie des énoncés est utile ; les autres sont des **distracteurs**. Le modèle doit **déduire lui-même** les énoncés pertinents.
- **Architecture (MemN2N)** : chaque énoncé $x_i$ → mémoire d'entrée $m_i$ (embedding A) et mémoire de sortie $c_i$ (embedding C) ; la **question** $q\to u$ (embedding B) ; **poids d'attention** $p_i=\mathrm{Softmax}(u^\top m_i)$ (produit scalaire) ; sortie $o=\sum_ip_ic_i$ (somme pondérée) ; prédiction $\hat a=\mathrm{Softmax}\big(W(o+u)\big)$, entraînée par **cross-entropie**. Version à **plusieurs « hops »** (couches empilées) pour des raisonnements en plusieurs étapes (le modèle fixe son attention sur le bon fait à chaque saut).
- Question posée : *comment ancrer la question-réponse dans des bases de connaissances ?*

---

## 4. Limites des RNN

- **Dépendances longues** ;
- **Représentation globale du message** (**indexation continue**) ;
- **Dépendances ciblées** : **résolution de coréférences**.

⇒ Motivation : un mécanisme où **tout token peut accéder directement à tout autre**.

---

## 5. Architecture : vue générale et sous-mots

**Pour comprendre les Transformers** (plus faciles que l'article original) : *The Illustrated Transformer* (J. Alammar), *The Annotated Transformer* (A. Rush), *Transformers from scratch* (P. Bloem). Objectif : dépasser les images classiques → **réimplémenter un bloc Transformer**.

### Représentation en sous-mots (*subword*)
Idée générale : **diffusion de la représentation + apprentissage de cette diffusion**. **Choix de l'échelle** :
| Échelle | Avantages | Inconvénients |
|---|---|---|
| **Caractère** | (très) petit vocabulaire | pas de sémantique |
| **Mot** | sémantique ++ | **vocabulaire énorme**, **mots inconnus** |
| **Sous-mot** | compromis des deux | — |

### Le bloc d'attention : idée générale
1. Calculer la **self-attention entre tokens** (*comment calculer les interactions*) ;
2. **Normaliser** l'attention (softmax) ;
3. **Exploiter** l'attention pour **recombiner** la représentation des tokens (*nouvelle représentation = représentation contextuelle*).

---

## 6. Le bloc d'attention

**Étape 1 — Clés, requêtes, valeurs** par **projections linéaires** de l'embedding (ce sont les **principaux paramètres** du Transformer). Hypothèse : $x_i\in\mathbb R^{d}$ embedding du token $i$.

$$q_i=W_Qx_i,\qquad k_i=W_Kx_i,\qquad v_i=W_Vx_i$$

- $q_i$ (*query*) : « ce que le token $i$ cherche » ; $k_j$ (*key*) : « ce que propose $j$ » ; $v_j$ (*value*) : « ce que $j$ transmet ».

**Étape 2 — Calcul de l'attention** :
1. **Produit scalaire** requête/clé : $q_i\cdot k_j$ ;
2. **Division par la taille des clés/requêtes** ($\sqrt{d_k}$) : le **softmax est trop sensible aux grandes valeurs** (**température**) ;
3. **Softmax** sur la bonne dimension (sur les clés $j$).

$$\alpha_{ij}=\frac{\exp\!\big(q_i\cdot k_j/\sqrt{d_k}\big)}{\sum_{j'}\exp\!\big(q_i\cdot k_{j'}/\sqrt{d_k}\big)}$$

*Exemple de la diapo* (tokens « Thinking », « Machines ») : $q_1\cdot k_1=112$ et $q_1\cdot k_2=96$ ; division par $8=\sqrt{64}$ → $14$ et $12$ ; softmax → **0,88** et **0,12**.

**Étape 3 — Nouvelles valeurs** (somme pondérée des valeurs) :
$$v^{\ell+1}_i=\sum_k\alpha_{ik}\,v^{\ell}_k$$
**Comment revenir à l'espace des embeddings ?** *value ↦ embedding* : on utilise un **réseau fully-connected** (feed-forward) pour projeter vers l'espace d'embedding.

> 🔎 **Forme matricielle** (hypothèses : $Q,K\in\mathbb R^{L\times d_k}$, $V\in\mathbb R^{L\times d_v}$ ; une ligne par token) :
> $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$
> Chaque **ligne** de la matrice $L\times L$ somme à 1 ; le coût est en $O(L^2)$.

---

## 7. Parallélisme, multi-têtes, multi-couches

- **Parallélisme sur le batch** : **tous les tokens d'un coup** (contrairement au RNN, séquentiel).
- **Multi-têtes** : *pourquoi une seule interaction ?* Chaque tête a ses propres projections $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$ ; les **valeurs sont concaténées puis projetées** pour reconstruire l'embedding (**défi de compatibilité des dimensions**). Exemples de la diapo :
| Modèle | Embedding | $k/q/v$ par tête | Nombre de têtes |
|---|---|---|---|
| **BERT** | 768 | 64 | 12 ($12\times64=768$) |
| **ChatGPT** (modèle de la diapo) | 12 288 | 128 | 96 ($96\times128=12\,288$) |

- **Multi-couches** : attention à la **connexion résiduelle (*skip connection*)** ; **la diffusion se fait pas à pas** ; la « **BERTologie** » montre que **différentes relations sont extraites à différentes échelles** : certaines têtes « regardent largement », d'autres le **token suivant**, le `[SEP]`, les **points**… Exemples de têtes spécialisées en syntaxe : *objets directs → verbe* (≈ 87 % d'accuracy pour la relation dobj), *déterminants → nom* (≈ 94 % pour det), *pronoms possessifs → tête du groupe nominal* (≈ 80 %), *auxiliaires passifs → verbe* (≈ 82 %).

> 🔎 **Bloc complet** d'un encodeur : self-attention multi-têtes → **Add & Norm** (résiduelle + LayerNorm) → **MLP position-wise** → **Add & Norm** ; on **empile** $N$ blocs (pas les mêmes poids à chaque couche).

---

## 8. Codage de position

> ⚠️ **Jusqu'ici, le Transformer ne prend pas la position en compte !** La self-attention ne dépend que du contenu : sans position, « le chat mange la souris » ≈ « la souris mange le chat » (sac de mots).

**Solution** : ajouter à chaque embedding de token $x_i$ un **embedding de position** (*positional encoding*) de même dimension (schéma : *the cat sat on the mat* + vecteurs de position additionnés aux embeddings, avant self-attention et fully connected).

🔎 *Complément* — encodage sinusoïdal (hypothèse : $p$ = position, $i$ = indice de paire de dimensions, $d$ = dimension du modèle) :
$$PE_{(p,2i)}=\sin\!\Big(\frac{p}{10000^{2i/d}}\Big),\qquad PE_{(p,2i+1)}=\cos\!\Big(\frac{p}{10000^{2i/d}}\Big)$$
Ou **embeddings de position appris** (`nn.Embedding(max_len, d)`).

---

## 9. Masques, tokens spéciaux, trois familles d'architectures

### Masquage du calcul d'attention
- **Penser aux batchs** : les **longueurs sont différentes** ⇒ on complète par du **padding** ; l'attention vers un token `PAD` doit valoir $0$, donc on met son **score à $-\infty$ avant le softmax** ($e^{-\infty}=0$).
- **Contrainte sur l'attention pour améliorer la prédiction** : **masque causal** (un token ne voit pas les suivants), indispensable pour la génération (décodeur).

### Tokens spéciaux
Comment découper les textes ou **agréger toute l'information** ? On introduit des tokens spéciaux : **`[CLS]`** (représentation agrégée de la séquence, pour la classification) et **`[SEP]`** (séparateur de segments/phrases).

### Trois grandes architectures (et usages)
| Famille | Exemples | Usage typique |
|---|---|---|
| **Encodeur-décodeur** | article original, T5, … | traduction, résumé (seq2seq) |
| **Encodeur seul** | **BERT** | représentation, classification, annotation |
| **Décodeur seul** | **GPT** | génération (prédiction du token suivant) |

---

## 10. Pré-entraînement et tâches d'entraînement

**De word2vec au MLM** : prédiction de mots à partir du contexte.
- **Masked Language Modeling (MLM)** : on **masque** des mots (`the cat [MASK] on the mat`) et le modèle les **prédit** (classification sur le vocabulaire) à partir des **embeddings contextualisés** ;
- **Prédiction du token suivant** (GPT) : modèle causal.
- **Next sentence** : apprendre à **agréger de manière peu coûteuse** (prédire si deux phrases se suivent), via le token `[CLS]`.

Sorties exploitables : *word prediction* (classification de mots), *IOBES* (annotation de séquence, par token), *classification de phrase* (via `[CLS]`), chacune par un **MLP** au-dessus des embeddings contextualisés du BERT (encodeur).

**T5 : généraliser l'IA générative** — **raffiner un modèle génératif pour répondre à différentes tâches** en les formulant toutes comme du **texte-vers-texte**. 🔎 *Complément :* un préfixe indique la tâche (traduction, résumé, classification, similarité…) et la réponse est générée sous forme de texte.

---

## 11. Exploitation : visualisation, exemples BERT, stratégies de fine-tuning

- **Visualisation des couches** : outil *bertviz* (liens d'attention têtes/couches).
- **Exemples d'usage de BERT** (encodeur + `[CLS]` + MLP) :
  1. **Analyse de sentiment** : *« An iPhone, what a scam! »* → prédiction +1/−1 via `[CLS]` ;
  2. **Extraction d'entités / relations** : *« JFK died in 1963, he was assassinated in Dallas »* → étiquettes PERSON, DATE, PLACE par token ;
  3. **Question-réponse** : *question* `<SEP>` *contexte* → l'extrait de réponse ;
  4. **Similarité de phrases** : *Phrase 1* `<SEP>` *Phrase 2* → « même sens ? » (+1/−1).

### Différentes stratégies de fine-tuning
| Stratégie | Principe |
|---|---|
| **Fine-tuning complet** | on ré-entraîne tout le modèle + MLP de tâche |
| **Adapters** | on insère de **petits modules** entraînables entre les blocs ; le reste est gelé |
| **Soft-prompting** | on apprend des **vecteurs de prompt** (embeddings « virtuels ») ; le modèle est gelé |
| **LoRA** (*Low-Rank Approximation*) | on apprend une **mise à jour de rang faible** des matrices ($K,Q,V,W$ des MLP) ; le modèle de base est gelé |

> 🔎 LoRA : $W'=W+BA$ avec $B\in\mathbb R^{d\times r}$, $A\in\mathbb R^{r\times d}$, $r\ll d$ ⇒ très peu de paramètres à apprendre.

---

## 12. Prudence face à la nouveauté

- **Les améliorations ne sont pas garanties** : se méfier des nouveautés.
- **Nouvelles façons de voir l'architecture Transformer** (cadre mathématique d'Anthropic : *transformer circuits*).
- **Séries temporelles** : *« Are Transformers effective for time series forecasting ? »* (Zeng et al.) — pas toujours supérieurs à des modèles simples.

---

## 13. Quiz d'auto-test

> Barème indicatif sur 30 : Q1–Q4 (4 pts) · Q5–Q7 (3 pts) · Q8–Q11 (2,5 pts) · Q12 (2 pts).

<details>
<summary><b>Question 1 (4 pts) :</b> Écrire la self-attention (scores, normalisation, nouvelle représentation) et expliquer le rôle de la division. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\alpha_{ij}=\mathrm{softmax}_j\!\Big(\frac{q_i\cdot k_j}{\sqrt{d_k}}\Big),\qquad v'_i=\sum_j\alpha_{ij}v_j\qquad\Big(\text{matriciellement }\mathrm{softmax}\big(QK^\top/\sqrt{d_k}\big)V\Big)$$

$q_i,k_j,v_j$ : requêtes, clés, valeurs (projections linéaires de l'embedding) ; $d_k$ : dimension des clés. La division par $\sqrt{d_k}$ évite que le **softmax soit trop sensible aux grandes valeurs** (température).
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Vérifier l'exemple : $q_1\cdot k_1=112$, $q_1\cdot k_2=96$, $d_k=64$. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$\sqrt{64}=8$ : $112/8=14$ et $96/8=12$ ; $\mathrm{softmax}(14,12)=\big(\tfrac{e^{14}}{e^{14}+e^{12}},\tfrac{e^{12}}{e^{14}+e^{12}}\big)\approx(0{,}88;0{,}12)$.
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Pourquoi faut-il un codage de position ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La self-attention est **invariante à l'ordre** : elle ne tient pas compte de la position des tokens. On **ajoute** à chaque embedding un embedding de position (sinusoïdal ou appris).
</details>

<details>
<summary><b>Question 4 (4 pts) :</b> Comment masque-t-on le padding (et le futur) dans l'attention ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
On met le **score à $-\infty$** avant le softmax pour les positions de padding (poids d'attention nul). Un **masque causal** (triangulaire) interdit à un token de voir les suivants (génération).
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Pourquoi plusieurs têtes ? Dimensions de BERT ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Chaque tête peut capturer une **relation différente** (têtes spécialisées en syntaxe). Les valeurs des têtes sont **concaténées** puis projetées. BERT : embedding 768, $k/q/v$ de dimension 64, 12 têtes ($12\times64=768$).
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Encodeur seul, décodeur seul, encodeur-décodeur : exemples et usages ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Encodeur seul (BERT) : représentation/classification/annotation. Décodeur seul (GPT) : génération par prédiction du token suivant. Encodeur-décodeur (article original, T5) : seq2seq (traduction, résumé).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Qu'est-ce que le MLM ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
**Masked Language Modeling** : on masque des tokens dans le texte et le modèle les prédit à partir du contexte (embeddings contextualisés + classification sur le vocabulaire). Alternative causale : prédiction du token suivant (GPT).
</details>

<details>
<summary><b>Question 8 (2,5 pts) :</b> Rôle de `[CLS]` et `[SEP]` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
`[CLS]` agrège l'information de toute la séquence (classification) ; `[SEP]` sépare deux segments (question/contexte, phrase 1/phrase 2).
</details>

<details>
<summary><b>Question 9 (2,5 pts) :</b> Quels avantages et inconvénients de l'échelle caractère / mot / sous-mot ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Caractère : petit vocabulaire mais pas de sémantique ; mot : sémantique mais vocabulaire énorme et mots inconnus ; sous-mot : compromis.
</details>

<details>
<summary><b>Question 10 (2,5 pts) :</b> Citer quatre stratégies de fine-tuning. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Fine-tuning complet, adapters, soft-prompting, LoRA (approximation de rang faible des matrices K, Q, V, W).
</details>

<details>
<summary><b>Question 11 (2,5 pts) :</b> Limites des RNN qui motivent le Transformer ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Dépendances longues, représentation globale du message, dépendances ciblées (coréférence). Le Transformer relie directement tous les tokens et traite tout le batch en parallèle.
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Pourquoi la connexion résiduelle est-elle utile dans le Transformer ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Elle laisse un chemin direct à l'information et au gradient (la diffusion se fait **pas à pas**) : on peut empiler de nombreuses couches.
</details>

---

## 14. 🚩 Pièges à éviter

- **Oublier la position** : sans codage de position, le modèle est un sac de mots.
- **Oublier le masque de padding** (ou causal pour la génération).
- **Mauvaise dimension de softmax** : sur les **clés**.
- **Pas de $\sqrt{d_k}$** : softmax saturé.
- **Confondre** BERT (encodeur, compréhension) et GPT (décodeur, génération).
- **Coût en $O(L^2)$** : limite de longueur du contexte.
- **Nouveautés** : les gains ne sont pas garantis (ex. séries temporelles).
