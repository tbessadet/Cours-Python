# 📘 Fiche de révision — Réseaux de neurones récurrents (RNN)

> Cours de Vincent Guigue (inspiré des supports de N. Baskiotis & B. Piwowarski, illustrations de C. Olah) · AgroParisTech – IODAA · 31 diapos (`RNN.pdf`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 notes perso

---

## Sommaire

1. [Le réseau récurrent](#1-le-réseau-récurrent)
2. [Spécificités et difficultés : gradient, LSTM](#2-spécificités-et-difficultés--gradient-lstm)
3. [Génération de données](#3-génération-de-données)
4. [Seq2Seq et architectures end-to-end](#4-seq2seq-et-architectures-end-to-end)
5. [Séries temporelles & machine learning](#5-séries-temporelles--machine-learning)
6. [Agrégation & attention](#6-agrégation--attention)
7. [Pièges à éviter](#7-pièges-à-éviter)
8. [Quiz auto-test](#8-quiz-auto-test)

---

## 1. Le réseau récurrent

### 1.1 *Recurrent Neural Network* (diapo 2)

Une même cellule $A$ est appliquée à chaque pas de temps ; l'état caché $h_{t-1}$ est réinjecté. Déplié dans le temps : $x_0 \to h_0$, $x_1 \to h_1$, …, avec **les mêmes poids** $W_1$ (entrée → état) et $W_2$ (état → état) à chaque pas.

Hypothèses : $x_t \in \mathbb R^n$ entrée au temps $t$, $h_t \in \mathbb R^d$ état caché.

$$h_t = \tanh(x_t W_1 + h_{t-1}W_2 + b) \quad\text{ou}\quad \tanh\big((x_t \oplus h_{t-1})W + b\big)$$

- $W_1$ : poids entrée → état ; $W_2$ : poids état → état ; $b$ : biais ;
- $\oplus$ : **concaténation** (les deux écritures sont équivalentes : $W$ empile $W_1$ et $W_2$).

> 💡 - **Peu de paramètres** (les mêmes à chaque pas de temps) ;
> - capacité à traiter des **longueurs variables** ;
> - capture la **dynamique** de la séquence.

### 1.2 Détails de la cellule (diapo 3)

RNN classique :
- **état latent** $h_t \in \mathbb R^d$ ; **entrée** $x_t \in \mathbb R^n$ ;
- $h_t = f_W(h_{t-1}, x_t) = \tanh\big((x_t \oplus h_{t-1})W + b\big)$, avec $W \in \mathbb R^{n \times d}$, $b \in \mathbb R$ ;
- **état initial** : $h_1$ ;
- séquence $[x_1, \dots, x_T] \Rightarrow [h_1, \dots, h_T]$ ;
- peut être calculé **de gauche à droite** ($h_t$)… **ou de droite à gauche** ($h^R_t$).

> 🔎 Dimensions exactes : $(x_t \oplus h_{t-1}) \in \mathbb R^{n+d}$, donc $W \in \mathbb R^{(n+d)\times d}$ et $b \in \mathbb R^d$ (la diapo abrège en $\mathbb R^{n\times d}$, $\mathbb R$).

### 1.3 RNN… pour quoi faire ? (diapo 4)

| Usage | État utilisé |
|---|---|
| **Prédire le pas suivant** | à partir de $h_T$ |
| **Génération** = prédiction (en boucle) | $\hat x_{t+1} = g(h_t)$ avec $W_3$ |
| **Classifier une séquence** | à partir de $h_T$ |
| **Détecter un événement** | à **chaque** $h_t$ |

**Bi-RNN** : utiliser $h_t$ et $h^R_t$ ; $(h_T \,\&\, h^R_1)$ = caractérisation de **toute la séquence** (chacun a lu la séquence entière, dans un sens).

### 1.4 Apprendre à générer (diapo 5)

| Mode | Prédiction | Entrée du pas suivant |
|---|---|---|
| **Teacher forcing** | $\hat x_{t+1} = g(h_t)$ | $h_{t+1} = f_W(h_t, \color{red}{x_{t+1}})$ : la **vraie** valeur |
| **Free generation** | $\hat x_{t+1} = g(h_t)$ | $h_{t+1} = f_W(h_t, \hat x_{t+1})$ : la valeur **prédite** |

**En pratique : *curriculum learning*** :
- commencer en teacher forcing (**plus stable**) ⇒ puis passer en génération libre ;
- l'apprentissage séquentiel est aussi une solution plus compliquée ⇒ **reinforcement learning**.

> 🔎 Le teacher forcing seul crée un **décalage** (*exposure bias*) : à l'entraînement le modèle ne voit jamais ses propres erreurs, qui s'accumulent en génération libre.

### 1.5 RNN en pratique (diapo 6)

```python
# Define the model
model = torch.nn.RNN(input_size=100, hidden_size=50,
                     num_layers=1, nonlinearity='tanh', bias=True)
# data (x_1, ..., x_T) : Seq. length T=23 & batch of size 10 ; input x_t ∈ R^n, dimension 100
seq_x = torch.rand(23, 10, 100)
# seq_h is a tensor (23, 10, 50) ; for classification purpose, we use last_h
seq_h, last_h = model(seq_x)
```

> 💡 **Convention** : tenseur de taille **time × sample/batch × (in) representation**.

---

## 2. Spécificités et difficultés : gradient, LSTM

### 2.1 Problèmes des RNN (diapo 7)

**Cas simple : linéaire + pas d'entrées.**
- Fonction d'erreur : $E$ ;
- $h_{t+1} = W h_t$ ;
- $t$ pas ⇒ calcul de $W^t$ (**puissance** de $t$) ;
- décomposition : $W = U\Sigma U^\top$, $\Sigma$ = matrice diagonale des valeurs propres.

$$\frac{\partial E}{\partial h_t} = \frac{\partial E}{\partial h_N}\,(W^\top)^{N-t} = \Delta U \begin{pmatrix}\sigma_1^{N-t} & & \\ & \ddots & \\ & & \sigma_n^{N-t}\end{pmatrix}$$

- $N$ : dernier pas de temps ; $\sigma_i$ : valeurs propres de $W$ ; $\Delta$ : terme regroupant $\frac{\partial E}{\partial h_N}$ (et $U^\top$).

> 💡 - si $\sigma_i > 1$ : **croissance exponentielle**, ***exploding gradient*** ;
> - si $\sigma_i < 1$ : ***gradient vanishing*** (l'influence des pas lointains disparaît).

### 2.2 *Exploding gradient* (diapo 8)

- Valeurs du gradient > 1 à chaque itération ; performances faibles ; ***NaN* partout très vite** !
- ⇒ **Gradient clipping** :

$$\nabla_W E = \begin{cases} \dfrac{\nabla_W E}{\lVert\nabla_W E\rVert} & \text{si } \lVert\nabla_W E\rVert > 1 \\[6pt] \nabla_W E & \text{sinon}\end{cases}$$

```python
# use before 'optimizer.step()'
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., norm_type=2)
```

> 🔎 On garde la **direction** du gradient mais on borne sa **norme** à `max_norm` (ici 1).

### 2.3 *Vanishing gradient* ⇒ architecture résiduelle (ex. ResNet) (diapo 9)

$$f(x) = x + f_0(x), \qquad \frac{\partial f(x)}{\partial x_j} = 1 + \frac{\partial f_0(x)}{\partial x_j}$$

Dans la rétro-propagation :

$$\frac{\partial E_f(x)}{\partial x_j} = \frac{\partial E_y}{\partial y_j} + \sum_i \frac{\partial E_y}{\partial y_i}\frac{\partial y_i}{\partial x_j}$$

> 💡 Le « $1 +$ » garantit qu'une partie du gradient passe **telle quelle** (le raccourci saute les couches $l-1$ → $l$), quelle que soit la profondeur.

### 2.4 *Vanishing gradient* ⇒ unité à portes (LSTM) (diapo 10)

**Le phénomène a été compris & (partiellement) surmonté : les neurones *apprennent* ce qui doit être *gardé en mémoire* et ce qui doit être *oublié*** (Hochreiter & Schmidhuber 1997, *Long short-term memory* ; blog de C. Olah).

- **Mémoire $c_t$** (modifiée par des opérateurs simples) ;
- **représentation externe $h_t$** extraite de $c_t$ ;
- **portes** : $y = \sigma(p) \otimes x$, avec $\oplus$ : concaténation, $\otimes$ : produit terme à terme, $\sigma$ : sigmoïde.

**Porte 1 (oubli)** :

$$g_t^{(f)} = \sigma\big[W_f(x_t \oplus h_{t-1}) + b_f\big], \qquad c_t^{(f)} = g_t^{(f)} \otimes c_{t-1}$$

**Porte 2 (mise à jour de la mémoire)** :

$$g_t^{(i)} = \sigma\big[W_{i,g}(x_t \oplus h_{t-1}) + b_{i,g}\big], \quad v_t^{(i)} = \tanh\big[W_{i,v}(x_t \oplus h_{t-1}) + b_{i,v}\big], \quad c_t = c_t^{(f)} + g_t^{(i)} \otimes v_t^{(i)}$$

**Porte 3 (sortie)** :

$$g_t^{(o)} = \sigma\big[W_{o,g}(x_t \oplus h_{t-1}) + b_{o,g}\big], \qquad v_t^{(o)} = \sigma\big[W_{o,v}c_t + b_{o,v}\big], \qquad h_t = g_t^{(o)} \otimes v_t^{(o)}$$

- $g^{(f)}, g^{(i)}, g^{(o)} \in\, ]0,1[^d$ : portes d'oubli, d'entrée, de sortie (0 = bloque, 1 = laisse passer) ;
- $v^{(i)}$ : valeur candidate à écrire en mémoire ; $c_t$ : mémoire (*cell state*) ; $h_t$ : sortie.

> ⚠️ 🔎 Dans le LSTM standard (et `nn.LSTM`), la sortie est $h_t = g_t^{(o)} \otimes \tanh(c_t)$ : il n'y a **pas** de couche $W_{o,v}$ ni de sigmoïde sur $c_t$. La diapo présente une variante.

> 💡 La mémoire $c_t$ se met à jour par une **addition** ($c_t = g^{(f)} \otimes c_{t-1} + \dots$) et non par une multiplication matricielle : si la porte d'oubli vaut ≈ 1, le gradient traverse le temps presque intact (même idée que le résiduel).

### 2.5 Bi-RNN, RNN multicouche… (diapo 11)

- **RNN multicouche** : agrégation temporelle plus complexe (ex. LSTM-1 512 → LSTM-2 256 → LSTM-3 128 → LSTM-4 64, avec **Dropout = 0,2** entre couches, puis une couche d'activation pour la sortie $Y_t$).
- **BiRNN** : lire la séquence **dans les deux sens** (une cellule $A$ de gauche à droite, une cellule $A'$ de droite à gauche ; la sortie $y_t$ combine les deux).

### 2.6 Conclusion de la partie (diapo 12)

- **LSTM** et sa variante (**GRU**) sont très utilisés ;
- les **biRNN** (biLSTM / BiGRU) offrent souvent une **convergence plus rapide** ;
- **le gradient est toujours un problème : surveillez-le !**

```python
# Calcul de ||∇_W E||²
with torch.no_grad():
    grad_sq12 = sum((p.grad ** 2).sum() for p in l.parameters())
```

- **Dropout** utile contre le sur-apprentissage ; les couches peuvent être **empilées** (!) ⇒ `num_layers`.

```python
# Module definition
model = torch.nn.LSTM(input_size=100, hidden_size=50,
                      batch_first=False, num_layers=1, dropout=.3)
```

> 🔎 `dropout` de `nn.LSTM` ne s'applique **qu'entre les couches empilées** : avec `num_layers=1` il n'a aucun effet (PyTorch émet un avertissement).

---

## 3. Génération de données

### 3.1 Génération basique (diapo 13)

Prédiction de mot : « the cat sat » → *embedding* → couches cachées ($W_1$, $W_2$) → $W_3$ → $P(w \mid h)$ → *argmax* → « on », réinjecté en entrée.

- $W_3 \approx$ **classification sur le vocabulaire** ;
- plusieurs variantes : **argmax / sampling**… ;
- ⇒ **réduire le vocabulaire** : à l'échelle de la **lettre** (dim ≈ 60) ; ***byte pair encoding*** ⇒ n-grammes de lettres les plus fréquents ;
- l'itération à $t+1$ ne nécessite que : $x_{t+1}$ et $h_t$ (qui **encode toute la séquence** des mots précédents).

### 3.2 Teacher forcing & forme des données (diapo 14)

Un batch = $B$ phrases $(x_1, \dots, x_T)$ empilées. Pour chaque phrase : **$X = (x_1, \dots, x_{T-1})$** et **$Y = (x_2, \dots, x_T)$** (la cible est l'entrée **décalée d'un pas**).

### 3.3 *Beam search* (diapo 15)

**Très difficile de trouver le meilleur mot au temps $t$… ⇒ explorer plusieurs possibilités, garder les plus probables !** Exemple : 4 faisceaux (*beams*) développant « A bedroom with a… », « A chair in a… », « A desk next to… », « A table and chairs… », chacun étendu par les mots les plus probables, en ne gardant que les meilleures séquences complètes.

- **Fonction d'échantillonnage requise** (cf. TP).

> 🔎 À chaque pas : pour chacune des $k$ séquences candidates, on considère les $k$ mots les plus probables, on calcule le **score** = somme des log-probabilités, et on garde les $k$ meilleures séquences parmi les $k^2$. Une **normalisation par la longueur** évite de favoriser les phrases courtes (somme de log-probas négatives).

### 3.4 Évaluation : BLEU / ROUGE (traduction & résumé) (diapo 16)

**Comment évaluer la génération de texte ? Très difficile… encore une question ouverte !**

Exemple : sortie du modèle « *the hello a cat dog fox jumps* », référence « *the fox jumps* » → n-grammes de la référence : `['the', 'fox', 'jumps']`.

$$\text{rappel} = \frac{\mathrm{count}_{match}(\text{gram}_n)}{\mathrm{count}(\text{gram}_n)} = \frac{3}{3} = 100\,\%, \qquad \text{précision} = \frac{3}{7} \approx 0{,}43$$

$$F_1 = 2\cdot\frac{0{,}43 \times 1{,}0}{0{,}43 + 1{,}0} = 0{,}6 \;\Rightarrow\; 60\,\%$$

- $\mathrm{count}_{match}$ : n-grammes de la référence retrouvés dans la sortie ; $\mathrm{count}$ : n-grammes de la référence (rappel) ou de la sortie (précision).
- Calculs de **f1** ; sur des **N-grammes**, sur la **plus longue sous-séquence commune**…

> 🔎 **BLEU** (traduction) est orienté **précision** (avec pénalité de brièveté) ; **ROUGE** (résumé) est orienté **rappel** (ROUGE-N : n-grammes, ROUGE-L : plus longue sous-séquence commune).

### 3.5 Conclusion (diapo 17)

- **Évaluation dans l'espace latent** (cf. BERT-score / CLS) ;
- **évaluation du contenu** (entités, chiffres…) ⇒ problèmes d'**hallucination**.

> ⚠️ L'IA générative est une opportunité dans de nombreux domaines… mais **les benchmarks ne sont pas (encore) adaptés à cet outil**.

---

## 4. Seq2Seq et architectures end-to-end

### 4.1 Architecture RNN : différents cadres (diapos 18 et 28)

| Cadre | Exemple |
|---|---|
| one to one | (réseau classique) |
| **one to many** | annotation d'image (*image captioning*) |
| **many to one** | classification de signal |
| **many to many** (synchronisé) | étiquetage POS/NER, annotation de séquence |
| **seq to seq** (many to many décalé) | traduction automatique |

(Blog de Karpathy, *The unreasonable effectiveness of RNNs*.)

### 4.2 Architecture encodeur-décodeur (diapo 19)

L'**encodeur** lit « A B C » ; à `<eos>`, le **décodeur** produit « W X Y Z <eos> » en réinjectant chaque mot produit. Objectif :

$$\max \frac{1}{|S|}\sum_{(T,S)\in\mathcal S}\log p(T \mid S)$$

- $S$ : phrase source ; $T$ : phrase cible ; $\mathcal S$ : corpus de paires.

- Exploiter le paradigme **seq-2-seq** en séries temporelles ;
- **encoder** la série temporelle (& être capable de la décoder). (Sutskever et al. 2014.)

### 4.3 Comment entraîner un traducteur ? (diapo 20)

- **Prédire le(s) mot(s) suivant(s) dans les deux langues** : quel vocabulaire ? un ou deux réseaux ?
- **De la prédiction de mots à la traduction** : comment apprendre le traducteur ?

> 💡 ***Curriculum learning* & architectures end-to-end sont complémentaires.**

---

## 5. Séries temporelles & machine learning

### 5.1 ML vs DL pour les séries temporelles (diapo 21)

Questions : quelles applications ? quelle description du signal ? quel contexte ? quelle cible ?

**Approches classiques :**
1. **Statistiques historiques** : AR, ARMA, SARIMA ;
2. **Machine learning** : SVM, Ridge, Random Forest, XGB — quels descripteurs ? (cf. pandas) : **stats** (min, max, mean, std…), **fréquentiels** (Fourier, PSD…), **contexte** (jour, météo, lieu) ;
3. **Dynamic Time Warping (DTW)** : calcul efficace de **distance entre signaux** → kNN, approches non supervisées.

**Quand utiliser le deep learning ?**

> 🔎 DTW aligne deux séries en autorisant des **décalages / étirements temporels** (programmation dynamique), contrairement à la distance euclidienne point à point.

### 5.2 Architectures neuronales & séries (multivariées) (diapo 22)

- **CNN** : **filtres différents pour chaque canal / mêmes filtres** (convolution 1D sur le temps).
- **RNN** : **aucun problème pour donner un vecteur en entrée à chaque pas de temps** (série multivariée : $x^t \in \mathbb R^n$, $s^t = f(Ux^t + Ws^{t-1})$, $\hat y^t = Vs^t$).

### 5.3 Cadre d'apprentissage non supervisé (diapo 23) — Franceschi et al., NeurIPS 2019

**Étape 1 : encodage** : convolutions **causales dilatées** empilées (dilatation $2^0 = 1$, $2^1 = 2$, $2^2 = 4$…, blocs résiduels : convolution causale → *weight norm* → Leaky ReLU, ×2, + raccourci) **+ max pooling** (vecteur de taille fixe).

**Étape 2 : apprentissage non supervisé** — étant donné une sous-séquence $x^{pos}$ et des contextes locaux $x^{ref}$ :

$$\text{Idée Skip-Gram : } \arg\max_\theta \prod_{x^{ref}}\prod_{x^{pos}\in x^{ref}} p(x^{ref} \mid x^{pos}; \theta)$$

- $p(D = 1 \mid x^{ref}, x^{pos}; \theta)$ ⇒ probabilité que $x^{pos}$ apparaisse dans le contexte $x^{ref}$ ;
- **triplet loss** :

$$\arg\max_\theta \prod_{ref, pos} p(D = 1 \mid x^{ref}, x^{pos}; \theta) \;+\; \underbrace{\prod_{ref, neg} p(D = 0 \mid x^{ref}, x^{neg}; \theta)}_{\text{Negative Sampling}}$$

- $x^{ref}$ : sous-série de référence ; $x^{pos}$ : sous-série **incluse** dans $x^{ref}$ ; $x^{neg}$ : sous-série prise dans une **autre** série.

**Espace latent intéressant** pour les séries (classes bien séparées sur DiatomSizeReduction, plus mélangées sur FordB, OSULeaf) → **très bons résultats de classification avec une simple régression logistique**.

> 💡 Transposition directe de Word2Vec (module 3) aux séries temporelles.

### 5.4 *Vector Quantization* (VQ) (diapo 24)

- Le machine learning est **surtout adapté aux valeurs continues** ;
- **quantification = continu ⇒ discret** ;
- hypothèse : les **valeurs catégorielles sont plus interprétables** → **identification de motifs** (*codebooks*, cellules de Voronoï).

### 5.5 VQ-Variational Auto-Encoder (diapo 25) — van den Oord et al., NeurIPS 2017

Image → **CNN encodeur** → $z_e(x)$ → chaque vecteur est remplacé par le **plus proche** élément $e_k$ du dictionnaire (*embedding space* $e_1, \dots, e_K$) → $z_q(x)$ → **CNN décodeur** → $p(x \mid z_q)$. Le gradient $\nabla_z L$ du décodeur est **recopié** vers l'encodeur (la quantification n'est pas dérivable).

- **Interprétabilité = représentation discrète ⇒ décomposition en objets** ;
- … & **dictionnaire (codebook) limité**.

### 5.6 Implémentation VQ-VAE pour séries temporelles (diapo 26) — MATS, Yu et al., ICLR 2023

- **Décomposition discrète, reconstruction du signal** ;
- **+ discriminateur adversarial** (⇒ reconstruction du bruit ?).

### 5.7 Conclusion (diapo 27)

**Nombreuses applications en séries temporelles :**
- modéliser / discriminer des **formes complexes** ;
- **capacité de prédiction** ;
- **modélisation du contexte** (ex. transports publics).

> 💡 **Comment choisir parmi les architectures deep ? RNN, CNN, …**

---

## 6. Agrégation & attention

### 6.1 RNN, CNN & pooling (diapo 29)

Variante d'architecture :
- **many to one** : on ne garde que le **dernier** état $h_T$ ;
- **many to many + agrégation** : on **agrège tous** les états cachés $h_1, \dots, h_T$ (pooling) en un vecteur.
- *Et les CNN ?*

> 🔎 Le dernier état d'un RNN « oublie » le début des longues séquences ; agréger (moyenne, max) tous les $h_t$ ou les pondérer (attention) donne une représentation plus robuste.

### 6.2 Attention & agrégation (diapo 30)

Les états $h_1, \dots, h_m$ sont pondérés par des poids $\alpha_1, \dots, \alpha_m$ (calculés à partir d'un **vecteur d'attention** $a_s$) puis **sommés** : $e_s = \sum_i \alpha_i h_i$.

- **Apprendre à pondérer une représentation localisée** ;
- de **nombreuses applications efficaces** [2014-2018] ;
- **peu de paramètres** ;
- **vecteur d'attention = vecteur *query*** ;
- comment le coder en PyTorch ?

🔎 Formulation (attention globale), hypothèses : $h_i \in \mathbb R^d$ états du RNN, $a_s \in \mathbb R^d$ vecteur d'attention **appris** :

$$\alpha_i = \frac{\exp(\langle a_s, h_i\rangle)}{\sum_{j=1}^{m}\exp(\langle a_s, h_j\rangle)}, \qquad e_s = \sum_{i=1}^{m}\alpha_i\, h_i$$

- $\alpha_i \in [0,1]$, $\sum_i\alpha_i = 1$ : importance du pas $i$ ; $e_s$ : représentation agrégée de la séquence.

**Séance pratique** : Q1 tests préliminaires : **attention uniforme** ($\alpha_i = 1/m$ = moyenne) ; Q2 **attention globale classique**.

### 6.3 Attention avancée (diapo 31)

- **L'attention dépend de l'entrée** (🔎 la *query* est calculée à partir de la séquence ou d'un autre élément, ex. l'état du décodeur en traduction) ;
- **attention sur les sorties du RNN** ;
- **batch, padding & attention masquée** (🔎 les positions de *padding* reçoivent un score $-\infty$ avant le softmax, donc un poids nul).

> 💡 Cette idée de pondération apprise mène directement aux **Transformers** (module 5).

---

## 7. Pièges à éviter

> 🚩 **Confondre les dimensions** : convention PyTorch par défaut **(temps, batch, features)** ; `batch_first=True` pour (batch, temps, features).

> 🚩 **Exploding gradient** → NaN très vite : **gradient clipping** (`clip_grad_norm_` **avant** `optimizer.step()`).

> 🚩 **Vanishing gradient** : un RNN simple oublie les pas lointains → LSTM/GRU, résiduel, attention.

> 🚩 **Teacher forcing seul** : bon à l'entraînement, mais erreurs qui s'accumulent en génération libre → *curriculum learning*.

> ⚠️ **Avec du padding**, le « dernier » état $h_T$ d'une séquence courte est calculé sur des `<pad>` : récupérer l'état au vrai dernier pas (longueurs) et **masquer** l'attention et la loss.

> ⚠️ **Argmax glouton** en génération : répétitions, phrases pauvres → **échantillonnage** ou **beam search** (avec normalisation de longueur).

> ⚠️ **Évaluer de la génération** : BLEU/ROUGE ne mesurent que des recouvrements de n-grammes ; hallucinations non détectées.

> ⚠️ **LSTM** : la sortie standard est $h_t = o_t \otimes \tanh(c_t)$ (variante sur la diapo) ; `nn.LSTM` renvoie `(output, (h_n, c_n))`.

> ⚠️ **Séries temporelles** : le deep n'est pas toujours nécessaire (AR/ARMA, features + Random Forest/XGB, DTW + kNN sont de bonnes bases).

---

## 8. Quiz auto-test

<details>
<summary><b>Question 1 (2 pts) :</b> Écrire l'équation d'un RNN simple et donner ses trois avantages. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$h_t = \tanh(x_tW_1 + h_{t-1}W_2 + b) = \tanh\big((x_t\oplus h_{t-1})W + b\big)$$

Peu de paramètres (partagés dans le temps), longueurs variables, capture de la dynamique de la séquence.
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Expliquer l'origine de l'exploding et du vanishing gradient dans un RNN linéaire. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Avec $h_{t+1} = Wh_t$ et $W = U\Sigma U^\top$, le gradient remonte avec la puissance $(W^\top)^{N-t}$, donc les valeurs propres à la puissance $N - t$ : $\sigma_i^{N-t}$. Si $\sigma_i > 1$ : croissance exponentielle (exploding, NaN) ; si $\sigma_i < 1$ : le gradient disparaît (vanishing) et le réseau n'apprend plus les dépendances lointaines.
</details>

<details>
<summary><b>Question 3 (2 pts) :</b> Qu'est-ce que le gradient clipping ? Où le placer dans la boucle ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Si $\lVert\nabla_W E\rVert > 1$ (ou `max_norm`), on renormalise le gradient : $\nabla_W E / \lVert\nabla_W E\rVert$ ; sinon on le garde. Direction conservée, norme bornée. `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)` entre `loss.backward()` et `optimizer.step()`.
</details>

<details>
<summary><b>Question 4 (4 pts) :</b> Écrire les équations des portes d'oubli et d'entrée du LSTM et la mise à jour de la mémoire. Pourquoi le LSTM limite-t-il le vanishing gradient ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$g^{(f)}_t = \sigma[W_f(x_t\oplus h_{t-1}) + b_f], \quad g^{(i)}_t = \sigma[W_{i,g}(x_t\oplus h_{t-1}) + b_{i,g}], \quad v^{(i)}_t = \tanh[W_{i,v}(x_t\oplus h_{t-1}) + b_{i,v}]$$

$$c_t = g^{(f)}_t\otimes c_{t-1} + g^{(i)}_t\otimes v^{(i)}_t$$

La mémoire est mise à jour par **addition**, pondérée par une porte apprise : quand la porte d'oubli vaut ≈ 1, le gradient traverse le temps sans être multiplié par une matrice (même idée que les connexions résiduelles). Les neurones apprennent quoi garder et quoi oublier.
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> Différence entre teacher forcing et free generation ? Qu'est-ce que le curriculum learning ici ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Teacher forcing : $h_{t+1} = f_W(h_t, x_{t+1})$ avec la **vraie** valeur ; free generation : $h_{t+1} = f_W(h_t, \hat x_{t+1})$ avec la valeur **prédite**. Curriculum : commencer en teacher forcing (plus stable) puis basculer progressivement vers la génération libre.
</details>

<details>
<summary><b>Question 6 (2 pts) :</b> Comment construire X et Y pour entraîner un modèle de langue sur une phrase (x₁, …, x_T) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$X = (x_1,\dots,x_{T-1})$ et $Y = (x_2,\dots,x_T)$ : la cible est l'entrée décalée d'un pas ; on prédit le token suivant à chaque position (classification sur le vocabulaire, en masquant le padding).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Principe du beam search. Pourquoi normaliser par la longueur ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Trouver le meilleur mot à chaque pas (argmax) ne donne pas la meilleure phrase : on garde $k$ séquences candidates, on étend chacune par ses $k$ mots les plus probables, on garde les $k$ meilleures (somme des log-probabilités). Comme chaque log-proba est négative, les phrases longues ont un score plus faible : on normalise par une fonction de la longueur pour ne pas favoriser les phrases courtes.
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Sortie « the hello a cat dog fox jumps », référence « the fox jumps ». Calculer rappel, précision et F1 sur les unigrammes. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

3 unigrammes de la référence sont retrouvés : rappel $= 3/3 = 1{,}0$ ; précision $= 3/7 \approx 0{,}43$ ;

$$F_1 = 2\cdot\frac{0{,}43\times1{,}0}{0{,}43+1{,}0}\approx 0{,}6$$

ROUGE (résumé) est orienté rappel, BLEU (traduction) précision.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Associer chaque cadre RNN à une application : one-to-many, many-to-one, many-to-many, seq-to-seq. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

One-to-many : annotation d'image ; many-to-one : classification de signal ; many-to-many : étiquetage POS/NER ; seq-to-seq : traduction automatique (encodeur-décodeur).
</details>

<details>
<summary><b>Question 10 (3 pts) :</b> Écrire le mécanisme d'attention globale sur les états d'un RNN. Comment gérer le padding ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\alpha_i = \frac{\exp(\langle a_s, h_i\rangle)}{\sum_j\exp(\langle a_s, h_j\rangle)}, \qquad e_s = \sum_i\alpha_i h_i$$

$a_s$ : vecteur d'attention (query) appris. Peu de paramètres, pondération apprise des positions. Padding : masquer les positions de pad en mettant leur score à $-\infty$ avant le softmax (poids nul). L'attention uniforme ($\alpha_i = 1/m$) est la moyenne.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Comment apprendre des représentations de séries temporelles sans supervision (Franceschi et al.) ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Encodeur à convolutions causales dilatées + max pooling, entraîné par **triplet loss** à la Word2Vec : une sous-série $x^{pos}$ incluse dans $x^{ref}$ doit être proche ($D=1$), une sous-série $x^{neg}$ d'une autre série éloignée ($D=0$, negative sampling). L'espace obtenu permet une bonne classification avec une simple régression logistique.
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Quelles approches classiques (non deep) pour les séries temporelles ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Statistiques (AR, ARMA, SARIMA) ; ML sur descripteurs (stats min/max/mean/std, fréquentiels Fourier/PSD, contexte jour/météo/lieu) avec SVM, Ridge, Random Forest, XGB ; DTW (distance entre signaux avec alignement temporel) + kNN ou non supervisé.
</details>

<details>
<summary><b>Question 13 (1 pt) :</b> Quel est l'intérêt d'un VQ-VAE ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Représentation **discrète** (chaque vecteur latent remplacé par le plus proche élément d'un dictionnaire limité) : plus interprétable, décomposition en motifs/objets, reconstruction du signal à partir d'un ensemble fini de motifs.
</details>
