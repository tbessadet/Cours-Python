# 🔁 Fiche de cours — Réseaux de neurones récurrents (RNN, LSTM, génération, seq2seq, séries temporelles, attention)

> Source : `4-RNN.pdf` (42 pages, 30 diapos numérotées). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours) · 📝 mes notes (aucune fournie pour ce cours). Les dates/références historiques ont été omises.

## Sommaire
1. [Le RNN : récurrence et usages](#1-le-rnn--récurrence-et-usages)
2. [Apprendre à générer : teacher forcing vs génération libre](#2-apprendre-à-générer--teacher-forcing-vs-génération-libre)
3. [Le RNN en pratique (PyTorch)](#3-le-rnn-en-pratique-pytorch)
4. [Problèmes : gradient qui explose / disparaît](#4-problèmes--gradient-qui-explose--disparaît)
5. [Architectures à portes : LSTM (et GRU)](#5-architectures-à-portes--lstm-et-gru)
6. [Conclusion sur les RNN](#6-conclusion-sur-les-rnn)
7. [Génération de données](#7-génération-de-données)
8. [Beam search et évaluation (BLEU/ROUGE)](#8-beam-search-et-évaluation-bleurouge)
9. [Seq2Seq et architectures end-to-end](#9-seq2seq-et-architectures-end-to-end)
10. [Séries temporelles](#10-séries-temporelles)
11. [Agrégation et attention](#11-agrégation-et-attention)
12. [Quiz d'auto-test](#12-quiz-dauto-test)
13. [🚩 Pièges à éviter](#13--pièges-à-éviter)

---

## 1. Le RNN : récurrence et usages

**RNN général** (hypothèse : $x_t\in\mathbb R^n$ entrée, $h_t\in\mathbb R^d$ état latent) :

$$h_t=\tanh\big(x_tW_1+h_{t-1}W_2+b\big)\quad\text{ou}\quad h_t=\tanh\big((x_t\oplus h_{t-1})W+b\big)=f_W(h_{t-1},x_t)$$

- $\oplus$ : concaténation ; $W_1,W_2$ (ou $W$) : matrices de poids **partagées à tous les pas** ; $b$ : biais ; état initial $h_1$ (souvent nul).
- 🔎 Dimensions : $W_1\in\mathbb R^{n\times d}$, $W_2\in\mathbb R^{d\times d}$ ; pour la version concaténée $W\in\mathbb R^{(n+d)\times d}$, $b\in\mathbb R^d$.

> 💡 Avantages : **peu de paramètres** ; capacité à traiter des **longueurs variables** ; capture la **dynamique** de la séquence.

Séquence $[x_1,\dots,x_T]\mapsto[h_1,\dots,h_T]$, calculable **de gauche à droite** ($h_t$) ou **de droite à gauche** ($h^R_t$).

### Pour quels usages ?
- **Prédire le pas suivant** (à partir de $h_T$) ; la **génération** = prédiction **en boucle**.
- **Classifier une séquence** (à partir de $h_T$).
- **Détecter un événement** (à chaque $h_t$).
- **Bi-RNN** : utiliser $h_t$ et $h^R_t$ ($h_T$ et $h^R_1$ caractérisent toute la séquence).

---

## 2. Apprendre à générer : teacher forcing vs génération libre

Hypothèse : $g$ = décodeur (ex. couche linéaire + softmax) qui transforme l'état en prédiction du prochain élément.

| Mode | Équations | Remarque |
|---|---|---|
| **Teacher forcing** | $\hat x_{t+1}=g(h_t),\quad h_{t+1}=f_W(h_t,\ x_{t+1})$ | on injecte le **vrai** élément $x_{t+1}$ ; plus stable |
| **Génération libre** | $\hat x_{t+1}=g(h_t),\quad h_{t+1}=f_W(h_t,\ \hat x_{t+1})$ | on réinjecte sa **propre prédiction** ; comme à l'inférence |

> 💡 En pratique : **curriculum learning** — commencer par le teacher forcing (plus stable) **puis** passer à la génération libre. L'apprentissage séquentiel donne aussi une solution plus compliquée (lien avec l'**apprentissage par renforcement**).

---

## 3. Le RNN en pratique (PyTorch)

```python
model = torch.nn.RNN(input_size=100, hidden_size=50, num_layers=1,
                     nonlinearity='tanh', bias=True)
# données (x_1..x_T) : T=23 pas de temps, batch de 10, x_t dans R^100
seq_x = torch.rand(23, 10, 100)
seq_h, last_h = model(seq_x)      # seq_h : (23, 10, 50) ; pour la classification on utilise last_h
```
> **Convention : tenseur de taille *temps × échantillon × représentation*.**

⚠️ Pour une classification, on prend l'**état final** ; pour la génération/prédiction pas à pas, **tous** les `seq_h`.

---

## 4. Problèmes : gradient qui explose / disparaît

**Cas d'école** : fonction d'erreur $E$, dynamique linéaire sans entrées $h_{t+1}=Wh_t$. Au pas $t$ : calcul de $W^t$ (puissance de $t$). Hypothèse : $W=U\Sigma U^\top$ diagonalisable, $\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_n)$ (valeurs propres).

$$\frac{\partial E}{\partial h_t}=\frac{\partial E}{\partial h_N}\,(W^\top)^{N-t}=\frac{\partial E}{\partial h_N}\,U\begin{pmatrix}\sigma_1^{N-t}&&\\&\ddots&\\&&\sigma_n^{N-t}\end{pmatrix}U^\top$$

- $N$ : longueur de la séquence ; $N-t$ : nombre de pas rétropropagés ; $\sigma_i$ : valeurs propres de $W$.
- si $\sigma_i>1$ : **croissance exponentielle** → **gradient qui explose** ;
- si $\sigma_i<1$ : **gradient qui disparaît** (*vanishing*).

### Explosion du gradient
Valeurs de gradient $>1$ à chaque itération → mauvaises performances, **NaN partout** rapidement.
**Remède : gradient clipping** (hypothèse : seuil 1) :
$$\nabla_WE\leftarrow\begin{cases}\dfrac{\nabla_WE}{\lVert\nabla_WE\rVert}&\text{si }\lVert\nabla_WE\rVert>1\\[2mm]\nabla_WE&\text{sinon}\end{cases}$$
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1., norm_type=2)   # AVANT optimizer.step()
```

### Disparition du gradient — architecture résiduelle (type ResNet)
$$f(x)=x+f_0(x)\ \Rightarrow\ \frac{\partial f(x)}{\partial x_j}=1+\frac{\partial f_0(x)}{\partial x_j}$$
Le « **1** » garantit un chemin où le gradient **ne s'annule pas**. En rétropropagation, avec $y=x+f_0(x)$ :
$$\frac{\partial E}{\partial x_j}=\underbrace{\frac{\partial E}{\partial y_j}}_{\text{terme direct}}+\underbrace{\sum_i\frac{\partial E}{\partial y_i}\frac{\partial f_{0,i}(x)}{\partial x_j}}_{\text{à travers la couche}}$$
(🔎 écriture explicitée à partir de la diapo.)

---

## 5. Architectures à portes : LSTM (et GRU)

> Le phénomène est **compris et (partiellement) surmonté** : les neurones **apprennent ce qui doit être gardé en mémoire et ce qui doit être oublié**.

**Principes** : mémoire $c_t$ (modifiée par des opérateurs simples) ; **représentation externe** $h_t$ extraite de $c_t$ ; **portes** $y=\sigma(p)\otimes x$, avec $\oplus$ = concaténation, $\otimes$ = produit terme à terme, $\sigma$ = sigmoïde (valeurs dans $(0,1)$ : 0 = bloque, 1 = laisse passer).

Hypothèse : $[x_t\oplus h_{t-1}]$ est l'entrée concaténée ; $W_\bullet,b_\bullet$ sont les paramètres de chaque porte.

**Porte 1 — oubli (*forget*)** :
$$g^{(f)}_t=\sigma\big[W_f(x_t\oplus h_{t-1})+b_f\big],\qquad c^{(f)}_t=g^{(f)}_t\otimes c_{t-1}$$

**Porte 2 — mise à jour de la mémoire** :
$$g^{(i)}_t=\sigma\big[W_{i,g}(x_t\oplus h_{t-1})+b_{i,g}\big],\quad v^{(i)}_t=\tanh\big[W_{i,v}(x_t\oplus h_{t-1})+b_{i,v}\big],\quad c_t=c^{(f)}_t+g^{(i)}_t\otimes v^{(i)}_t$$

**Porte 3 — sortie** :
$$g^{(o)}_t=\sigma\big[W_{o,g}(x_t\oplus h_{t-1})+b_{o,g}\big],\quad v^{(o)}_t=\sigma\big[W_{o,v}\,c_t+b_{o,v}\big],\quad h_t=g^{(o)}_t\otimes v^{(o)}_t$$

> 🔎 Formulation « standard » de la sortie : $h_t=g^{(o)}_t\otimes\tanh(c_t)$ (pas de matrice supplémentaire $W_{o,v}$). La diapo utilise $\sigma[W_{o,v}c_t+b_{o,v}]$ ; l'idée est identique : $h_t$ est une version **filtrée** de la mémoire $c_t$.

**Pourquoi ça aide** : la mémoire $c_t$ est modifiée par **addition** (comme une connexion résiduelle) → le gradient se propage sans s'écraser à travers $c_t$.

> 🔎 **GRU** : variante simplifiée (2 portes, pas de $c_t$ séparé) ; plus légère, souvent aussi performante.

---

## 6. Conclusion sur les RNN

- **LSTM et variantes (GRU)** : très utilisés.
- **BiRNN (biLSTM/BiGRU)** : convergence souvent plus rapide.
- **Le gradient est toujours un problème : le surveiller** :
```python
with torch.no_grad():
    grad_sq_l2 = sum((p.grad ** 2).sum() for p in l.parameters())   # ||∇_W E||²
```
- **Dropout** utile contre le sur-apprentissage ; couches **empilables** (`num_layers`).
```python
model = torch.nn.LSTM(input_size=100, hidden_size=50, batch_first=False, num_layers=1, dropout=.3)
```
⚠️ `dropout` de `nn.LSTM` s'applique **entre** les couches empilées (sans effet avec `num_layers=1`).

---

## 7. Génération de données

**Génération de base** (prédiction de mots) : *embedding* → **état caché** → **décodeur** ($W_3$) = **classification sur le vocabulaire**. Variantes de choix : **argmax / échantillonnage**.
> L'itération à $t+1$ ne requiert que : le token $x_{t+1}$ et **l'état $h_t$** (qui **encode toute la séquence de mots précédents**).

**Réduire le vocabulaire** :
- au niveau de la **lettre** (dimension ≈ 60) ;
- **byte pair encoding** (BPE) : n-grammes de lettres les plus fréquents.

**Teacher forcing et forme des données** (batch) : toutes les phrases d'un batch sont **complétées (padding)** à la longueur de la plus longue ; entrée = séquence décalée de la cible.

---

## 8. Beam search et évaluation (BLEU/ROUGE)

**Beam search** : il est très difficile de trouver le meilleur mot à l'instant $t$ → **explorer plusieurs possibilités et garder les plus probables** (*fonction d'échantillonnage requise*, cf. TP).

**Évaluation (traduction/résumé)** : **BLEU / ROUGE** — calculs de **F1 sur des n-grammes**, sur la **plus longue sous-séquence commune**… **Comment évaluer une génération de texte ? Question encore ouverte.**

**Conclusion** : évaluation **dans l'espace latent** (cf. BERT-score / CLS) ; **évaluation du contenu** (entités, chiffres…) → problèmes d'**hallucination**. L'IA générative est une opportunité dans divers domaines, mais **les benchmarks ne sont pas (encore) adaptés** à cet outil.

---

## 9. Seq2Seq et architectures end-to-end

**Différents schémas RNN** :
| Schéma | Exemple |
|---|---|
| **One to many** | annotation d'image |
| **Many to one** | classification de signal |
| **Many to many** | POS/NER tagging, annotation de séquence |
| **Seq to seq** | traduction automatique |

**Architecture encodeur-décodeur** : exploiter le paradigme seq2seq aussi pour les **séries temporelles** — encoder la série (et pouvoir la décoder).

**Comment entraîner un traducteur ?** Prédire le(s) mot(s) suivant(s) **dans les deux langues** ; questions : quel vocabulaire ? un ou deux réseaux ? **de la prédiction de mots à la traduction** : comment apprendre le traducteur ? *Curriculum learning* et architectures end-to-end sont **complémentaires**.

---

## 10. Séries temporelles

**Machine learning vs deep learning** — quelles applications, quelle description du signal, quel contexte, quelle cible ?

**Approches classiques** :
1. **Statistiques historiques** : AR, ARMA, SARIMA ;
2. **Machine learning** : SVM, Ridge, Random Forest, XGB — avec des **descripteurs** : *stats* (min, max, moyenne, écart-type…), *fréquentiels* (Fourier, PSD…), *contexte* (jour, météo, lieu) ;
3. **Dynamic Time Warping (DTW)** : calcul efficace d'une **distance entre signaux** → k-NN, approches non supervisées.

*Quand utiliser le deep learning ?* — **Architectures neuronales pour séries (multivariées)** : **CNN** (filtres différents pour chaque canal / mêmes filtres) ; **RNN** : donne le vecteur en entrée à chaque pas (many-to-many, ex. parole).

**Apprentissage non supervisé (représentation de séries)** :
- **Étape 1 – encodage** : pile de **convolutions causales dilatées** (dilatations $2^0,2^1,2^2,\dots$, aucune valeur future utilisée) + normalisation de poids + Leaky ReLU, puis **max pooling global** qui écrase la dimension temporelle → **vecteur de taille fixe** (sortie de l'encodeur par transformation linéaire).
- **Étape 2 – apprentissage non supervisé par triplet loss**, inspiré du **skip-gram/word2vec** : pour une sous-série $x^{ref}$, un **positif** $x^{pos}$ (sous-série contenue dans $x^{ref}$) et des **négatifs** $x^{neg}$ (sous-séries d'autres séries tirées au hasard) ; objectif :

$$\arg\max\ \ \log\sigma\big(f(x^{ref})^\top f(x^{pos})\big)+\sum_{k}\log\sigma\big(-f(x^{ref})^\top f(x^{neg}_k)\big)$$
*(🔎 la diapo (extrait de l'article) montre les probabilités $p(D=1\mid x^{ref},x^{pos};\theta)$ et $p(D=0\mid x^{ref},x^{neg};\theta)$ ; la forme ci-dessus est la formulation « negative sampling » correspondante, avec $f$ = encodeur et $\sigma$ = sigmoïde)* — l'**espace latent** obtenu regroupe les séries de même classe (visualisation t-SNE).

**Quantification vectorielle (VQ)** : le ML convient surtout aux valeurs **continues** ; **quantifier = continu → discret** ; hypothèse : les valeurs **catégorielles sont plus interprétables** (identification de motifs).
**VQ-VAE** : la sortie de l'encodeur $z(x)$ est mappée au **point le plus proche d'un dictionnaire** (*codebook*) $e_k$ ; en forward, l'embedding quantifié est passé au décodeur ; en backward le **gradient** $\nabla_zL$ est **copié sans modification** vers l'encodeur (*straight-through*). Implémentation séries temporelles : **auto-encodeur + banque de mémoire** (patterns locaux) + **discriminateur** (reconstruction indiscernable) + prédicteur d'attention en deuxième étape.

**Conclusion** : nombreuses applications (modélisation/discrimination de formes complexes, **capacité de prédiction**, **modélisation du contexte**, ex. **transport public**) ; *comment choisir entre architectures (RNN, CNN, …) ?*

---

## 11. Agrégation et attention

Rappel des schémas RNN (one-to-many … seq-to-seq).

**RNN, CNN et pooling** — variantes d'architecture : *et avec un CNN ?*

**Attention & agrégation** : *apprendre à pondérer des représentations localisées.* Vecteur $h_1,\dots,h_m$ → **vecteur d'attention** $a$ → somme pondérée.
- Beaucoup d'applications efficaces ; **peu de paramètres** ;
- **vecteur d'attention = vecteur de requête** (*query*) ;
- Comment le coder en PyTorch ? **TP** : Q1 tests préliminaires avec une attention **uniforme** ; Q2 **attention globale classique**.

**Attention avancée** : l'attention **dépend de l'entrée** ; attention **sur les sorties d'un RNN** ; **batch, padding et attention masquée**.

---

## 12. Quiz d'auto-test

> Barème indicatif sur 30 : Q1, Q2, Q3, Q4 (4 pts) · Q5, Q6, Q7, Q8 (3 pts) · Q9–Q12 (2 pts).

<details>
<summary><b>Question 1 (4 pts) :</b> Écrire l'équation du RNN simple et dire quels paramètres sont partagés. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$h_t=\tanh\big(x_tW_1+h_{t-1}W_2+b\big)$$

Les matrices $W_1,W_2$ et le biais $b$ sont **les mêmes à tous les pas de temps** (peu de paramètres, longueurs variables). $h_1$ est l'état initial.
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Pourquoi le gradient explose ou disparaît-il dans un RNN ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La rétropropagation fait intervenir $(W^\top)^{N-t}$. Avec $W=U\Sigma U^\top$, le gradient est multiplié par $\sigma_i^{N-t}$ : si $\sigma_i>1$, croissance exponentielle (explosion) ; si $\sigma_i<1$, il s'évanouit.
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Quelles solutions pour chaque problème ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
**Explosion** : *gradient clipping* (si $\lVert\nabla\rVert>1$, on renormalise ; `clip_grad_norm_` avant `step()`). **Disparition** : architecture **résiduelle** ($f(x)=x+f_0(x)$, dérivée $1+\dots$) ou **portes** (LSTM/GRU).
</details>

<details>
<summary><b>Question 4 (4 pts) :</b> Détailler les 3 portes d'un LSTM. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- Oubli : $g^{(f)}_t=\sigma[W_f(x_t\oplus h_{t-1})+b_f]$, $c^{(f)}_t=g^{(f)}_t\otimes c_{t-1}$.
- Mise à jour : $g^{(i)}_t=\sigma[\dots]$, $v^{(i)}_t=\tanh[\dots]$, $c_t=c^{(f)}_t+g^{(i)}_t\otimes v^{(i)}_t$.
- Sortie : $g^{(o)}_t=\sigma[\dots]$, $h_t=g^{(o)}_t\otimes v^{(o)}_t$ (version filtrée de $c_t$).

$\oplus$ concaténation, $\otimes$ produit terme à terme, $\sigma$ sigmoïde.
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Différence entre teacher forcing et génération libre ? Qu'est-ce que le curriculum learning ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Teacher forcing : on injecte le vrai $x_{t+1}$ ; génération libre : on injecte sa prédiction $\hat x_{t+1}$. Curriculum : commencer par teacher forcing (plus stable) puis passer à la génération libre.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Quels schémas RNN existe-t-il et pour quelles tâches ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
One-to-many (annotation d'image), many-to-one (classification de signal), many-to-many (POS/NER tagging), seq-to-seq (traduction automatique).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Pourquoi un beam search plutôt que l'argmax ? Comment évalue-t-on la génération ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Le meilleur mot à $t$ ne donne pas la meilleure séquence : on explore plusieurs hypothèses et on garde les plus probables. Évaluation : BLEU/ROUGE (F1 sur n-grammes, plus longue sous-séquence commune), BERT-score ; question ouverte (hallucinations).
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Comment réduit-on le vocabulaire pour un RNN génératif ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Niveau lettre (dimension ≈ 60) ou **byte pair encoding** (n-grammes de lettres fréquents).
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Quelle est la convention de forme d'un tenseur pour `nn.RNN` ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
*temps × échantillon (batch) × représentation* ; `seq_h` : (T, B, hidden), `last_h` : (1, B, hidden).
</details>

<details>
<summary><b>Question 10 (2 pts) :</b> Quelles approches classiques pour les séries temporelles ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
AR/ARMA/SARIMA ; ML (SVM, Ridge, RF, XGB) sur descripteurs (stats, fréquentiels, contexte) ; DTW pour des distances entre signaux (k-NN, non supervisé).
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Principe de la triplet loss non supervisée pour les séries ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Ancre = sous-série $x^{ref}$ ; positif = une de ses sous-séries ; négatif = sous-série d'une autre série. On rapproche les représentations ancre/positif et on éloigne ancre/négatif (analogie word2vec / negative sampling).
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Qu'est-ce qu'une attention en une phrase ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Une **moyenne pondérée apprise** de représentations localisées $h_i$, dont les poids (softmax d'un score avec un vecteur de requête) dépendent éventuellement de l'entrée ; peu de paramètres ; nécessite un masque avec du padding.
</details>

---

## 13. 🚩 Pièges à éviter

- **Gradient** : explosion (clipping avant `step()`) et disparition (LSTM/GRU, résiduel) — surveiller la norme du gradient.
- **Forme** `(T, B, d)` : temps en premier.
- **Teacher forcing** à l'entraînement vs **génération libre** à l'inférence : écart de distribution.
- **Padding** : ne pas le laisser peser dans la perte/attention (masque).
- **Génération gloutonne** : privilégier beam search ou échantillonnage.
- **Évaluer** un texte généré : BLEU/ROUGE sont imparfaits ; vérifier le contenu (hallucinations).
- **Bi-RNN** inutilisable pour la génération (voit le futur).
