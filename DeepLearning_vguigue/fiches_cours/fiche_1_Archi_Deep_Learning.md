# 🧠 Fiche de cours — Module 1 : Introduction au Deep Learning, architecture modulaire, prise en main PyTorch

> Source : `1-archi.pdf` (52 pages, 27 diapos numérotées). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours, ajouté pour l'autonomie) · 📝 mes notes.

## Sommaire
1. [Inspiration biologique et perceptron](#1-inspiration-biologique-et-perceptron)
2. [Limites du perceptron : combiner des neurones, non-linéarité](#2-limites-du-perceptron--combiner-des-neurones-non-linéarité)
3. [Vocabulaire de l'inférence (passe forward)](#3-vocabulaire-de-linférence-passe-forward)
4. [Apprentissage : coût, rétropropagation, descente de gradient](#4-apprentissage--coût-rétropropagation-descente-de-gradient)
5. [Calcul du gradient : règle de la chaîne](#5-calcul-du-gradient--règle-de-la-chaîne)
6. [Architecture modulaire](#6-architecture-modulaire)
7. [Zoom sur le module k : forward / backward](#7-zoom-sur-le-module-k--forward--backward)
8. [Premier réseau de neurones (fully-connected) en PyTorch](#8-premier-réseau-de-neurones-fully-connected-en-pytorch)
9. [Supervision multi-classes : MSE vs cross-entropie](#9-supervision-multi-classes--mse-vs-cross-entropie)
10. [Lutter contre le sur-apprentissage](#10-lutter-contre-le-sur-apprentissage)
11. [Améliorer le gradient](#11-améliorer-le-gradient)
12. [Conclusion](#12-conclusion)
13. [Quiz d'auto-test](#13-quiz-dauto-test)
14. [🚩 Pièges à éviter](#14--pièges-à-éviter)

---

## 1. Inspiration biologique et perceptron

- Le **réseau de neurones** est un opérateur complexe, avec une **logique d'activation et de fusion des messages** (nom évocateur et « vendeur »). Inspiration biologique plus ou moins lointaine : **fusion des messages = addition**, **activation = signe (décision)**.

### Le perceptron
Hypothèses : jeu de données $(x,y)\in\mathbb R^d\times\{-1,1\}$.

$$f_w(x)=w_0+\sum_{i=1}^{d}x_i w_i=w_0+\langle x,w\rangle,\qquad g(x)=\mathrm{sign}(x),\qquad \text{sortie}=g\big(f_w(x)\big)=\mathrm{sign}(\langle x,w\rangle)$$

- $x_i$ : descripteurs ; $w_i$ : poids ; $w_0$ : biais ; $f_w$ : fonction linéaire ; $g$ : fonction de décision.

**Problème d'apprentissage** (coût du perceptron, par exemple) : $\displaystyle \min_w\ \mathbb E_{x,y}\big[\max(0,-y\,f_w(x))\big]$. ⚠️ La diapo écrit « argmax » : par construction (erreur à minimiser), c'est un **minimum** (coquille probable).

**Algorithme du perceptron** — tant qu'il n'y a pas convergence : pour tous les exemples $(x^i,y^i)$, **si** $y^i\langle w,x^i\rangle<0$ (exemple mal classé) **alors** $w\leftarrow w+y^i x^i$.
(Équivaut à une descente de gradient sur le coût ci-dessus.)

---

## 2. Limites du perceptron : combiner des neurones, non-linéarité

- **Limite** : le perceptron ne sépare que des classes **linéairement séparables** (la diapo montre un jeu de points qu'il ne sépare pas : classes en « bande » centrale).
- **Combiner deux neurones** : $f_1(x)=w^{(1)}_{11}x_1+w^{(1)}_{21}x_2+w^{(1)}_{01}$, $f_2(x)=w^{(1)}_{12}x_1+w^{(1)}_{22}x_2+w^{(1)}_{02}$, puis $f_3(x)=w^{(2)}_{11}f_1+w^{(2)}_{21}f_2+w^{(2)}_{01}$.
- **Mais** la diapo développe $f_3$ : c'est encore une fonction **linéaire** de $(x_1,x_2)$, de la forme $x_1(\dots)+x_2(\dots)+\text{cste}$ ⇒ **équivalente à un perceptron**.

> 💡 **Il faut introduire de la non-linéarité**, sinon un empilement de couches linéaires = une seule couche linéaire.

**Quelle non-linéarité ?**
- Fonction **signe** : dérivée problématique (nulle presque partout, non définie en 0).
- Fonctions **tanh, sigmoïde, …** + **biais**. Sigmoïde : $g(x)=\dfrac{1}{1+\exp(-x)}$.

*Exemple de la diapo* : $f_1=\mathrm{sig}(3x_1-3)$, $f_2=\mathrm{sig}(3x_1+3)$, $f_3=f_2-f_1-0{,}5$ : la différence de deux sigmoïdes décalées forme une **« bosse »** qui isole la bande centrale → le problème devient séparable.

---

## 3. Vocabulaire de l'inférence (passe forward)

Avec $g(x)=\dfrac1{1+e^{-x}}$ :

$$a_1=w_{01}+w_{11}x_1+w_{21}x_2,\quad a_2=w_{02}+w_{12}x_1+w_{22}x_2,\quad z_1=g(a_1),\ z_2=g(a_2)$$
$$a_3=w^{(2)}_{01}+w^{(2)}_{11}z_1+w^{(2)}_{21}z_2,\qquad \hat y=g(a_3)$$

- **Inférence** = passe forward ; $g$ = fonction d'activation (non-linéarité du réseau) ; $a_i$ = **activation** du neurone $i$ ; $z_i$ = **sortie** du neurone $i$ (transformée non linéaire de l'activation).

---

## 4. Apprentissage : coût, rétropropagation, descente de gradient

- Objectif : **apprendre les poids**. Choix d'un coût, ici les **moindres carrés** : $L(\hat y,y)=(\hat y-y)^2$ (« pourquoi est-ce un bon choix ? »).
- Question : **comment répartir l'erreur entre les poids ?** ⇒ **rétropropagation de l'erreur**.

**Descente de gradient** — calcul des gradients partiels $\forall i,j:\ \dfrac{\partial L(\hat y,y)}{\partial w_{ij}}$ :
- **Forward** : calcul de $\hat y$ ; **Backward** : calcul des gradients ; **Optimisation** : descente de gradient.

$$w_{ij}\leftarrow w_{ij}-\varepsilon\,\frac{\partial L(\hat y,y)}{\partial w_{ij}}$$

- $\varepsilon$ : **learning rate** (pas d'apprentissage) ; $w_{ij}$ : poids reliant l'entrée $i$ au neurone $j$.

---

## 5. Calcul du gradient : règle de la chaîne

### Poids de la dernière couche (exemple de la diapo : $\hat y=0{,}5$, $y=-1$)
Hypothèse : $L=(g(a_3)-y)^2$ avec $a_3=w^{(2)}_{01}+w^{(2)}_{11}z_1+w^{(2)}_{21}z_2$.

$$\frac{\partial L}{\partial w^{(2)}_{i1}}=\frac{\partial L}{\partial a_3}\frac{\partial a_3}{\partial w^{(2)}_{i1}},\qquad \frac{\partial L}{\partial a_3}=\frac{\partial L}{\partial g(a_3)}\frac{\partial g(a_3)}{\partial a_3}=2\,g'(a_3)\,(g(a_3)-y),\qquad \frac{\partial a_3}{\partial w^{(2)}_{i1}}=z_i$$

Donc $\ \dfrac{\partial L}{\partial w^{(2)}_{i1}}=2\,g'(a_3)\,(\hat y-y)\,z_i$ ⇒ mise à jour possible.

- $g'$ : dérivée de l'activation ; $z_i$ : sortie du neurone $i$ de la couche précédente (le poids est « corrigé » proportionnellement à l'activité de la connexion).

### Poids de la première couche
$$\frac{\partial L}{\partial w^{(1)}_{i1}}=\frac{\partial L}{\partial a_1}\,x_i,\qquad \frac{\partial L}{\partial a_1}=\frac{\partial L}{\partial a_3}\,\frac{\partial a_3}{\partial a_1}=\frac{\partial L}{\partial a_3}\,g'(a_1)\,w^{(2)}_{11}$$

Donc $\dfrac{\partial L}{\partial w^{(1)}_{i1}}=\underbrace{\dfrac{\partial L}{\partial a_3}}_{\text{erreur à propager}}\underbrace{g'(a_1)\,w^{(2)}_{11}}_{\text{poids de la connexion}}\,x_i$.

### Cas général (couches intermédiaires)
Hypothèse : le neurone $j$ de la couche courante envoie sa sortie $z_j=g(a_j)$ aux neurones $k$ de la couche suivante via les poids $w_{jk}$.

$$\frac{\partial L}{\partial w_{ij}}=\frac{\partial a_j}{\partial w_{ij}}\frac{\partial L}{\partial a_j}=z_i\,\frac{\partial L}{\partial a_j},\qquad \frac{\partial L}{\partial a_j}=\sum_k\frac{\partial a_k}{\partial a_j}\frac{\partial L}{\partial a_k}=\sum_k\big(g'(a_j)\,w_{jk}\big)\frac{\partial L}{\partial a_k}$$

On note $\delta_j=\dfrac{\partial L}{\partial a_j}$ (**erreur sur $j$**).

- Lorsque l'erreur **arrive de plusieurs sources** ⇒ on **somme**.
- On exprime l'erreur de la **couche $j$** à partir de l'erreur de la **couche $k$** (la suivante) : c'est la **rétropropagation**.

> 💡 La rétropropagation = application récursive de la **règle de la chaîne** ; chaque poids est corrigé par (erreur « venue d'en aval ») × (dérivée locale) × (entrée de la connexion).

---

## 6. Architecture modulaire

Un **réseau = assemblage de modules** $M_k$. Un module :
- a des **entrées** : le résultat de la couche précédente $z^{k-1}$ ;
- a possiblement des **paramètres** $W^{(k)}$ (vus également comme des entrées) ;
- produit une **sortie** $z^k=M_k(z^{k-1},W^k)$.

**Types usuels de modules**
| Module | Définition |
|---|---|
| **Couche linéaire** (`Linear`) | transformation paramétrée $\mathbb R^d\to\mathbb R^{d'}$ : $z^k=M_k(z^{k-1},W^k)={W^k}^{\top}z^{k-1}$, $W^k\in\mathbb R^{d\times d'}$ |
| **Module d'activation** (ex. `tanh`) | fonction de $\mathbb R^d\to\mathbb R^d$ appliquée composante à composante : $\tanh(z)=(\tanh z_1,\dots,\tanh z_d)$ ; **pas de paramètres** |
| **Coût** (bloc final) | deux entrées : la **supervision** $y$ et la sortie du réseau $\hat y=z^{k}$ ; $L(\hat y,y)$ |

*(et d'autres composantes plus ésotériques.)*

**Apprentissage du réseau** : pour chaque module, calculer $\nabla_{W^k}L(\hat y,y)$ ; cas simple : paramètres constants (module d'activation) → **gradient nul** (rien à apprendre) ; rétropropagation pour les autres.

---

## 7. Zoom sur le module k : forward / backward

Hypothèses : $z^k_j$ = composante $j$ de la sortie du module $k$ ; $w^k_{ij}$ ne touche que $z^k_j$ mais impacte, via $z^k_j$, **tous** les $z^{k+1}_n$.

**Gradient par rapport aux paramètres** :
$$\nabla_{w^k_{ij}}L=\sum_j\frac{\partial L}{\partial z^k_j}\frac{\partial z^k_j}{\partial w^k_{ij}}=\sum_j\delta^k_j\,\frac{\partial M_k(z^{k-1},W^k)}{\partial w^k_{ij}}$$

**Gradient par rapport aux entrées (à propager)** :
$$\delta^k_j=\frac{\partial L}{\partial z^k_j}=\sum_n\frac{\partial L}{\partial z^{k+1}_n}\frac{\partial z^{k+1}_n}{\partial z^k_j}=\sum_n\delta^{k+1}_n\,\frac{\partial M_{k+1}(z^k,W^{k+1})}{\partial z^k_j}$$

- $\delta^k_j$ : « delta » = gradient du coût par rapport à la sortie $j$ du module $k$.
- **Dernière couche** : $\delta^{end}_j=\dfrac{\partial L(z^{end},y)}{\partial z^{end}_j}$ = gradient du coût **par rapport à la prédiction**.
- **Cas de la MSE** : $\delta^{end}_n=\dfrac{2}{N}(z_n-y_n)$.
- **Cas d'un module linéaire** (suivant) : $\nabla_{w^{k+1}}L=\sum_n\delta^{end}_n\,z^k_j$ et $\delta^k_j=\sum_n\delta^{end}_n\,w^{k+1}_{jn}$.

**Algorithme** : (1) **forward** : $z^1=M_1(x,W^1)$, $z^k=M_k(z^{k-1},W^k)$, avec **stockage des $z^k$**, jusqu'à $\hat y$ ; (2) **backward** : on part de $\delta^{end}$ et on remonte module par module.

> ⚠️ Il faut **stocker** les sorties intermédiaires du forward : elles sont réutilisées dans le backward (d'où la consommation mémoire du graphe de calcul).

---

## 8. Premier réseau de neurones (fully-connected) en PyTorch

> **Réseau fully-connected** = succession de couches linéaires et de fonctions d'activation. Idéal pour de simples tâches de classification (multi-classes aussi) ; architecture présente dans quasiment toutes les autres ; **mais** non adapté aux entrées complexes (texte, image) et **très sujet au sur-apprentissage** quand on augmente le nombre de couches.

**Architecture logicielle** — récupérer des modules existants. Création : `M = torch.nn.Linear(args)` ; inférence : `z = M(x)`.

```python
Xdim = housing_x.size(1)
net = torch.nn.Linear(Xdim, 1)                       # couche linéaire Xdim -> 1 (décision linéaire simple)
mseloss = torch.nn.MSELoss()                         # fonction de coût
optim = torch.optim.SGD(params=net.parameters(), lr=EPS)   # optimiseur (+ récupération des paramètres)
yhat = net(housing_x)                                # inférence = appel direct du module
loss = mseloss(net(housing_x).view(-1,1), housing_y.view(-1,1))
loss.backward(); optim.step()
```

> **Un réseau de neurones est un module** : définition des couches (`__init__`) + inférence (`forward`) … **et c'est tout (backward automatique !)**.

```python
class DeuxCouches(torch.nn.Module):               # héritage
    def __init__(self):                           # attributs
        super(DeuxCouches, self).__init__()
        self.un   = torch.nn.Linear(Xdim, 5)
        self.act  = torch.nn.Tanh()
        self.deux = torch.nn.Linear(5, 1)
    def forward(self, x):                         # inférence
        return self.deux(self.act(self.un(x)))
netDeuxCouches = DeuxCouches()                    # instanciation
netDeuxCouches(housing_x)                         # usage en inférence
```

**Boucle d'apprentissage standard** (la standardisation des modules ⇒ standardisation du processus) :
```python
housing_x, housing_y = ...       # données
net = ...                        # réseau de neurones
loss = ...                       # coût
optim = ...                      # optimiseur
for i in range(EPOCHS):
    loss = mseloss(net(housing_x).view(-1,1), housing_y.view(-1,1))
    print(f"iteration : {i}, loss : {loss}")
    optim.zero_grad()            # mise à zéro des gradients
    loss.backward()              # calcul des gradients
    optim.step()                 # MAJ des paramètres
```

---

## 9. Supervision multi-classes : MSE vs cross-entropie

**Principe** : pour $K$ classes → **$K$ sorties** et supervision par un **vecteur one-hot** $y=(0,0,\dots,1,\dots,0)$ ($y_k=1$ pour l'indice de la bonne classe, 0 ailleurs).

### Utilisation de la MSE (problématique)
- Sortie du réseau entre 0 et 1 ⇒ utilisation d'une **sigmoïde**. Exemple de la diapo : sortie $(0{,}4;0{,}8;0{,}2;0{,}1)$ vs $y=(0,1,0,0)$ → termes d'erreur $0{,}16;0{,}04;0{,}04;0{,}01$.
- Mais : la **similarité au vecteur de sortie n'est pas critique** — c'est l'**argmax** qui compte. On veut **maximiser** la sortie de la bonne classe et **minimiser** les autres.

### Coût cross-entropique (= maximum de vraisemblance multinomial)
**SoftMax** (sorties → distribution + renforcement du max) — hypothèse : $z\in\mathbb R^K$ (scores bruts, *logits*) :
$$\mathrm{SoftMax}(z)_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}},\qquad \sum_{i=1}^{K}\mathrm{SoftMax}(z)_i=1$$

**Cross-entropie** :
$$CE(y,\hat y)=-\sum_{i=1}^{K}y_i\log\hat y_i$$
Si $y$ est **one-hot** de la classe $k$ : $CE(y,\hat y)=-\log\hat y_k$.

**Combinaison SoftMax + cross-entropie** :
$$CE\big(y,\mathrm{SoftMax}(z)\big)=-z_k+\log\sum_{j=1}^{K}e^{z_j},\qquad \frac{\partial\,CE(y,\mathrm{SoftMax}(z))}{\partial z_i}=\mathrm{SoftMax}(z)_i-\mathbf 1_{i=k}$$

- $z_k$ : score de la bonne classe ; $\mathbf 1_{i=k}$ : indicatrice (1 si $i$ est la bonne classe, 0 sinon). Le gradient est simplement **« probabilité prédite − cible »**.

**Cross-entropie binaire (BCE)** — pour le **multi-label** en particulier : cross-entropie sur chaque sortie, considérée comme des **Bernoulli indépendantes** :
$$BCE(y,\hat y)=-\sum_{i=1}^{K}\Big[y_i\log\hat y_i+(1-y_i)\log(1-\hat y_i)\Big]$$

> 🔎 En PyTorch : `nn.CrossEntropyLoss` (logits bruts + indices de classe) = `LogSoftmax` + `NLLLoss` ; `nn.BCEWithLogitsLoss` pour le multi-label (sigmoïde intégrée). Ne pas ajouter de softmax avant `CrossEntropyLoss`.

---

## 10. Lutter contre le sur-apprentissage

Techniques visant toutes à **régulariser** le réseau :

| Technique | Principe |
|---|---|
| **Régularisation des couches** (L1, L2) | terme de pénalisation en $\lVert W\rVert_p$ sur les poids des couches |
| **Dropout** | retirer, pendant une itération, quelques neurones au hasard → **robustesse** accrue |
| **Data augmentation** | perturber les données d'entrée pour améliorer la généralisation ; outils paramétrables et automatisés dans `torchvision` |
| **Gradient clipping** | la norme du gradient rétropropagé est **bornée** pour éviter une trop grosse instabilité |

> 💡 *Data augmentation* : idée simple pour **régulariser par la masse et les variations**.

---

## 11. Améliorer le gradient

### Gradient vanishing (disparition du gradient)
Le gradient tend à disparaître : (i) dans les couches **éloignées de la supervision** ; (ii) dans les **sigmoïdes saturées**.
- Remède : activation spécifique **ReLU** : $\mathrm{ReLU}(x)=\max(0,x)$ — garde un **gradient fort** lorsque le neurone est activé.

### Topologie de l'espace de recherche et gradient
- **BatchNorm** : pour une couche, centrer/normaliser **chaque sortie** (estimation sur chaque mini-batch).
- **LayerNorm** : à la sortie d'une couche, normalisation de **chaque exemple séparément** sur ses dimensions.
- *Centrer les données = meilleure topologie pour l'apprentissage.*

---

## 12. Conclusion

- **Multiplication des modules et des hyper-paramètres** : l'architecture = beaucoup d'hyper-paramètres → besoin de **normalisation** pour mieux comparer les architectures et avoir des *a priori* sur les bons paramètres.
- **Lexique des modules** : comprendre les enjeux, savoir **lire les articles**.
- **Outils supplémentaires** : `DataLoader`, *check-pointing*, `TensorBoard`.

---

## 13. Quiz d'auto-test

> Barème indicatif sur 30 : Q1, Q3, Q5 (4 pts) · Q2, Q4, Q6, Q7 (3 pts) · Q8–Q10 (2,5 pts) · Q11 (2 pts).

<details>
<summary><b>Question 1 (4 pts) :</b> Pourquoi deux couches linéaires empilées sans activation ne valent-elles pas mieux qu'une seule ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Parce que la composition de deux fonctions linéaires est une fonction **linéaire** : en développant $f_3$ on retrouve $x_1(\dots)+x_2(\dots)+\text{cste}$, soit un **perceptron**. Il faut une **non-linéarité** (tanh, sigmoïde, ReLU) entre les couches.
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Pourquoi ne pas utiliser la fonction signe comme activation ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Sa dérivée est nulle presque partout et non définie en 0 ⇒ **le gradient ne peut pas se propager**. On utilise tanh, sigmoïde ou ReLU (dérivables, avec un biais).
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Écrire la mise à jour d'un poids de la dernière couche pour $L=(g(a_3)-y)^2$. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\frac{\partial L}{\partial w^{(2)}_{i1}}=2\,g'(a_3)\,(\hat y-y)\,z_i\qquad\Rightarrow\qquad w^{(2)}_{i1}\leftarrow w^{(2)}_{i1}-\varepsilon\,2g'(a_3)(\hat y-y)z_i$$

Règle de la chaîne : $\partial L/\partial a_3\times\partial a_3/\partial w^{(2)}_{i1}$, avec $\partial a_3/\partial w^{(2)}_{i1}=z_i$.
</details>

<details>
<summary><b>Question 4 (3 pts) :</b> Que représente $\delta^k_j$ et comment se calcule-t-il à partir de la couche suivante ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
$\delta^k_j=\partial L/\partial z^k_j$ est l'**erreur sur la sortie $j$ du module $k$**. Récursion :

$$\delta^k_j=\sum_n\delta^{k+1}_n\,\frac{\partial M_{k+1}(z^k,W^{k+1})}{\partial z^k_j}$$

Pour un module linéaire : $\delta^k_j=\sum_n\delta^{k+1}_n w^{k+1}_{jn}$. Si l'erreur arrive de plusieurs sources, on **somme**.
</details>

<details>
<summary><b>Question 5 (4 pts) :</b> Pourquoi préfère-t-on la cross-entropie à la MSE pour de la classification ? Donner le gradient SoftMax + CE. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
En classification, c'est l'**argmax** qui compte, pas la similarité au vecteur one-hot : la CE (maximum de vraisemblance multinomial) **maximise la sortie de la bonne classe et minimise les autres**. Pour $y$ one-hot de la classe $k$ :

$$CE=-z_k+\log\sum_j e^{z_j},\qquad \frac{\partial CE}{\partial z_i}=\mathrm{SoftMax}(z)_i-\mathbf 1_{i=k}$$
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Quand utiliser la BCE plutôt que SoftMax+CE ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Pour le **multi-label** : chaque sortie est une Bernoulli indépendante ($\hat y_i\in(0,1)$ via sigmoïde) ; plusieurs classes peuvent être vraies simultanément. SoftMax+CE suppose une seule bonne classe parmi $K$ (distribution qui somme à 1).
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Quels sont les 4 leviers contre le sur-apprentissage cités et leur principe ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
(1) Régularisation L1/L2 (pénalité sur les poids) ; (2) Dropout (neurones retirés au hasard à chaque itération) ; (3) Data augmentation (perturber les entrées) ; (4) Gradient clipping (borner la norme du gradient rétropropagé).
</details>

<details>
<summary><b>Question 8 (2,5 pts) :</b> Quelles sont les deux causes du gradient vanishing et le remède associé à l'activation ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Couches éloignées de la supervision et sigmoïdes saturées. La **ReLU** ($\max(0,x)$) conserve un gradient fort quand le neurone est actif.
</details>

<details>
<summary><b>Question 9 (2,5 pts) :</b> Différence entre BatchNorm et LayerNorm ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
BatchNorm : centre/normalise chaque sortie avec des statistiques estimées **sur le mini-batch**. LayerNorm : normalise **chaque exemple séparément** sur ses dimensions.
</details>

<details>
<summary><b>Question 10 (2,5 pts) :</b> Quelles sont les étapes d'une itération de la boucle d'apprentissage standard, dans l'ordre ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Forward (calcul du coût) → `optim.zero_grad()` → `loss.backward()` → `optim.step()`.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Qu'est-ce qu'un module et que faut-il définir pour un réseau ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Un module a des entrées, éventuellement des paramètres, et produit une sortie. Pour un réseau (classe héritant de `torch.nn.Module`), on définit les couches dans `__init__` et l'inférence dans `forward` ; le backward est automatique.
</details>

---

## 14. 🚩 Pièges à éviter

- **Empiler du linéaire** sans activation = un perceptron.
- **Gradient vanishing** avec sigmoïde/tanh profondes ; privilégier ReLU, normalisations, connexions résiduelles.
- **MSE + one-hot en classification** : pas adapté ; utiliser la cross-entropie (sur logits).
- **Softmax + `CrossEntropyLoss`** : double softmax ; sortie = logits.
- **Oublier `zero_grad()`** : les gradients s'accumulent.
- **Plus de couches = plus de sur-apprentissage** : régulariser (L2, dropout, augmentation).
- **Hyper-paramètres** nombreux : comparer les architectures de façon normalisée.
