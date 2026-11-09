# 📘 Fiche de révision — Module 1 : Introduction au Deep Learning & architectures en PyTorch

> Cours de Vincent Guigue (inspiré de N. Baskiotis & B. Piwowarski) · AgroParisTech – IODAA · 34 diapos (`Architecture.pdf`)
>
> **Légende des encadrés** : 💡 idée clé · ⚠️ attention · 🚩 piège fréquent · 🔎 complément (absent du cours) · 📝 notes perso

---

## Sommaire

1. [Introduction : du perceptron au réseau de neurones](#1-introduction--du-perceptron-au-réseau-de-neurones)
2. [Rétro-propagation du gradient (chain rule)](#2-rétro-propagation-du-gradient-chain-rule)
3. [Architecture modulaire](#3-architecture-modulaire)
4. [Premier réseau de neurones en PyTorch](#4-premier-réseau-de-neurones-en-pytorch)
5. [Autres modules importants](#5-autres-modules-importants)
6. [Optimisation](#6-optimisation)
7. [Conclusion : hyper-paramètres et outils](#7-conclusion--hyper-paramètres-et-outils)
8. [Pièges à éviter](#8-pièges-à-éviter)
9. [Quiz auto-test](#9-quiz-auto-test)

---

## 1. Introduction : du perceptron au réseau de neurones

### 1.1 Inspiration biologique [plus ou moins lointaine] (diapo 2)

| Réseau de neurones biologique | Transposition |
|---|---|
| Opérateur complexe ; logique d'activation et de fusion des messages ; nom évocateur et « vendeur » | Neurone A → axone → synapses → dendrites du neurone B |
| | **Feature** = entrée ; **fusion de message = addition** ; **activation = signe** (= décision) |

### 1.2 Le perceptron (diapo 3)

Hypothèses : jeu de données $(\mathbf x, y) \in \mathbb R^d \times \{-1, 1\}$ (classification binaire).

$$f_{\mathbf w}(\mathbf x) = w_0 + \sum_{i=1}^{d} x_i w_i = w_0 + \langle \mathbf x, \mathbf w\rangle$$

- $\mathbf x = (x_1,\dots,x_d)$ : entrée ; $\mathbf w = (w_1,\dots,w_d)$ : poids ; $w_0$ : biais ;
- $\langle \mathbf x, \mathbf w \rangle$ : produit scalaire.

- **Fonction de décision** : $g(x) = \mathrm{sign}(x)$ → **sortie** : $g(f(\mathbf x)) = \mathrm{sign}(\langle \mathbf x, \mathbf w\rangle)$.
- **Problème d'apprentissage** (tel qu'écrit sur la diapo) :

$$\arg\max_{\mathbf w}\; \mathbb E_{\mathbf x, y}\big[\max(0, -y f_{\mathbf w}(\mathbf x))\big]$$

- $\max(0, -y f_{\mathbf w}(\mathbf x))$ : **coût du perceptron**, nul si l'exemple est bien classé ($y f_{\mathbf w}(\mathbf x) > 0$), positif sinon ;
- $\mathbb E_{\mathbf x,y}$ : espérance sur la distribution des données.

> 🔎 Ce coût est une **erreur** qu'on cherche à **minimiser** : il faut lire $\arg\min$ (ou $\arg\max$ de son opposé). La descente de gradient sur ce coût redonne exactement l'algorithme ci-dessous.

**Algorithme du perceptron** :
- Tant qu'il n'y a pas convergence :
  - pour tous les exemples $(\mathbf x^i, y^i)$ : si $y^i \times \langle \mathbf w, \mathbf x^i\rangle < 0$ (mal classé), alors $\mathbf w \leftarrow \mathbf w + \varepsilon\, y^i \mathbf x^i$.
- = **descente de gradient sur le coût**.

### 1.3 Limites du perceptron (diapo 4)

*Est-il capable de séparer ces données ?* Trois jeux montrés : nuages 3D séparables par un plan ; **damier/XOR** (rouge/vert mélangés) ; **3 groupes alignés** (violet – jaune – violet sur l'axe $x_1$).

> 💡 Le perceptron ne trace qu'**une frontière linéaire** (un hyperplan) : il échoue dès que les classes ne sont pas linéairement séparables (XOR, classe « au milieu »).

### 1.4 Combinons deux neurones (diapo 5)

Réseau : entrées $x_1, x_2$ → neurones $f_1, f_2$ → neurone de sortie $f_3$.

$$f_1(\mathbf x) = w^{(1)}_{11}x_1 + w^{(1)}_{21}x_2 + w^{(1)}_{01}, \qquad f_2(\mathbf x) = w^{(1)}_{12}x_1 + w^{(1)}_{22}x_2 + w^{(1)}_{02}$$

$$f_3(\mathbf x) = w^{(2)}_{11} f_1(\mathbf x) + w^{(2)}_{21} f_2(\mathbf x) + w^{(2)}_{01}$$

- $w^{(c)}_{ij}$ : poids de la couche $c$ reliant l'entrée $i$ au neurone $j$ ; $w^{(c)}_{0j}$ : biais du neurone $j$.

*Combiner des neurones ⇒ suffisant ?* En développant :

$$f_3(\mathbf x) = x_1\big(w^{(2)}_{11}w^{(1)}_{11} + w^{(2)}_{21}w^{(1)}_{12}\big) + x_2\big(w^{(2)}_{11}w^{(1)}_{21} + w^{(2)}_{21}w^{(1)}_{22}\big) + w^{(2)}_{01} + w^{(2)}_{11}w^{(1)}_{01} + w^{(2)}_{21}w^{(1)}_{02}$$

> 💡 **Non !** C'est encore une fonction **linéaire** de $\mathbf x$ : composer des couches linéaires = une seule couche linéaire. **Il faut introduire de la non-linéarité**, sinon le réseau est équivalent à un perceptron.

### 1.5 Non-linéarité (diapo 6)

On insère une fonction $g$ après chaque neurone caché : $f_1 \to g_1$, $f_2 \to g_2$.

- **Quelle non-linéarité ?** La fonction **signe** ? ⇒ **dérivée problématique** (nulle partout, non définie en 0 : pas de gradient exploitable).
- ⇒ Fonctions **tanh**, **sigmoïde**, … **+ biais** :

$$g(x) = \frac{1}{1 + \exp(-x)}$$

**Exemple travaillé** (courbes de la diapo) : avec $f_1 = \mathrm{sig}(3x_1 - 3)$, $f_2 = \mathrm{sig}(3x_1 + 3)$ et $f_3 = f_2 - f_1 - 0{,}5$, on obtient une **bosse** : $f_3 > 0$ pour $x_1 \in [-1, 1]$ environ, $f_3 < 0$ ailleurs → le réseau isole la classe **jaune du milieu**, impossible pour un perceptron.

### 1.6 Vocabulaire de l'inférence (diapo 7)

Avec $g(x) = \frac{1}{1+\exp(-x)}$ :

| Grandeur | Formule |
|---|---|
| $a_1$ | $w_{01} + w_{11}x_1 + w_{21}x_2$ |
| $a_2$ | $w_{02} + w_{12}x_1 + w_{22}x_2$ |
| $z_1$, $z_2$ | $g(a_1)$, $g(a_2)$ |
| $a_3$ | $w^{(2)}_{01} + w^{(2)}_{11}z_1 + w^{(2)}_{21}z_2$ |
| $\hat y$ | $g(a_3)$ |

> 💡 **Vocabulaire**
> - **Inférence** : *passe forward* ;
> - $g$ : **fonction d'activation** (non-linéarité du réseau) ;
> - $a_i$ : **activation** du neurone $i$ (somme pondérée) ;
> - $z_i$ : **sortie** du neurone $i$ (transformé non linéaire de l'activation).

### 1.7 Apprentissage (diapo 8)

**Objectif : apprendre les poids.**
- Choix d'un coût : **moindres carrés** $L(\hat y, y) = (\hat y - y)^2$ *[pourquoi est-ce un bon choix ?]*
- Mais **comment répartir l'erreur entre les poids** ? ⇒ **rétro-propagation de l'erreur**.

> 🔎 Pourquoi un bon choix : il est **dérivable partout**, convexe en $\hat y$, pénalise fortement les grosses erreurs, et correspond au **maximum de vraisemblance** sous hypothèse de bruit gaussien.

---

## 2. Rétro-propagation du gradient (chain rule)

### 2.1 Descente de gradient (diapo 9)

Objectif : calculer les gradients partiels par rapport aux paramètres :

$$\forall i, j, \qquad \frac{\partial L(\hat y, y)}{\partial w_{ij}}$$

- **Forward** : calcul de $\hat y$ (entre autres : on stocke aussi les $a$ et $z$) ;
- **Backward** : calcul des gradients ;
- **Optimisation** : descente de gradient.

$$w_{ij} \leftarrow w_{ij} - \varepsilon \,\frac{\partial L(\hat y, y)}{\partial w_{ij}}$$

- $\varepsilon$ : **learning rate** (pas d'apprentissage).

### 2.2 Dernière couche (diapo 10)

Exemple : forward donne $\hat y = 0{,}5$ avec $y = -1$. *Backward*, poids de la **dernière couche** : $\nabla_{w^{(2)}_{ij}} L(\hat y, y)$.

$$L(\hat y, y) = (g(a_3) - y)^2 = \Big(g\big(w^{(2)}_{01} + w^{(2)}_{11}z_1 + w^{(2)}_{21}z_2\big) - y\Big)^2$$

$$\frac{\partial L}{\partial w^{(2)}_{i1}} = \frac{\partial L}{\partial a_3}\,\frac{\partial a_3}{\partial w^{(2)}_{i1}}
\quad\text{avec}\quad
\frac{\partial L}{\partial a_3} = \frac{\partial L}{\partial g(a_3)}\frac{\partial g(a_3)}{\partial a_3} = 2g'(a_3)\big(g(a_3) - y\big), \qquad
\frac{\partial a_3}{\partial w^{(2)}_{i1}} = z_i$$

$$\boxed{\frac{\partial L}{\partial w^{(2)}_{i1}} = 2g'(a_3)(\hat y - y)\,z_i} \quad\Longrightarrow\quad \text{mise à jour possible}$$

- $g'(a_3)$ : dérivée de l'activation en $a_3$ ; $z_i$ : sortie du neurone caché $i$ (avec $z_0 = 1$ pour le biais).

### 2.3 Première couche (diapo 11)

*Backward*, poids de la **première couche** $w^{(1)}_{i1}$ (par exemple) :

$$\frac{\partial L}{\partial w_{i1}} = \frac{\partial L}{\partial a_1}\,\frac{\partial a_1}{\partial w_{i1}}
\quad\text{avec}\quad
\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial a_3}\frac{\partial a_3}{\partial a_1} = \frac{\partial L}{\partial a_3}\,g'(a_1)\,w^{(2)}_{11}, \qquad
\frac{\partial a_1}{\partial w_{i1}} = \frac{\partial\big(w^{(1)}_{01} + w^{(1)}_{11}x_1 + w^{(1)}_{21}x_2\big)}{\partial w^{(1)}_{i1}} = x_i$$

$$\underbrace{\frac{\partial L}{\partial w_{i1}}}_{\text{correction de } w_{i1}} = \frac{\partial L}{\partial a_1}x_i = \underbrace{\frac{\partial L}{\partial a_3}}_{\text{erreur à propager}}\;\underbrace{g'(a_1)\,w^{(2)}_{11}}_{\text{poids de la connexion}}\;x_i$$

> ⚠️ La diapo écrit « $g'(a_1)w_{13}$ » dans la dernière ligne : il s'agit du même terme $g'(a_1)\,w^{(2)}_{11}$ (poids de la connexion $a_1 \to a_3$) que dans la ligne précédente.

### 2.4 Cas général dans les couches intermédiaires (diapo 12)

Schéma : $z_i \xrightarrow{w_{ij}} a_j \to z_j = g(a_j) \xrightarrow{w_{jk}} a_k$.

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial a_j}{\partial w_{ij}}\frac{\partial L}{\partial a_j} = z_i\,\frac{\partial L}{\partial a_j}
\qquad\qquad
\frac{\partial L}{\partial a_j} = \sum_k \frac{\partial a_k}{\partial a_j}\frac{\partial L}{\partial a_k}$$

$$\underbrace{\frac{\partial L}{\partial a_j}}_{\text{erreur sur } j} = \sum_k \big(g'(a_j)\,w_{jk}\big)\underbrace{\frac{\partial L}{\partial a_k}}_{\text{erreur à propager}}
\qquad\text{On note : } \delta_j = \frac{\partial L}{\partial a_j}$$

- $\delta_j$ : erreur sur le neurone $j$ ; $k$ parcourt les neurones de la couche suivante reliés à $j$.

> ⚠️ La diapo écrit $g'(a_k)$ dans la somme : c'est **$g'(a_j)$** qui est correct, car $a_k = \sum_j w_{jk}\,g(a_j)$ donc $\frac{\partial a_k}{\partial a_j} = w_{jk}\,g'(a_j)$ (et $g'(a_j)$ peut d'ailleurs sortir de la somme).

- Lorsque l'erreur **arrive de plusieurs sources** ⇒ **somme**.
- **Expression de l'erreur de la couche $j$ par rapport à l'erreur de la couche $k$** → on remonte couche par couche.

---

## 3. Architecture modulaire

### 3.1 Réseau = assemblage de modules (diapo 13)

Le réseau précédent se découpe en modules $M_1$ (linéaire, $W^{(1)}$), $M_2$ (activation), $M_3$ (linéaire, $W^{(3)}$), $M_4$ (activation), puis le coût $L$ (avec la *ground truth* $y$).

> 💡 **Un module $M^k$** :
> - a des **entrées** : le résultat de la couche précédente $\mathbf z^{k-1}$ ;
> - a **possiblement des paramètres** $W^{(k)}$ [vus également comme des entrées] ;
> - produit une **sortie** $\mathbf z^k$.

### 3.2 Types usuels de modules (diapo 14)

| Module | Définition | Paramètres |
|---|---|---|
| **Couche linéaire** `[Linear]` | Transformation paramétrée de $\mathbb R^d$ vers $\mathbb R^{d'}$ : $\mathbf z^k = M^k(\mathbf z^{k-1}, W^k) = W^{k\,\top}\mathbf z^{k-1}$, avec $W^k \in \mathbb R^d \times \mathbb R^{d'}$ ; composante : $z^{k+1}_i = w_{1i}z^k_1 + w_{2i}z^k_2 + \dots + w_{di}z^k_d$ | $W$ (et biais) |
| **Module d'activation** | Fonction de $\mathbb R^d$ vers $\mathbb R^d$, appliquée composante par composante : $M^k(\mathbf z^{k-1}, \emptyset) = \tanh(\mathbf z^{k-1}) = (\tanh(z^{k-1}_1), \dots, \tanh(z^{k-1}_d))$ | **aucun** ($\emptyset$) |
| **Un coût** | **Bloc final** : deux entrées, la **supervision** $y$ et la **sortie du réseau** $\hat y$ → $L(\hat y, y)$ | aucun |

… et d'autres composantes plus ésotériques.

Chaîne générale : $\mathbf x \to M^1 \to \mathbf z^1 = M^1(\mathbf x, W^1) \to M^2 \to \dots \to \hat{\mathbf y} = \mathbf z^k = M^k(\mathbf z^{k-1}, W^k) \to L(\hat{\mathbf y}, \mathbf y)$ (avec éventuellement des branches parallèles).

### 3.3 Apprentissage du réseau (diapo 15)

**Pour apprendre le réseau :**
- pour chaque module : calculer $\nabla_{W^k} L(\hat y, y)$ ;
- cas simple : paramètres constants (module d'activation) → le gradient est nul (il n'y a **rien à apprendre** pour ce module) ;
- **rétro-propagation** pour les autres.

### 3.4 Rétro-propagation pour $M^k$, $\mathbf z^k = M(\mathbf z^{k-1}, W^k)$ (diapo 16)

$$\frac{\partial L}{\partial w^k_{ij}} = \sum_j \frac{\partial L}{\partial z^k_j}\frac{\partial z^k_j}{\partial w^k_{ij}} = \frac{\partial L}{\partial z^k_j}\frac{\partial z^k_j}{\partial w^k_{ij}} = \frac{\partial L}{\partial z^k_j}\,\frac{\partial M^k(\mathbf z^{k-1}, W^k)}{\partial w^k_{ij}} \qquad [w^k_{ij} \text{ n'impacte que } z^k_j]$$

$$\frac{\partial L}{\partial z^k_j} = \sum_n \frac{\partial L}{\partial z^{k+1}_n}\frac{\partial z^{k+1}_n}{\partial z^k_j} = \sum_n \frac{\partial L}{\partial z^{k+1}_n}\,\frac{\partial M^{k+1}(\mathbf z^k, W^{k+1})}{\partial z^k_j} \qquad [z^k_j \text{ impacte tous les } z^{k+1}_n]$$

On introduit :

$$\delta^k_j = \frac{\partial L}{\partial z^k_j} = \sum_n \delta^{k+1}_n\,\frac{\partial M^{k+1}(\mathbf z^k, W^{k+1})}{\partial z^k_j} \qquad;\qquad \frac{\partial L}{\partial w^k_{ij}} = \delta^k_j\,\frac{\partial M^k(\mathbf z^{k-1}, W^k)}{\partial w^k_{ij}}$$

**Dernière couche** : $\delta^{end}_j = \dfrac{\partial L(\mathbf z^{end}, y)}{\partial z^{end}_j}$, le **gradient du coût par rapport à la prédiction**.

- $\delta^k_j$ : erreur sur la sortie $j$ du module $k$ ; $n$ parcourt les sorties du module $k+1$.

> 💡 Chaque module n'a besoin que de **deux dérivées locales** : par rapport à ses **paramètres** (pour sa mise à jour) et par rapport à ses **entrées** (pour transmettre $\delta$ au module précédent). C'est ce qui rend l'architecture **modulaire** (et ce que fait l'autograd de PyTorch).

### 3.5 Zoom sur le module $k$ : forward / backward (diapo 17)

**Forward** (input $\mathbf x$) :
- $\mathbf z^1 = M_1(\mathbf x, W^1)$ ; $\mathbf z^k = M_k(\mathbf z^{k-1}, W^k)$ ; **+ stockage des $\mathbf z^k$** ; jusqu'à $\hat{\mathbf y}$.

**Backward** :
1. Cas de la **MSE** : $\hat{\mathbf y}_m \equiv \mathbf z^{end}$, $\quad \delta^{end}_j = \dfrac{\partial L(\mathbf z^{end}, y)}{\partial z^{end}_n} = \dfrac{2}{N}(z^{end}_n - y_n)$.
2. Supposons que $\mathbf z^{k+1} = \hat{\mathbf y}_m = \mathbf z^{end}$ ⇒ $\delta^{k+1}_n = \delta^{end}_n = \frac{2}{N}(z^{end}_n - y_n)$. Gradient, **dans le cas d'un module linéaire** :

$$\frac{\partial L}{\partial w^{k+1}_{jn}} = \delta^{end}_n\,\frac{\partial z^{end}_n}{\partial w^{k+1}_{jn}} = \delta^{end}_n\,\frac{\partial M^{k+1}(\mathbf z^k, W^{k+1})}{\partial w^{k+1}_{jn}} = \delta^{end}_n\, z^k_j$$

3. Propagation vers le module $k$ (**module linéaire**) :

$$\delta^k_j = \frac{\partial L}{\partial z^k_j} = \sum_n \frac{\partial L}{\partial z^{k+1}_n}\frac{\partial z^{k+1}_n}{\partial z^k_j} = \sum_n \delta^{end}_n\,\frac{\partial M^{k+1}(\mathbf z^k, W^{k+1})}{\partial z^k_j} = \sum_n \delta^{end}_n\, w^{k+1}_{jn}$$

4. On obtient ainsi successivement $\nabla_{W^{k+1}}L$, $\nabla_{W^k}L$, $\nabla_{W^{k-1}}L$… en remontant.

- $N$ : nombre d'exemples (ou de sorties) moyennés dans la MSE ; $y_n$ : cible ; $m$ : indice de l'exemple.

> 💡 Le **forward stocke** les $\mathbf z^k$ ; le **backward** les réutilise ($z^k_j$ apparaît dans le gradient des poids). C'est pourquoi l'entraînement consomme beaucoup de mémoire.

---

## 4. Premier réseau de neurones en PyTorch

### 4.1 Réseau *fully-connected* (diapo 18)

Un réseau *fully-connected* est une **succession de couches linéaires et de fonctions d'activation** (ex. Input → Dense #1 → Dense #2 → Softmax → classes 1/2/3).

> 💡 **Propriétés**
> - Idéal pour les **simples tâches de classification** (multi-classes également) ;
> - architecture que l'on **retrouve dans quasiment toutes les autres** architectures ;
> - mais **non adapté sur des entrées complexes** (texte, image) ;
> - **très sujet au sur-apprentissage** avec l'augmentation du nombre de couches.

### 4.2 Architecture logicielle (diapo 19)

Récupération de **modules existants** pour une construction rapide. **Création** : `M = torch.nn.Linear(args)` ; **inférence** : `z = M(x)`.

Réseau à une couche linéaire (= décision linéaire simple) :

```python
Xdim = housing_x.size(1)
## Création d'une couche linéaire de dimension Xdim -> 1
net = torch.nn.Linear(Xdim, 1)           # récupération d'un module
## Fonction de coût
mseloss = torch.nn.MSELoss()
## Optimiseur (& récupération des paramètres)
optim = torch.optim.SGD(params=net.parameters(), lr=EPS)

yhat = net(housing_x)                    # inférence des modules = appel direct
loss = mseloss(net(housing_x).view(-1, 1), housing_y.view(-1, 1))
loss.backward()
optim.step()
```

### 4.3 Un réseau de neurones est un module (diapo 20)

> 💡 **Module** = définition des **couches** + **inférence** (= `forward`, définition des entrées/sorties)… **et c'est tout (backward automatique !)**.

```python
class DeuxCouches(torch.nn.Module):            # extension / héritage
    def __init__(self):                        # attributs
        super(DeuxCouches, self).__init__()
        self.un   = torch.nn.Linear(Xdim, 5)
        self.act  = torch.nn.Tanh()
        self.deux = torch.nn.Linear(5, 1)
    def forward(self, x):                      # inférence
        return self.deux(self.act(self.un(x)))

netDeuxCouches = DeuxCouches()                 # instanciation
netDeuxCouches(housing_x)                      # usage en inférence
```

### 4.4 Convergence vers une architecture d'apprentissage standard (diapo 21)

La **standardisation des modules** (& architecture logicielle) ⇒ **standardisation du processus d'apprentissage** :

```python
# données :
housing_x, housing_y = ...
# définition de :
net   = ...   # réseau de neurones
loss  = ...   # coût
optim = ...   # optimiseur

for i in range(EPOCHS):                       # boucle d'apprentissage
    loss = mseloss(net(housing_x).view(-1, 1), housing_y.view(-1, 1))
    print(f"iteration : {i}, loss : {loss}")
    optim.zero_grad()                         # mise à zéro des gradients
    loss.backward()                           # calcul des gradients
    optim.step()                              # MAJ des paramètres
```

---

## 5. Autres modules importants

### 5.1 Supervision multi-classes (diapo 22)

Quand il faut prédire **$K$ classes** :
- **$K$ sorties** $\hat y_1, \dots, \hat y_K$ ;
- utilisation de **vecteurs 1-hot** pour la supervision :

$$\mathbf y = (0, 0, \dots, 1, \dots, 0) \quad\text{avec } y_i = 0 \text{ pour } i \text{ différent de la bonne classe}, \; y_k = 1 \text{ pour } k \text{ l'indice de la bonne classe}$$

### 5.2 Utilisation de la MSE (diapo 23)

Exemple : sortie (après $\sigma$) $(0{,}4;\ 0{,}8;\ 0{,}2;\ \dots;\ 0{,}1)$, cible $(0, 1, 0, \dots, 0)$ → erreurs carrées $(0{,}16;\ 0{,}04;\ 0{,}04;\ \dots;\ 0{,}01)$.

> ⚠️ **Fonction de coût problématique**
> - Sortie du réseau entre 0 et 1 ⇒ utilisation d'une **sigmoïde** ;
> - mais : la **similarité au vecteur de sortie** n'est **pas critique** ⇒ c'est l'**argmax** qui est important ;
> - on veut **++ maximisation de la sortie de la bonne classe** | **− minimisation des autres sorties**.

### 5.3 Coût cross-entropique (= MV multinomiale) (diapo 24)

**SoftMax** : sorties ⇒ **distribution** + **renforcement du max**.

Hypothèses : $\mathbf z \in \mathbb R^K$ (scores bruts du réseau), $i \in \{1..K\}$.

$$\mathrm{SoftMax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_{i=1}^{K}\mathrm{SoftMax}(\mathbf z)_i = 1$$

- $z_i$ : score (logit) de la classe $i$ ; $K$ : nombre de classes.

**Coût cross-entropique** — hypothèses : $\mathbf y$ distribution cible, $\hat{\mathbf y}$ distribution prédite :

$$CE(\mathbf y, \hat{\mathbf y}) = -\sum_{i=1}^{K} y_i \log(\hat y_i)$$

- Si $\mathbf y$ est un vecteur **one-hot de la classe $k$** : $CE(\mathbf y, \hat{\mathbf y}) = -\log(\hat y_k)$.
- **Combinaison SoftMax et cross-entropie** :

$$CE(\mathbf y, \mathrm{SoftMax}(\mathbf z)) = -z_k + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big), \qquad \frac{\partial CE(\mathbf y, \mathrm{SoftMax}(\mathbf z))}{\partial z_i} = \mathrm{SoftMax}(\mathbf z)_i - \mathbb 1_{i=k}$$

- $\mathbb 1_{i=k}$ : vaut 1 si $i$ est la bonne classe, 0 sinon.

> 💡 Le gradient est **très simple** : (probabilité prédite − cible). Il pousse le score de la bonne classe vers le haut et **tous** les autres vers le bas, exactement ce que la MSE faisait mal.
> 🔎 « MV multinomiale » : minimiser la CE = **maximiser la vraisemblance** d'un modèle multinomial (catégoriel). En PyTorch, `nn.CrossEntropyLoss` prend les **scores bruts** (logits) et applique elle-même le log-softmax.

**Cross-entropie binaire** — pour le **multi-label** en particulier : cross-entropie sur chaque sortie (considérée comme des **Bernoulli indépendantes**) :

$$BCE(\mathbf y, \hat{\mathbf y}) = -\sum_{i=1}^{K} y_i\log(\hat y_i) + (1 - y_i)\log(1 - \hat y_i)$$

- $y_i \in \{0,1\}$ : présence du label $i$ ; $\hat y_i \in ]0,1[$ : probabilité prédite (sigmoïde) ;
- 🔎 le signe $-$ porte sur toute la somme : $-\sum_i \big[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\big]$.

### 5.4 Lutter contre le sur-apprentissage (diapo 25)

Différentes techniques qui visent toutes à **régulariser** le réseau :

| Technique | Principe |
|---|---|
| **Régularisation des couches (l1, l2)** | Ajout d'un terme de **pénalisation en $\lVert W\rVert^p$** sur les poids des couches |
| **Dropout** | Retirer **pendant une itération** quelques neurones **au hasard** dans le réseau ; permet d'augmenter la **robustesse** |
| **Augmented Data** | **Perturbation des données d'entrée** pour améliorer la généralisation |
| **Gradient Clipping** | La **norme du gradient** rétro-propagé est **bornée** maximalement pour éviter une trop grosse instabilité |

- **Dropout** (schéma) : (a) réseau standard entièrement connecté ; (b) après dropout, des neurones (croix) sont éteints avec toutes leurs connexions.
- **Data augmentation** : *une idée simple pour régulariser par la masse et les variations*. Ex. un papillon → image originale, dé-texturée, dé-colorisée, contours renforcés, carte de contours saillants, retournements/rotations.
- **Comment automatiser le processus ?** ⇒ outils **paramétrables** et disponibles dans `torchvision` (ex. politiques d'augmentation sur des chiffres « 15 » : ShearX, Invert, Solarize, AutoContrast, Equalize… chacune avec une probabilité et une magnitude).

> 🔎 Pénalisation L2, hypothèses : $L$ coût initial, $\lambda \ge 0$ :
> $$L_{reg} = L + \lambda\sum_{\text{couches}}\lVert W\rVert_2^2$$
> - $\lambda$ : force de la régularisation (favorise des poids faibles). L1 ($\lVert W\rVert_1$) pousse en plus des poids à **exactement 0**.

### 5.5 Amélioration du gradient (diapo 26)

**Gradient vanishing** : le gradient **tend à disparaître** :
- dans les couches **éloignées de la supervision** ;
- dans les **sigmoïdes saturées**.

Propriétés de la sigmoïde : domaine $(-\infty, +\infty)$, image $(0, +1)$, $\sigma(0) = 0{,}5$ ;

$$\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}, \qquad \sigma(x) = 1 - \sigma(-x), \qquad \sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big)$$

> 💡 $\sigma'(x) \le 0{,}25$ (maximum en 0) et $\to 0$ dès que $|x|$ est grand (saturation). En rétro-propagation, on **multiplie** ces dérivées couche après couche → le gradient s'**évanouit** dans les premières couches.

Fonction d'activation spécifique :

$$\mathrm{ReLU}(x) = \max(0, x)$$

→ permet de **garder un gradient fort lorsque le neurone est activé** (dérivée = 1 pour $x > 0$).

**Topologie de l'espace de recherche & gradient :**
- **BatchNorm** : pour une couche, **centrée/normée chaque sortie** (estimation sur chaque **mini-batch**) ;
- **LayerNorm** : à la sortie d'une couche, normalisation de **chaque exemple séparément** de ses dimensions.

Place dans le réseau : `Linear₁ → Activation₁ → BatchNorm₁ → Linear₂ …`

Calculs de la BatchNorm (schéma), hypothèses : mini-batch de $M$ exemples, $A_i$ = activation de la *feature* $i$ :

$$\mu_i = \frac{1}{M}\sum A_i, \qquad \sigma_i = \sqrt{\frac{1}{M}\sum (A_i - \mu_i)^2}, \qquad \hat A_i = \frac{A_i - \mu_i}{\sigma_i}, \qquad BN_i = \gamma \odot \hat A_i + \beta$$

$$\mu_{mov_i} = \alpha\,\mu_{mov_i} + (1 - \alpha)\,\mu_i, \qquad \sigma_{mov_i} = \alpha\,\sigma_{mov_i} + (1 - \alpha)\,\sigma_i$$

- $\mu_i, \sigma_i$ : moyenne et écart-type **du mini-batch** ;
- $\gamma, \beta$ : paramètres **appris** (échelle et décalage) ;
- $\mu_{mov}, \sigma_{mov}$ : **moyennes glissantes** (facteur $\alpha$), utilisées en **inférence**.

> 💡 **Centrer les données = meilleure topologie pour l'apprentissage** : les courbes de niveau du coût passent d'ellipses très allongées (le gradient zigzague) à des cercles (le gradient pointe vers le minimum).

---

## 6. Optimisation

### 6.1 Apprendre par descente de gradient (diapo 27)

**Espace de description vs espace des paramètres** : à gauche, des droites de régression dans l'espace $(X_1, Y)$ (une mauvaise $\mathbf w_0$, la bonne $\mathbf w^\star$) ; à droite, le coût $C$ en fonction de $\mathbf w$ : la pente $\frac{\partial C}{\partial w}\big|_{\mathbf w_0}$ indique dans quel sens aller vers $\mathbf w^\star$.

**Algorithme itératif de la descente de gradient** — hypothèses : $C = \sum_{i=1}^{n}(\mathbf x_i \mathbf w - y_i)^2$ :

1. Initialiser $\mathbf w_0$ ;
2. En boucle (avec mise à jour du gradient) :

$$\mathbf w^{t+1} = \mathbf w^t - \varepsilon\,\nabla_{\mathbf w} C$$

- $\varepsilon$ : **learning rate** ; $t$ : itération.

> 💡 Surface d'erreur d'un neurone linéaire = **bol convexe** (un seul minimum). Pour un réseau : surface **non convexe** avec **minima locaux**, **points selle** et un **minimum global**.

### 6.2 Stochastic, batch… ou mini-batch (diapo 28)

| Variante | Gradient calculé sur | Trajectoire |
|---|---|---|
| **Batch** gradient descent | **tout** le jeu de données | régulière, mais chaque pas est coûteux |
| **Mini-batch** gradient descent | un **sous-ensemble** (ex. 32 exemples) | compromis : un peu bruitée, rapide |
| **Stochastic** gradient descent | **un seul** exemple | très bruitée |

> ⚠️ **Le nombre d'itérations n'est pas le coût !** (une itération de batch coûte autant que $n$ itérations stochastiques).

### 6.3 Plein de variantes très efficaces pour le gradient (diapo 29)

Références : **Sebastian Ruder** (`ruder.io/optimizing-gradient-descent/`) et **Francis Bach** (`francisbach.com`). Schéma : SGD classique qui **oscille** dans une vallée étroite vs SGD avec **momentum** qui amortit les oscillations et avance vers le minimum.

> 🔎 Momentum : on accumule une « vitesse » $\mathbf v^{t+1} = \mu\,\mathbf v^t + \nabla C$, puis $\mathbf w^{t+1} = \mathbf w^t - \varepsilon\,\mathbf v^{t+1}$. **Adam** combine momentum et pas adaptatif par paramètre (le choix par défaut, `lr=1e-3`).

### 6.4 Fonction standard d'apprentissage (diapo 30)

```python
net = torch.nn.Sequential([...])
net.name = "mon_premier_reseau"
net = net.to(device)
MyLoss = torch.nn.MSELoss()
optim = torch.optim.Adam(params=net.parameters(), lr=1e-3)

summary = SummaryWriter(f"/tmp/logs/model-{time.asctime()}")
for epoch in tqdm(range(EPOCHS)):
    net.train()
    cumloss = 0
    for xbatch, ybatch in train_loader:
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)
        outputs = net(xbatch)
        loss = MyLoss(outputs.view(-1), ybatch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cumloss += loss.item()
    summary.add_scalar("loss/train loss", cumloss/len(train_loader), epoch)  # 🔎 fin de ligne coupée sur la diapo
```

### 6.5 Faire varier le learning rate (diapo 31)

```python
from torch.optim.lr_scheduler import OneCycleLR   # pour accélérer

nepoch = 1000
max_lr = 5e-1   # version optimisée OneCycle
optimizer = torch.optim.AdamW([noise], lr=max_lr)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=1, epochs=nepoch)  # 🔎 fin coupée sur la diapo
[...]
    loss = MyLoss(outputs.view(-1), ybatch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

> 🔎 **OneCycleLR** : le learning rate **monte** d'abord jusqu'à `max_lr` puis **redescend** très bas en fin d'apprentissage ; `scheduler.step()` s'appelle **après** `optimizer.step()`, une fois par batch (ici `steps_per_epoch=1`). **AdamW** = Adam avec une pénalisation des poids (*weight decay*) découplée.

---

## 7. Conclusion : hyper-paramètres et outils

### 7.1 Multiplication des modules et des hyper-paramètres (diapo 32)

- **Architecture = beaucoup d'hyper-paramètres.**
- Besoin de **normalisation** pour : mieux **comparer** les architectures ; avoir des **a priori** sur les bons paramètres.
- **Lexique des modules** : comprendre les enjeux, savoir **lire les articles**.

**Outils supplémentaires très importants** : `DataLoader`, **Check-pointing**, **TensorBoard**.

### 7.2 DataLoader (diapo 33)

| Étape | Élément | Rôle |
|---|---|---|
| **Step 1** | **Custom Dataset class** | Définit **comment les données sont chargées** ; possède une méthode `__getitem__` |
| **Step 2** | Instanciation | Training dataset, Validation dataset, Test dataset |
| **Step 3** | **DataLoader class** (classe parente importée de PyTorch) | Train / Validation / Test dataloader ; définit le **shuffling**, la **taille de batch**, le **nombre de processus**, etc. |

⇒ **Le lien entre *n'importe quelles* données et *un* modèle.**

### 7.3 TensorBoard (diapo 34)

- Visualisation des courbes (loss…) **dans une autre fenêtre** (ou sur un autre serveur) ; on peut superposer plusieurs *runs* (ex. `test/DeuxCouches` vs `test/Sequentiel`) et lisser les courbes.

> 🔎 **Checkpointing** : sauvegarder régulièrement `model.state_dict()` et `optim.state_dict()` (avec `torch.save`) pour reprendre un apprentissage long en cas d'interruption.

---

## 8. Pièges à éviter

> 🚩 **Empiler des couches linéaires sans activation** : le réseau reste **linéaire** (équivalent à un perceptron), quel que soit le nombre de couches.

> 🚩 **Utiliser la fonction signe comme activation dans un réseau entraîné par gradient** : dérivée nulle presque partout → aucun apprentissage.

> 🚩 **MSE pour la classification multi-classes** : elle s'occupe de la similarité du vecteur, alors que seul l'**argmax** compte ; préférer **SoftMax + cross-entropie**.

> 🚩 **Mettre un Softmax avant `nn.CrossEntropyLoss`** (🔎) : la loss applique déjà un log-softmax sur les scores bruts.

> ⚠️ **Sigmoïdes saturées / réseaux profonds** : *vanishing gradient* → ReLU, normalisation (BatchNorm/LayerNorm), bonne initialisation.

> ⚠️ **Dropout et BatchNorm se comportent différemment en apprentissage et en inférence** (🔎 `model.train()` / `model.eval()`).

> ⚠️ **« Le nombre d'itérations n'est pas le coût »** : comparer batch / mini-batch / stochastique à coût de calcul égal (nombre d'exemples vus).

> ⚠️ **Réseaux fully-connected profonds** : très sujets au **sur-apprentissage** et inadaptés aux entrées complexes (image, texte).

> ⚠️ **Erreur provenant de plusieurs sources** : dans la rétro-propagation, les contributions se **somment**.

> ⚠️ **Coquilles des diapos** : $g'(a_k)$ (diapo 12) doit se lire $g'(a_j)$ ; $w_{13}$ (diapo 11) doit se lire $w^{(2)}_{11}$ ; $\arg\max$ du coût du perceptron (diapo 3) doit se lire $\arg\min$.

---

## 9. Quiz auto-test

<details>
<summary><b>Question 1 (3 pts) :</b> Pourquoi combiner deux couches linéaires sans activation ne suffit-il pas ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

La composition de fonctions affines est affine : en développant $f_3(\mathbf x) = w^{(2)}_{11}f_1(\mathbf x) + w^{(2)}_{21}f_2(\mathbf x) + w^{(2)}_{01}$, on obtient $\alpha x_1 + \beta x_2 + \gamma$. Le réseau reste équivalent à un perceptron et ne peut séparer que linéairement. Il faut une **non-linéarité** $g$ (sigmoïde, tanh, ReLU) entre les couches.
</details>

<details>
<summary><b>Question 2 (3 pts) :</b> Avec L = (g(a₃) − y)² et a₃ = w₀₁⁽²⁾ + w₁₁⁽²⁾z₁ + w₂₁⁽²⁾z₂, que vaut ∂L/∂wᵢ₁⁽²⁾ ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\frac{\partial L}{\partial w^{(2)}_{i1}} = \frac{\partial L}{\partial a_3}\frac{\partial a_3}{\partial w^{(2)}_{i1}} = 2g'(a_3)(\hat y - y)\,z_i$$

avec $\hat y = g(a_3)$ et $z_0 = 1$ pour le biais.
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Écrire la règle de rétro-propagation de l'erreur δⱼ = ∂L/∂aⱼ dans une couche intermédiaire, et le gradient d'un poids wᵢⱼ. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\delta_j = \frac{\partial L}{\partial a_j} = \sum_k \frac{\partial a_k}{\partial a_j}\,\delta_k = g'(a_j)\sum_k w_{jk}\,\delta_k, \qquad \frac{\partial L}{\partial w_{ij}} = z_i\,\delta_j$$

L'erreur d'un neurone est la **somme** des erreurs des neurones qu'il alimente, pondérées par les poids des connexions, multipliée par la dérivée de son activation.
</details>

<details>
<summary><b>Question 4 (2 pts) :</b> Qu'est-ce qu'un module Mᵏ, et quels types usuels de modules existe-t-il ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Un module reçoit la sortie du module précédent $\mathbf z^{k-1}$, possède éventuellement des paramètres $W^{(k)}$ et produit une sortie $\mathbf z^k = M^k(\mathbf z^{k-1}, W^k)$. Types usuels : **couche linéaire** ($\mathbb R^d \to \mathbb R^{d'}$, paramètres $W$), **module d'activation** ($\mathbb R^d \to \mathbb R^d$, sans paramètre, gradient des paramètres nul), **coût** (bloc final à deux entrées : supervision et sortie).
</details>

<details>
<summary><b>Question 5 (4 pts) :</b> Pourquoi la MSE est-elle un mauvais coût en multi-classes ? Donner la cross-entropie et son gradient combinée au SoftMax. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

La MSE cherche à reproduire le vecteur one-hot, alors que seul l'**argmax** compte : on veut surtout maximiser la sortie de la bonne classe et minimiser les autres.

$$\mathrm{SoftMax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad CE(\mathbf y, \mathrm{SoftMax}(\mathbf z)) = -z_k + \log\sum_{j=1}^{K}e^{z_j}, \qquad \frac{\partial CE}{\partial z_i} = \mathrm{SoftMax}(\mathbf z)_i - \mathbb 1_{i=k}$$

Elle correspond au **maximum de vraisemblance multinomial**.
</details>

<details>
<summary><b>Question 6 (2 pts) :</b> Quand utiliser la BCE plutôt que la CE ? Écrire la BCE. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

En **multi-label** (plusieurs classes vraies à la fois) : chaque sortie est une Bernoulli indépendante (sigmoïde par sortie).

$$BCE(\mathbf y, \hat{\mathbf y}) = -\sum_{i=1}^{K}\big[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\big]$$
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Citer les 4 techniques de régularisation du cours et leur principe. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. **Pénalisation l1/l2** : terme en $\lVert W\rVert^p$ ajouté au coût ;
2. **Dropout** : éteindre aléatoirement des neurones à chaque itération (robustesse) ;
3. **Data augmentation** : perturber les entrées (flip, rotation, couleurs, contours…, outils `torchvision`) ;
4. **Gradient clipping** : borner la norme du gradient pour éviter l'instabilité.
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Qu'est-ce que le vanishing gradient, d'où vient-il, et quelles solutions donne le cours ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le gradient tend à disparaître dans les couches **éloignées de la supervision** et dans les **sigmoïdes saturées** : $\sigma'(x) = \sigma(x)(1-\sigma(x)) \le 0{,}25$ et tend vers 0 quand $|x|$ grandit ; ces facteurs se multiplient en remontant. Solutions : **ReLU** $= \max(0,x)$ (dérivée 1 quand le neurone est actif), **BatchNorm / LayerNorm** (meilleure topologie du coût).
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Différence entre BatchNorm et LayerNorm ? Que se passe-t-il en inférence pour la BatchNorm ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**BatchNorm** : centre/réduit chaque sortie (feature) avec les statistiques **du mini-batch**, puis applique $\gamma \odot \hat A + \beta$ (appris). **LayerNorm** : normalise **chaque exemple séparément** sur ses dimensions. En inférence, la BatchNorm utilise les **moyennes glissantes** $\mu_{mov}, \sigma_{mov}$ accumulées pendant l'apprentissage.
</details>

<details>
<summary><b>Question 10 (2 pts) :</b> Différence entre descente de gradient batch, mini-batch et stochastique ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le gradient est calculé respectivement sur **tout** le jeu, sur un **sous-ensemble**, ou sur **un seul** exemple. Plus le lot est petit, plus la trajectoire est bruitée mais plus chaque pas est rapide. Le nombre d'itérations n'est pas le coût de calcul.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Écrire les 3 lignes au cœur de toute boucle d'apprentissage PyTorch, dans l'ordre, et leur rôle. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

```python
optim.zero_grad()   # remise à zéro des gradients
loss.backward()     # calcul des gradients (rétro-propagation)
optim.step()        # mise à jour des paramètres
```
</details>

<details>
<summary><b>Question 12 (1 pt) :</b> Que faut-il définir quand on écrit son propre nn.Module ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Les **couches** (dans `__init__`, après `super().__init__()`) et l'**inférence** (`forward`). Le backward est automatique.
</details>

<details>
<summary><b>Question 13 (2 pts) :</b> Quel est le rôle du DataLoader et de la classe Dataset ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le **Dataset** définit comment charger un exemple (`__getitem__`). Le **DataLoader** construit les mini-batchs (taille, mélange, nombre de processus) pour l'entraînement, la validation et le test. C'est le lien entre n'importe quelles données et un modèle.
</details>

<details>
<summary><b>Question 14 (1 pt) :</b> Quel est l'algorithme de mise à jour du perceptron ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Tant qu'il n'y a pas convergence, pour chaque exemple $(\mathbf x^i, y^i)$ : si $y^i\langle \mathbf w, \mathbf x^i\rangle < 0$ alors $\mathbf w \leftarrow \mathbf w + \varepsilon\,y^i\mathbf x^i$. C'est une descente de gradient sur le coût $\max(0, -y f_{\mathbf w}(\mathbf x))$.
</details>
