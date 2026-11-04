# 🧠 Fiche — Introduction à l'IA explicable (XAI)

## Sommaire

1. [Motivation](#s1)
   - 1.1 [Les explications des systèmes experts](#s1-1)
   - 1.2 [Réseaux profonds : des décisions opaques](#s1-2)
   - 1.3 [Un principe de base : la conservation de l'information](#s1-3)
   - 1.4 [Quelles connaissances peut-on extraire ?](#s1-4)
   - 1.5 [Quand l'interprétabilité est-elle (in)utile ?](#s1-5)
   - 1.6 [L'effet « Clever Hans »](#s1-6)
2. [Les familles de méthodes](#s2)
3. [Interprétabilité par conception (by design)](#s3)
   - 3.1 [La régression linéaire](#s3-1)
   - 3.2 [Les arbres de décision](#s3-2)
   - 3.3 [Bilan des modèles ante hoc](#s3-3)
4. [Interprétation post-hoc](#s4)
   - 4a. [Indépendante du modèle : LIME](#s4a-1)
   - 4a. [Indépendante du modèle : valeurs de Shapley](#s4a-2)
   - 4a. [Remarques LIME vs Shapley](#s4a-3)
   - 4b. [Dépendante du modèle : méthodes par gradient en vision](#s4b-1)
   - 4b. [Dépendante du modèle : Concept Bottleneck Models](#s4b-2)
   - 4b. [Évaluer un système explicatif](#s4b-3)
5. [Quel avenir pour l'IA explicable ?](#s5)
6. [Quiz d'auto-évaluation](#s6)
7. [Pièges à éviter](#s7)

---

<a id="s1"></a>
## 1. Motivation

<a id="s1-1"></a>
### 1.1 Les explications des systèmes experts (SE)

Un **système expert** (ex. **MYCIN**, diagnostic d'infections et prescription d'antibiotiques) raisonne avec des **règles explicites**. On peut donc l'interroger sur son raisonnement :

- **« Pourquoi ? »** : pourquoi le système pose-t-il telle question ? Exemple avec TEIRESIAS, le module d'explication de MYCIN. Le système demande si l'infection a été contractée à l'hôpital. Il justifie la question ainsi : on sait déjà que l'organisme est Gram négatif, en bâtonnet et qu'il se développe avec ou sans air. Si l'infection n'est pas nosocomiale, l'organisme est presque certainement (0,8) une entérobactérie, avec une petite possibilité (0,1, règle 37) de *Pseudomonas aeruginosa*.
- **« Pourquoi n'avez-vous pas envisagé X ? »** : la règle 33 aurait identifié un streptocoque, mais l'une de ses clauses (« morphologie de type coque ») était déjà connue comme fausse. La règle n'a donc jamais été essayée.

> 💡 Un SE explique **son cheminement** (quelles règles ont été déclenchées), mais pas forcément la **raison profonde** derrière ces règles.

**Exemple de connaissances justificatives** (pourquoi ne pas prescrire de tétracycline à un enfant de moins de 8 ans ?). La chaîne causale est la suivante : la drogue se dépose sur les os en développement, ce qui provoque un noircissement définitif des dents, donc une coloration socialement indésirable. On en tire la règle « ne pas administrer ». Cette justification mobilise deux types de connaissances : la **notion d'effets secondaires indésirables** et des **relations de causalité**.

**Raisonnement graphique.** Prenons la formule $1+2+\dots+n = \frac{n^2}{2}+\frac{n}{2}$. Un dessin en escalier (un carré $n\times n$ coupé en deux, plus $n$ demi-cases) **convainc** bien mieux qu'une suite de règles formelles. Une bonne explication n'est donc pas toujours une trace de règles.

#### Types d'explications

| Type | Nature | Utile pour |
|---|---|---|
| **De « surface »** | Liées au **fonctionnement du système** | Vérifier ce fonctionnement → **expert IA** |
| **« Profondes »** | **Difficiles à découvrir et à expliciter** | **Expert du domaine**, **utilisateurs** |

> 📝 **Tes notes :** un SE est « bon pour décider mais pas pour expliquer ». Il répond « parce que ça enfreint telle règle du système », pas la raison réelle.
> ✅ C'est juste, et ça correspond exactement à la distinction **surface / profonde** de la diapo. Petite nuance : un SE peut produire des explications profondes **si** on lui fournit les connaissances justificatives (exemple de la tétracycline). Par défaut, il ne donne que des explications de surface.

<a id="s1-2"></a>
### 1.2 Réseaux de neurones profonds : des décisions opaques

- **Réseaux profonds** : grand nombre de couches (parfois plusieurs centaines) et très grand nombre de paramètres ($10^7$ à $10^9$). Exemple d'architecture : **GoogleNet**.
- **« Illusions d'optique »** (exemples adverses) : une image classée *Boxer : 0,40 / Tiger Cat : 0,18*, une fois légèrement perturbée, devient *Airliner : 0,9999*. **Quelle explication ?**
- **Exemple en médecine** (attaques adverses sur le ML médical) :
  - un bruit imperceptible (+0,04 × bruit) fait passer un grain de beauté de *bénin* à *malin* ;
  - une simple **rotation** de l'image produit le même effet ;
  - une **substitution de texte** dans le dossier (« back pain » → « lumbago », « alcohol abuse » → « alcohol dependence ») fait passer le risque d'abus d'opioïdes de *High* à *Low* ;
  - un changement de **codage** des diagnostics fait passer un remboursement de *refusé* à *accepté*.
- **Le cas AlphaGo** : un joueur « extraterrestre », au jeu stupéfiant, qui a révolutionné la manière de jouer et créé l'effervescence dans les écoles de go. Comprendre comment il joue relèverait de la **découverte scientifique**.

> 📝 **Tes notes :** l'explicabilité est très pratique pour le **débogage**. Quand un système de reconnaissance d'image change totalement de réponse à cause d'un bruit ajouté, elle aide à comprendre comment modifier le modèle pour le rendre moins sensible. Elle sert aussi à comprendre comment un modèle **innove** (AlphaGo, disposition « alien » de puces conçue par une IA).
> ✅ Ces deux usages correspondent aux motivations « debugging » et « curiosité scientifique » (§1.5).

<a id="s1-3"></a>
### 1.3 Un principe de base : la conservation de l'information

- L'apprentissage automatique **ne fait que reformuler** ce qu'on lui a donné en entrée.
- **« Théorème de conservation »** : **aucune information n'est « ajoutée »**. Le résultat vient des **données + connaissances a priori**.
- Ces deux sources se compensent :
  - **peu de données + beaucoup de connaissances a priori** ;
  - **beaucoup de données (big data) + moins de connaissances a priori**.

Schéma du cours : **Knowledge → (Data, Algorithm) → (How it does it, The result / Prediction rule) → Knowledge**. On peut donc questionner deux choses : **(1)** la connaissance injectée en entrée, et **(2)** la connaissance qu'on peut extraire en sortie.

#### Biais sur les données : de la connaissance dans les données

- le dispositif expérimental ;
- le **choix des descripteurs** (features) ;
- l'enrichissement par des **ontologies** ;
- la **normalisation** des valeurs ;
- la gestion des **valeurs manquantes** ;
- d'éventuels points de données ajoutés, **en pensant à des invariances** (augmentation de données).

#### Biais sur les algorithmes : de la connaissance dans l'algorithme

- **Contraintes sur l'espace d'hypothèses = biais de représentation**. Exemple : chercher des **hypothèses linéaires parcimonieuses** (*sparse*) favorise les hypothèses avec peu de coefficients non nuls.
- **CNN** : la connaissance (localité, invariance par translation) est **incorporée dans l'architecture** du réseau.

> 🔎 **Complément :** la parcimonie s'obtient typiquement avec une pénalité $L_1$ (Lasso), qui pousse de nombreux coefficients exactement à 0.

<a id="s1-4"></a>
### 1.4 Quelles connaissances peut-on extraire ?

**1. L'hypothèse retournée : une fonction de décision.** Exemples : recommander un film ou un partenaire de vie, reconnaître des caractères manuscrits ou des panneaux routiers, évaluer une position au go, prédire un risque de crime, accorder un prêt, embaucher ou non quelqu'un.

**2. Juste une fonction de décision, ou une brique d'un système d'inférence plus large ?** Dans les affaires juridiques, par exemple, on demande **d'où vient la décision**.

**3. Interprétabilité de la fonction de décision**, du plus au moins lisible :

| Modèle | Interprétabilité |
|---|---|
| Arbres de décision | Semblent **directement interprétables** |
| Fonctions linéaires | **Moins** |
| Forêts aléatoires | **Encore beaucoup moins** |
| SVM, réseaux de neurones | Exigent une **analyse difficile** |

**4. Interprétabilité du processus d'apprentissage** qui a mené à la fonction de décision :
- **analyse de sensibilité** : si on change cette valeur d'entrée, que se passe-t-il ?
- **Explanation-Based Learning** (EBL).

> 🔎 **Complément :** l'**EBL** est une forme d'apprentissage symbolique. Le système généralise à partir d'un seul exemple en **expliquant** pourquoi cet exemple est une instance du concept, à l'aide d'une théorie du domaine. La règle apprise est donc justifiée par construction.

<a id="s1-5"></a>
### 1.5 Quand l'interprétabilité est-elle (in)utile ?

| ❌ **Pas nécessaire** | ✅ **Nécessaire** |
|---|---|
| **Faible risque** associé à la décision (ex. recommandation de film) | **1. Décisions à haut risque** : opération chirurgicale, arrêt d'une centrale nucléaire, véhicule autonome |
| **Bonnes garanties de performance** (ex. reconnaissance de caractères) | **2. Satisfaire la curiosité** (c'est le but de la science) : expliquer des résultats surprenants, cas sans explication simple, ou quand la fonction de décision doit s'intégrer dans un système d'inférence plus large (une **théorie du domaine**) |
| | **3. Débogage** : pourquoi cette décision est fausse (**contrefactuel**) ? Si un vélo est reconnu grâce à ses deux roues, que se passe-t-il si l'une est cachée par des sacoches ? Pourquoi le système semble-t-il biaisé selon le genre ? |
| | **4. L'interprétabilité exige des systèmes de meilleure qualité** : un système interprétable **peut être manipulé** (si on sait qu'un prêt est accordé à partir de 2 cartes de crédit, on en prend 3). Pour ne pas être manipulé, le système doit utiliser des **facteurs causaux** |

> 💡 **Objectif du cours** : des algorithmes **« transparents »**, (1) dans lesquels on peut **injecter l'expertise humaine**, et (2) dont les **modèles appris sont interprétables**.

<a id="s1-6"></a>
### 1.6 L'effet « Clever Hans » (Hans le Malin)

- **L'histoire** : Hans, un cheval, semblait savoir compter et répondre à des questions en tapant du sabot. En réalité, il **interprétait les signaux corporels inconscients** de son maître ou des questionneurs. Il donnait **la bonne réponse pour la mauvaise raison**.
- **En apprentissage supervisé** : sur les 9963 images de chevaux de Pascal VOC, une carte d'explication montre que le classifieur regarde le **copyright en bas à gauche** (filigrane d'un site de photos de chevaux), pas le cheval.
  - Ce sont des **corrélations accidentelles (*spurious*) mais très prédictives**…
  - … **sauf quand** on ajoute ce filigrane à une **image artificielle de voiture**, qui est alors **reconnue comme un cheval**.

> 📝 **Tes notes :** Clever Hans = le modèle donne la bonne réponse pour la mauvaise raison. « Si on sait comment le système prend sa décision, on peut le tromper (une sorte de data leak…). Le système peut être manipulé, donc on vérifie la raison de sa décision pour éviter ça. »
> ⚠️ **Correction / précision :** tu mélanges deux idées distinctes du cours.
> - **Effet Clever Hans** : le modèle exploite une **corrélation fallacieuse** présente dans les données (le filigrane). On parle aussi d'*apprentissage par raccourci*. Le lien avec la **fuite de données** (*data leakage*) est pertinent : une information qui ne devrait pas aider à prédire se retrouve dans les données. Mais ce n'est **pas** une question de « tromper » le système. On utilise l'explication pour **détecter** le raccourci (débogage, motivation n°3).
> - **Manipulation** (motivation n°4, diapo 41) : c'est un **utilisateur** qui, connaissant la règle (« plus de 2 cartes de crédit »), modifie ses entrées pour obtenir la décision voulue. La parade est d'utiliser des **facteurs causaux**.
>
> Les deux se rejoignent sur un point : un modèle fondé sur des facteurs non causaux est fragile.

---

<a id="s2"></a>
## 2. Les familles de méthodes

```
                 ML interpretability
                /                   \
          By-design               Post-hoc
                                 /        \
                    Model-agnostic      Model-specific
                      /       \
                  Local      Global
```

| Famille | Principe | Exemples |
|---|---|---|
| **Par conception (by-design / ante hoc)** | Fournissent l'**importance relative des descripteurs** **par nature** de la méthode | Régression linéaire, arbre de décision |
| **Post-hoc** | **Découplent** l'apprentissage d'un modèle de son explication | — |
| ↳ **Indépendantes du modèle (model-agnostic)**, **locales** | « Explication » de la **prédiction associée à un exemple** | LIME, valeurs de Shapley |
| ↳ **Indépendantes du modèle**, **globales** | « Explication » de la **stratégie de décision globale** du modèle | — |
| ↳ **Dépendantes du modèle (model-specific)** | Exploitent la structure interne du modèle | **Cartes de saillance** (réseaux de neurones), **identification des concepts** mis en jeu |

> 📝 **Tes notes :** « Local : agnostic sur 1 exemple ; Global : agnostic sur la stratégie globale (pas dépendant d'un exemple) ».
> ✅ Correct, et fidèle à la diapo 45.

> 🔎 **Complément :** exemples de méthodes agnostiques **globales**, non détaillées dans le cours : *permutation feature importance*, *Partial Dependence Plots* (PDP), et la moyenne des $|\phi_j|$ de Shapley sur tout le jeu de données.

---

<a id="s3"></a>
## 3. Interprétabilité par conception (by design)

<a id="s3-1"></a>
### 3.1 La régression linéaire

On suppose $d$ descripteurs $\mathbf{x}^{(1)},\dots,\mathbf{x}^{(d)}$ et $m$ exemples d'apprentissage $(\mathbf{x}_i, y_i)_{1\le i\le m}$. Le modèle prédit la sortie par une **somme pondérée des descripteurs** :

$$
y \;=\; \beta_0 + \beta_1\,\mathbf{x}^{(1)} + \dots + \beta_d\,\mathbf{x}^{(d)}
$$

Les coefficients sont typiquement trouvés par **minimisation de l'écart quadratique** (moindres carrés) :

$$
\widehat{\beta} \;=\; \underset{\beta_0,\dots,\beta_d}{\mathrm{Argmin}} \sum_{i=1}^{m}\Big(y_i - \big(\beta_0 + \sum_{j=1}^{d}\beta_j\,\mathbf{x}_i^{(j)}\big)\Big)^2
$$

- $y$, $y_i$ : sortie prédite, sortie observée de l'exemple $i$
- $\beta_0$ : ordonnée à l'origine (*intercept*)
- $\beta_j$ : coefficient (poids) du descripteur $j$
- $\mathbf{x}_i^{(j)}$ : valeur du descripteur $j$ pour l'exemple $i$
- $d$ : nombre de descripteurs ; $m$ : nombre d'exemples

**Exemple du cours** (score de crédit) : `y = 674.95 + 0.00009*INCOME − 9.26*R_DEBT_INCOME + 253.13*R_EXPENDITURE − 414.78*R_EXPENDITURE_SQR − 98.45*R_ENTERTAINMENT − 7.09*GAMBLING_LOW − 24.82*GAMBLING_HIGH`

#### Hypothèses pour qu'un modèle linéaire soit correct

1. La **fonction cible est linéaire** (ou proche).
2. La **sortie suit une loi normale étant donnés les descripteurs**. Sinon, les **estimateurs de confiance** (intervalles, p-valeurs) sont **invalides**.
3. **Homoscédasticité** : la variance est constante pour toutes les valeurs de chaque descripteur. Par exemple, quelle que soit la taille de la maison, la variance autour du prix prédit est la même.
4. **Non-corrélation des descripteurs.**

> 📝 **Tes notes :** « Sortie $\sim N$ ».
> ⚠️ **Précision :** c'est la sortie **conditionnellement aux descripteurs** qui est normale : $y\,|\,\mathbf{x} \sim \mathcal{N}(\beta_0+\sum_j\beta_j x^{(j)},\,\sigma^2)$. Autrement dit, ce sont les **résidus** qui sont normaux. La distribution globale de $y$ n'a pas besoin d'être gaussienne.

#### Ce que nous disent les coefficients $\beta_j$

$\beta_j$ indique **de combien varie la sortie quand on augmente le descripteur $\mathbf{x}^{(j)}$ d'une unité**, toutes choses égales par ailleurs. Par exemple, de combien varie le prix d'une maison **s'il y a une pièce de plus**.

**Lecture d'une sortie de régression** (type `statsmodels`) :

| Colonne | Signification |
|---|---|
| `coef` | $\widehat{\beta}_j$ |
| `std err` | Erreur standard $\mathrm{SE}(\widehat{\beta}_j)$ |
| `t` | **t-statistique** = rapport du coefficient à son erreur standard. **Plus elle est grande, plus le coefficient est important** |
| `P>\|t\|` | **p-valeur** du test d'hypothèse ($H_0$ : $\beta_j=0$). Si elle est petite (typiquement < 0,05), le descripteur a un **impact significatif** sur la sortie |
| `[0.025  0.975]` | **Intervalle de confiance** à 95 % du coefficient |

$$
t_{\widehat{\beta}_j} \;=\; \frac{\widehat{\beta}_j}{\mathrm{SE}(\widehat{\beta}_j)}
$$

- $\widehat{\beta}_j$ : coefficient estimé
- $\mathrm{SE}(\widehat{\beta}_j)$ : erreur standard (incertitude d'estimation) du coefficient

> 📝 **Tes notes :** « l'attribut le plus important pour la cible est celui qui a le plus gros $\beta$ » (exemple : surface vs localisation pour le prix d'une maison).
> 🚩 **Faux tel quel.** $\beta_j$ **dépend de l'unité** du descripteur. Dans le tableau, `INCOME` a $\beta = 9\times10^{-5}$, un coefficient minuscule, alors que c'est un descripteur très significatif ($t = 11{,}08$) : le revenu se compte en dizaines de milliers. C'est pour ça que la diapo utilise la **t-statistique** (sans unité) pour juger l'importance. On peut aussi **standardiser** les descripteurs (centrer-réduire) avant de comparer les $|\beta_j|$.

#### Matrice de corrélation entre descripteurs

Elle révèle comment les descripteurs sont **linéairement corrélés entre eux** : +1 = corrélation positive maximale, −1 = corrélation négative maximale, 0 = pas de relation linéaire. Une **corrélation élevée** (positive ou négative) signifie que les **descripteurs sont redondants**.

#### Graphique des effets individuels (*individual effect plot*)

Il mesure la **contribution de chaque descripteur à la prédiction pour un exemple donné** (un patient, une maison) :

$$
\text{effet}_j(\mathbf{x}) \;=\; \widehat{\beta}_j \, x^{(j)}
\qquad\text{et}\qquad
\hat{y}(\mathbf{x}) = \widehat{\beta}_0 + \sum_{j=1}^{d}\text{effet}_j(\mathbf{x})
$$

- $\text{effet}_j(\mathbf{x})$ : part de la prédiction due au descripteur $j$ pour l'exemple $\mathbf{x}$
- $x^{(j)}$ : valeur du descripteur $j$ pour cet exemple

Dans l'exemple du cours (code Python avec `LinearRegression`, prix fictif d'une maison selon la surface, le nombre de chambres et la distance au centre), on trace en barres horizontales l'effet de chaque variable. La surface a un gros effet positif, l'intercept constitue la base, et la distance au centre a un effet négatif. La somme donne la prédiction totale (≈ 211 854 €).

> 💡 C'est déjà une **explication locale** : la même idée sera généralisée par les valeurs de Shapley.

#### Avantages et inconvénients des modèles linéaires

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| **Simples à mettre en œuvre** | **Supposent les descripteurs décorrélés**. Si deux descripteurs sont corrélés et que l'un a un très fort coefficient positif, **l'autre peut avoir un coefficient négatif** alors qu'il est corrélé positivement à la cible |
| Permettent d'obtenir des **intervalles de confiance** | **Rendent mal compte des dépendances non linéaires** entre descripteurs et cible |

<a id="s3-2"></a>
### 3.2 Les arbres de décision

Ils sont **lisibles… jusqu'à un certain point**. Un arbre très profond devient illisible.

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| **Capturent naturellement les interactions** entre descripteurs | **Mal adaptés aux dépendances linéaires** (cf. **arbres obliques**) |
| Montrent comment on obtient **différents groupes** dans les données | **Instables** : peuvent changer beaucoup pour de petits changements de la base d'apprentissage |
| La **structure en arbre** se prête à l'interprétation | Peuvent être **opaques si les descripteurs sont mal choisis** (ex. explication des fins de parties aux échecs) |

> 📝 **Tes notes :** « Ne sont pas bien adaptés à capturer les dépendances **non** linéaires ».
> 🚩 **Inversé !** La diapo dit **dépendances linéaires**. Un arbre découpe l'espace avec des seuils parallèles aux axes ($x_j \le s$). Pour approcher une relation linéaire comme $y = 2x$ ou une frontière oblique $x_1 + x_2 = 1$, il lui faut un **escalier** de nombreuses coupes. D'où les **arbres obliques**, qui coupent selon des combinaisons linéaires de descripteurs. À l'inverse, les arbres gèrent très bien le non-linéaire et les interactions.

<a id="s3-3"></a>
### 3.3 Bilan des modèles par conception (ante hoc)

| ✅ | ❌ |
|---|---|
| **Plus facilement interprétables** | En général **enclins au sous-apprentissage**, car trop simples pour capturer les interdépendances entre descripteurs et avec la cible |
| Interprétation **du modèle**, et pas seulement d'une prédiction donnée | |

> 🔎 **Complément :** la référence donnée en ressources, **Rudin, « Stop explaining black box models for high stakes decisions »**, défend la position inverse de la post-hoc. Pour les décisions à fort enjeu, il faut **utiliser directement des modèles interprétables** plutôt qu'expliquer des boîtes noires après coup.

---

<a id="s4"></a>
## 4. Interprétation post-hoc

### 4a. Indépendante du modèle (model-agnostic) : expliquer une prédiction individuelle

<a id="s4a-1"></a>
#### LIME (*Local Interpretable Model-agnostic Explanations*)

**Idée** : identifier un **modèle interprétable localement fidèle** au modèle appris.

**Figure du cours** : la fonction de décision $f$ est **complexe et inconnue** (frontière entre la zone bleue et la zone rose). La **grande croix rouge** est l'exemple à expliquer. Les points bleus et croix rouges autour sont des exemples proches, dont la **taille dépend de la proximité**. La **droite en pointillés** est la fonction $g$ apprise par LIME, qui approxime **localement** $f$.

**Algorithme :**
1. **Sélectionner l'exemple** à expliquer.
2. **Générer des exemples virtuels** en **permutant des valeurs d'attributs** (perturbations autour de l'exemple).
3. **Assigner un poids** à chaque exemple virtuel **en fonction de sa distance à l'exemple**.
4. **Prédire la classe** de ces exemples virtuels **grâce au modèle appris** $f$.
5. **Entraîner un modèle surrogé simple** (ex. fonction linéaire) sur les exemples virtuels étiquetés et pondérés.
6. **Interpréter le modèle surrogé**.

> 📝 **Tes notes** (formule vue à l'oral, absente du texte des diapos) :
>
> Pour un exemple $x$ à expliquer et une famille $G$ de modèles interprétables :
>
> $$\xi(x) \;=\; \underset{g\in G}{\arg\min}\; \mathcal{L}(f, g, \pi_x) + \Omega(g)$$
>
> - $\xi(x)$ : l'explication retenue pour $x$ (le modèle surrogé)
> - $f$ : le modèle boîte noire à expliquer
> - $g$ : modèle interprétable candidat (ex. linéaire), pris dans $G$
> - $\pi_x$ : **noyau de proximité** qui pondère les exemples virtuels selon leur distance à $x$ (étape 3)
> - $\mathcal{L}(f,g,\pi_x)$ : **infidélité locale**, c'est-à-dire l'écart entre $f$ et $g$ pondéré par $\pi_x$ (étapes 4-5)
> - $\Omega(g)$ : **pénalité de complexité** de $g$ (ex. nombre de coefficients non nuls), qui garantit que $g$ reste interprétable
>
> ✅ Formule juste (c'est celle de l'article de Ribeiro et al.). Il manquait juste « $g \in G$ » sous l'argmin : on minimise sur les modèles interprétables. Ton résumé « on approxime localement par un modèle linéaire un problème globalement complexe » est exactement l'idée.

**Remarques du cours :**
- Le **choix de la fonction de distance** (et donc de la notion de voisinage) est **crucial**.
- Ribeiro et al. utilisent un **noyau exponentiel** de largeur (*width*) $= 0{,}75 \times \sqrt{d}$, avec $d$ le nombre de descripteurs.
- Implémentations : Python (librairies `lime`, `PiML`), R (packages `lime`, `iml`).

> 🔎 **Complément :** le noyau exponentiel s'écrit $\pi_x(z) = \exp\!\big(-D(x,z)^2/\sigma^2\big)$, avec $D$ une distance (euclidienne, cosinus…) et $\sigma$ la largeur ci-dessus. Plus $\sigma$ est petit, plus l'explication est locale, mais aussi plus elle est instable.

<a id="s4a-2"></a>
#### La méthode des valeurs de Shapley

**Inspiration : la théorie des jeux coopératifs.**
- Chaque **descripteur** est un **« joueur »** dans un jeu de prédiction.
- La **prédiction** est le **« gain »**.
- Les joueurs **coopèrent dans une coalition** pour obtenir le gain. La méthode vise à **rétribuer équitablement chaque joueur**.
- Elle s'applique aux modèles **linéaires ou non linéaires**.

##### Exemple 1 : trois élèves dans un challenge Kaggle

E1 apporte l'expertise IODAA, E2 une machine très puissante, E3 est un geek. Ensemble, ils gagnent le **1er prix = 10 000 $** (2e : 7 500 $, 3e : 5 000 $). Comment partager équitablement ?

Gains de chaque coalition, si on rejouait :

| Coalition | ∅ | E1 | E2 | E3 | E1E2 | E1E3 | E2E3 | E1E2E3 |
|---|---|---|---|---|---|---|---|---|
| Gain ($) | 0 | 7 500 | 5 000 | 2 500 | 10 000 | 7 500 | 5 000 | 10 000 |

On considère **tous les ordres d'arrivée** ($3! = 6$). Pour chaque ordre, la **contribution marginale** d'un joueur est le gain de la coalition après son arrivée moins le gain avant son arrivée :

| Ordre | Apport du 1er | Apport du 2e | Apport du 3e |
|---|---|---|---|
| E1 → E2 → E3 | 7 500 | 2 500 | 0 |
| E1 → E3 → E2 | 7 500 | 0 | 2 500 |
| E2 → E1 → E3 | 5 000 | 5 000 | 0 |
| E2 → E3 → E1 | 5 000 | 0 | 5 000 |
| E3 → E1 → E2 | 2 500 | 5 000 | 2 500 |
| E3 → E2 → E1 | 2 500 | 2 500 | 5 000 |

On moyenne ensuite les apports de chaque élève sur les 6 ordres :
- $\mathrm{Val}(E_1) = (7500+7500+5000+5000+5000+5000)/6 \approx \mathbf{5833}$
- $\mathrm{Val}(E_2) = (2500+2500+5000+5000+2500+2500)/6 \approx \mathbf{3333}$
- $\mathrm{Val}(E_3) = (0+0+0+0+2500+2500)/6 \approx \mathbf{833}$

> ⚠️ La diapo 70 affiche d'abord Val(E1) = 7500 : c'est une **coquille**, corrigée diapo 71 (35 000 / 6 ≈ 5833).

> 💡 **Vérification** : $5833 + 3333 + 833 \approx 10\,000$. La somme des valeurs de Shapley redonne le gain total. C'est la propriété d'**efficacité**.

> 📝 **Tes notes :** « on regarde le résultat cumulé avec l'ajout successif de chaque attribut, puis on moyenne l'apport de chaque attribut ».
> ✅ C'est exactement le calcul. Précision importante : la moyenne se fait sur **tous les ordres possibles** d'ajout. Un seul ordre ne suffit pas, car l'apport d'un joueur dépend de qui est déjà là. E3 apporte 0 s'il arrive après E1, mais 2 500 s'il arrive en premier.

##### Exemple 2 : prédiction de salaire

Un modèle prédit le salaire avec deux variables :
- $x_1$ : l'âge, dans $\{18,\dots,60\}$, supposé de distribution uniforme ;
- $x_2$ : le diplôme, dans $\{0,1\}$, avec probabilités égales.

Le modèle appris est $h(x_1,x_2) = 200\,x_1 + 1000\,x_2$. Pour une personne d'âge 20 avec diplôme, la prédiction vaut $200\times20 + 1000\times1 = 5000$ €. **Quelle est la valeur de Shapley de $x_2$ (diplôme = 1) ?**

Il faut d'abord répondre à une autre question : **quelle serait la prédiction si on ne connaissait pas $x_2$ ?** On **intègre sur sa distribution**. C'est la contribution marginale de $x_2$ à la coalition {âge} :

$$
\mathrm{Val}_{\mathbf{x}}(x_1) \;=\; \int h(20, x_2)\,\mathrm{dP}_{x_2} \;=\; \sum_{i=0}^{1} h(20,i)\,\mathrm{P}(x_2=i) \;=\; \tfrac12\cdot4000 + \tfrac12\cdot5000 \;=\; 4500
$$

$$
\mathrm{Val}_{\mathbf{x}}(x_2) \;=\; \mathrm{Val}_{\mathbf{x}}(x_1,x_2) - \mathrm{Val}_{\mathbf{x}}(x_1) \;=\; 5000 - 4500 \;=\; \mathbf{500}
$$

- $\mathrm{Val}_{\mathbf{x}}(S)$ : prédiction pour l'exemple $\mathbf{x}$ quand seuls les attributs de $S$ sont connus
- $\mathrm{P}_{x_2}$ : distribution de $x_2$ (ici 1/2 – 1/2)

> 🔎 **Complément (calcul complet) :** la diapo ne calcule que la contribution de $x_2$ à la coalition {âge}. La vraie valeur de Shapley moyenne aussi sur l'ordre « $x_2$ en premier », c'est-à-dire sa contribution à la coalition vide. L'âge moyen vaut $\mathbb{E}[x_1] = 39$, donc $\mathrm{Val}(\varnothing) = 200\times39 + 500 = 8300$ et $\mathrm{Val}(\{x_2\}) = 200\times39 + 1000 = 8800$. La contribution est encore $8800 - 8300 = 500$, d'où $\phi_2 = \tfrac12(500+500) = 500$.
> Pour un **modèle linéaire à descripteurs indépendants**, on a toujours $\phi_j = \beta_j\,(x_j - \mathbb{E}[x_j])$. Ici $\phi_2 = 1000\,(1-0{,}5) = 500$ et $\phi_1 = 200\,(20-39) = -3800$. Vérification : $500 - 3800 = -3300 = 5000 - 8300$ (prédiction − prédiction moyenne).

##### Exemple 3 : prix d'appartements

Un modèle prédit le prix selon la surface, l'étage, la proximité d'un parc et l'interdiction ou non des chats. Prenons l'exemple *area = 50, floor = 2, park = nearby, cat = banned*. La prédiction vaut **300 k€** alors que le **prix moyen est de 310 k€**. **L'objectif est d'expliquer la différence, soit −10 k€.**

Une explication possible : area = 50 → **+30 k€**, floor = 2 → **+10 k€**, park = nearby → **0 €**, cat = banned → **−50 k€**. Le total fait bien **−10 k€**.

> 💡 Les valeurs de Shapley expliquent l'**écart entre la prédiction pour cet exemple et la prédiction moyenne**, pas la prédiction brute.

##### Formule générale

Pour un exemple $\mathbf{x}$ et un attribut $j$ parmi $d$ attributs, on somme sur **tous les sous-ensembles** $S \subseteq \{1,\dots,d\}\setminus\{j\}$ ne contenant pas $j$ :

$$
\phi_j(val) \;=\; \sum_{S\subseteq\{1,\dots,d\}\setminus\{j\}} \frac{|S|!\,(d-|S|-1)!}{d!}\,\Big(val(S\cup\{j\}) - val(S)\Big)
$$

avec

$$
val_{\mathbf{x}}(S) \;=\; \int \hat{h}(x_1,\dots,x_d)\,\mathrm{dP}_{X_C} \;-\; \mathbb{E}\big[\hat{h}(\mathbf{X})\big]
$$

- $\phi_j$ : valeur de Shapley (contribution) de la valeur de l'attribut $j$ pour l'exemple $\mathbf{x}$
- $S$ : sous-ensemble (**coalition**) des attributs utilisés dans le modèle
- $d$ : nombre total d'attributs
- $\mathbf{x}$ : exemple dont on cherche les contributions des valeurs d'attributs
- $X_C$ : tous les attributs **non inclus** dans la coalition $S$. On **intègre** sur leur distribution, tandis que les attributs de $S$ sont fixés à leur valeur dans $\mathbf{x}$
- $\hat{h}$ : le modèle appris ; $\mathbb{E}[\hat{h}(\mathbf{X})]$ : prédiction moyenne
- $val(S\cup\{j\}) - val(S)$ : **contribution marginale** de $j$ à la coalition $S$
- $\frac{|S|!\,(d-|S|-1)!}{d!}$ : proportion des $d!$ ordres d'arrivée où exactement les joueurs de $S$ précèdent $j$. C'est le poids qui reproduit la **moyenne sur tous les ordres** de l'exemple Kaggle

**Exemple** avec $d = 4$ et $S = \{1,3\}$ : on fixe $x_1$ et $x_3$ et on intègre sur $X_2$ et $X_4$.

$$
val_{\mathbf{x}}(\{1,3\}) \;=\; \int \hat{h}(x_1, X_2, x_3, X_4)\,\mathrm{dP}_{X_2X_4} \;-\; \mathbb{E}\big[\hat{h}(\mathbf{X})\big]
$$

> 🔎 **Complément :** dans l'exemple du salaire, la diapo omet le terme $-\mathbb{E}[\hat h(\mathbf{X})]$. Ça ne change rien au résultat : ce terme constant s'annule dans la différence $val(S\cup\{j\}) - val(S)$. Il sert seulement à ce que $val(\varnothing) = 0$.

> 🔎 **Complément : les 4 propriétés qui rendent Shapley « équitable »**
> - **Efficacité** : $\sum_j \phi_j = \hat h(\mathbf{x}) - \mathbb{E}[\hat h(\mathbf{X})]$.
> - **Symétrie** : deux attributs qui contribuent pareil à toutes les coalitions ont la même valeur.
> - **Joueur nul** : un attribut qui n'apporte jamais rien a $\phi_j = 0$.
> - **Additivité** : pour un modèle somme de deux modèles, les $\phi$ s'additionnent (utile pour les forêts).

##### Deux questions pratiques

**1. Que faire quand on ne connaît pas les distributions des attributs ?** Pour estimer $\mathrm{P}_{X_2X_4}$, on **tire aléatoirement** des valeurs de $X_2$ et $X_4$ **présentes dans l'ensemble d'apprentissage** (idéalement toutes les valeurs).

**2. Comment faire face à la croissance exponentielle du nombre de coalitions** ($2^{d-1}$ sous-ensembles) avec $d$ ? On utilise une **estimation approchée par Monte-Carlo** (Molnar). Entrées : nombre d'itérations $M$, exemple d'intérêt $\mathbf{x}$, indice $j$, matrice de données $\mathbf{X}$, modèle $f$.

Pour $m = 1,\dots,M$ :
1. tirer une instance aléatoire $\mathbf{z}$ dans $\mathbf{X}$ ;
2. tirer une permutation aléatoire $\mathbf{o}$ des attributs ;
3. ordonner $\mathbf{x}$ et $\mathbf{z}$ selon $\mathbf{o}$ ;
4. construire deux instances hybrides :
   - **avec $j$** : $\mathbf{x}_{+j} = (x_{(1)},\dots,x_{(j-1)}, x_{(j)}, z_{(j+1)},\dots,z_{(p)})$
   - **sans $j$** : $\mathbf{x}_{-j} = (x_{(1)},\dots,x_{(j-1)}, z_{(j)}, z_{(j+1)},\dots,z_{(p)})$
5. calculer la contribution marginale $\phi_j^{(m)} = \hat f(\mathbf{x}_{+j}) - \hat f(\mathbf{x}_{-j})$.

Puis :

$$
\phi_j(\mathbf{x}) \;=\; \frac{1}{M}\sum_{m=1}^{M}\phi_j^{(m)}
$$

- $M$ : nombre d'itérations de Monte-Carlo
- $\mathbf{z}$ : instance aléatoire, qui fournit les valeurs « inconnues » (attributs hors coalition)
- $\mathbf{o}$ : permutation aléatoire, qui simule un ordre d'arrivée des joueurs
- $p$ : nombre d'attributs (noté $d$ ailleurs dans le cours)

> 💡 Les attributs placés **avant** $j$ dans la permutation forment la coalition : ils gardent la valeur de $\mathbf{x}$. Ceux placés **après** prennent la valeur de $\mathbf{z}$, ce qui revient à les intégrer. La seule différence entre les deux instances est l'attribut $j$ lui-même.

<a id="s4a-3"></a>
#### Remarques : LIME vs Shapley

| | **LIME** | **Shapley** |
|---|---|---|
| Fournit | L'**importance relative** de chaque attribut (poids du surrogé local) | La contribution **« exacte »** de **chaque valeur mesurée** de chaque attribut |
| Fondement | Approximation linéaire locale | Théorie des jeux (propriétés d'équité) |
| Point sensible | Choix du noyau / de la largeur du voisinage | Coût exponentiel (→ échantillonnage), estimation des distributions |

> ⚠️ **ATTENTION** : SHAP ne mesure pas l'importance d'un descripteur **dans le monde réel**, seulement son importance **pour le modèle** (Gianluca Zuin).

> 📝 **Tes notes :** « permet une sélection d'attributs au final ».
> ⚠️ **À nuancer.** Une valeur de Shapley est **locale** (un exemple). Pour sélectionner des attributs, il faut d'abord **agréger** (ex. moyenne des $|\phi_j|$ sur le jeu de données). Et d'après la remarque ci-dessus, on mesure l'importance **pour ce modèle**. Deux attributs corrélés peuvent se partager la contribution, et un attribut causalement important mais redondant peut sembler peu utile.

### 4b. Dépendante du modèle (model-specific)

<a id="s4b-1"></a>
#### Explications pour les réseaux de neurones en vision : méthodes à base de gradient

##### Cartes de saillance : algorithme « Vanilla gradient »

On calcule le **gradient d'une sortie logit par rapport aux pixels** de l'image d'entrée :

$$
\frac{\partial y_c}{\partial X}
$$

- $y_c$ : logit de la classe $c$ de **plus fort logit**, c'est-à-dire la classe prédite
- $X$ : image d'entrée (tenseur des pixels)

Typiquement, un **gradient élevé** indique qu'un **pixel est important** pour la sortie. On affiche souvent la valeur absolue (« Absolute Gradients »). Le résultat est très bruité (exemple de l'étal de sushis).

##### Input × Gradients

On multiplie le gradient par la valeur de l'entrée : $X \odot \frac{\partial y_c}{\partial X}$.

**Exemple du cours** : $y = 100\,x_1 + 5\,x_2 + 1000$, avec $x_1$ = âge et $x_2$ = salaire.
- $\frac{\partial y}{\partial x_1} = 100$ et $\frac{\partial y}{\partial x_2} = 5$ : **$x_2$ semble moins important**.
- Mais pour un client de **30 ans** gagnant **50 000 €** : $x_1\frac{\partial y}{\partial x_1} = 3000$ et $x_2\frac{\partial y}{\partial x_2} = 250\,000$. **L'importance du salaire apparaît cette fois bien plus grande.**

> 💡 Même leçon que pour les $\beta_j$ (§3.1) : un gradient seul ignore l'**échelle** de la variable. Pour un modèle linéaire, Input × Gradient redonne exactement l'**effet individuel** $\beta_j x_j$.

**Problèmes :**
1. Dans une image, **les valeurs des pixels n'ont pas de rapport linéaire direct avec la sortie**. Le résultat dépend de l'**encodage** choisi. Avec une normalisation min-max, les **pixels sombres ont des valeurs faibles** et sont donc défavorisés. Avec la normalisation ImageNet, ce sont les pixels de valeur centrale (souvent le fond) qui le sont.
2. La méthode met en évidence la **structure de l'image**, ce qui est **plaisant au regard**, mais **pas nécessairement relié** à ce que le système utilise pour prédire.

##### Guided Backpropagation (GBP)

On **modifie la rétropropagation** standard en **ne laissant passer que les gradients positifs**.
- En rétropropagation standard, les **gradients négatifs** servent à **réduire l'activation** des éléments **non associés** à la classe cible.
- Avec la GBP, on veut visualiser **uniquement** les éléments associés à la classe cible.
- En **supprimant les gradients négatifs**, on **réduit le bruit** et on obtient des **visualisations plus nettes**.
- La GBP peut aussi servir à identifier **ce qui est important dans les couches intermédiaires**. On calcule alors des gradients par rapport aux cartes de caractéristiques (1re couche après l'entrée, 6e couche…).

> 🔎 **Complément :** concrètement, on ne rétropropage un gradient à travers une ReLU que si l'activation en avant **et** le gradient arrière sont tous deux positifs.

##### Grad-CAM (*Gradient-weighted Class Activation Mapping*)

- La plupart des méthodes propagent le gradient **jusqu'à l'image d'entrée**. **Grad-CAM s'arrête à la dernière couche de convolution.**
- Grad-CAM est **moins précise** que d'autres méthodes, mais met en évidence les **régions importantes** pour une classification donnée. Sa sortie est une **heat map**.
- Grad-CAM est **rapide** et **applicable à tout réseau convolutif**, d'où sa popularité malgré ses défauts.
- La dernière couche convolutive est proche de la sortie et contient beaucoup d'**informations sémantiques et spatiales**. Les gradients sont donc les plus forts aux **emplacements des objets ayant contribué positivement au logit cible**.
- Son inconvénient est que ces couches profondes ont une **résolution plus faible**.

> 🔎 **Complément : la formule de Grad-CAM.** Notons $A^k$ la $k$-ième carte de caractéristiques de la dernière couche convolutive, de taille $u \times v$, avec $Z = u \cdot v$.
>
> $$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y_c}{\partial A^k_{ij}} \qquad\qquad L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\Big(\sum_k \alpha_k^c\,A^k\Big)$$
>
> - $\alpha_k^c$ : importance de la carte $k$ pour la classe $c$ (moyenne spatiale des gradients)
> - $A^k_{ij}$ : activation de la carte $k$ à la position $(i,j)$
> - le ReLU ne garde que les zones qui contribuent **positivement** à la classe $c$
>
> La heat map de taille $u\times v$ est ensuite agrandie à la taille de l'image.

**Ce qui paraît être un succès :**
- Grad-CAM localise correctement un pot de fleurs.
- Sur une image adverse classée *Airliner* à 0,9999, Grad-CAM pour « Dog » et « Cat » **localise encore correctement** le chien et le chat. Cela montre sa robustesse au bruit adverse.

**Ce qui ne paraît pas être un succès :** des images classées *lévrier* (35 %) ou, à tort, *anguille* (70 %), dont la heat map n'éclaire pas la décision.

**Limites de la GBP** (qui motivent la combinaison avec Grad-CAM) :
- La GBP **ne distingue pas les classes**. Si plusieurs animaux à fourrure sont dans l'image, elle peut éclairer **toutes les zones de fourrure**, quel que soit l'animal. Elle montre ce qui **contribue** à l'activation de la classe cible, mais **pas exclusivement ce qui est spécifique** à cette classe.
- **Retour sur la GBP** : les premières couches extraient des **formes basiques** (contours, textures), combinées ensuite en descripteurs plus abstraits. Or une carte d'une couche antérieure peut contenir des **caractéristiques similaires**. Un détecteur de contours peut donc contribuer positivement à une couche profonde **même si ce contour vient d'une région non pertinente** de l'image.

##### Guided Grad-CAM

| Méthode | Force |
|---|---|
| **Guided Backprop** | Met en évidence les **détails** (haute résolution) |
| **Grad-CAM** | **Discrimine les classes** : révèle les parties pertinentes pour une classe donnée |

Les deux sont **complémentaires**, d'où **Guided Grad-CAM**.

> 🔎 **Complément :** en pratique, Guided Grad-CAM = (carte GBP) × (heat map Grad-CAM agrandie), multiplication pixel à pixel. On obtient des détails fins, mais seulement dans les régions propres à la classe.

##### SmoothGrad : « ajouter du bruit pour enlever le bruit »

On calcule les cartes de saillance de **plusieurs copies bruitées** de l'image, puis on fait leur **moyenne**. La carte obtenue est beaucoup plus nette. Exemple du cours : SmoothGrad + GBP sur un bouquetin.

> 🔎 **Complément (formule) :** pour $n$ copies bruitées de l'image $x$,
>
> $$\widehat{M}_c(x) = \frac{1}{n}\sum_{k=1}^{n} M_c\big(x + \varepsilon_k\big), \qquad \varepsilon_k \sim \mathcal{N}(0,\sigma^2)$$
>
> - $M_c$ : carte de saillance de base (ex. $\partial y_c/\partial X$) ; $n$ : nombre de copies ; $\sigma$ : niveau de bruit
>
> Le bruit « haute fréquence » du gradient se compense d'une copie à l'autre, alors que les zones réellement importantes restent stables.

##### Integrated Gradients

Principe illustré dans le cours :
1. partir d'une **image de référence** (*baseline*, ici noire) ;
2. interpoler progressivement vers l'image réelle (étapes 1 à 5) ;
3. calculer $\partial y_c/\partial X$ à chaque étape et en faire la **moyenne** ;
4. **multiplier cette moyenne par (image − baseline)**.

> 🔎 **Complément (formule) :** pour une entrée $x$ et une baseline $x'$, la composante $i$ vaut
>
> $$\mathrm{IG}_i(x) = (x_i - x'_i)\int_0^1 \frac{\partial y_c\big(x' + \alpha(x - x')\big)}{\partial x_i}\,d\alpha$$
>
> - $x$ : image ; $x'$ : baseline ; $\alpha$ : position sur le chemin baseline → image
>
> C'est un « Input × Gradient » corrigé : on intègre le gradient le long du chemin au lieu de le prendre en un seul point. Cela évite la **saturation** (gradient nul alors que le pixel compte). Propriété : $\sum_i \mathrm{IG}_i = y_c(x) - y_c(x')$, une forme d'efficacité comme chez Shapley.

##### Limites générales des méthodes par gradient en vision

- **Difficile de savoir si une « explication » est correcte**, c'est-à-dire réellement **causale** dans la prédiction.
- On peut croire qu'une explication est bonne **parce qu'elle satisfait nos attentes** (biais de confirmation).
- De **petites variations des images** peuvent donner des résultats très différents. Exemple : **ajouter une constante à tous les pixels** (Kindermans et al.). Le **neurone de biais corrige parfaitement** le changement et les poids restent les mêmes, donc la prédiction est identique. **Pourtant les « explications » sont différentes !**

<a id="s4b-2"></a>
#### Concept Bottleneck Models (CBM)

Ce sont des modèles conçus pour **décider via des concepts intermédiaires interprétables par l'humain**, plutôt que de passer directement des données brutes à l'étiquette.

On part d'un **backbone CNN partagé** qui produit une représentation $\omega(x)$ :
- la **prédiction standard** va directement de $\omega(x)$ à la classe $\hat y$ ;
- la **prédiction par concepts** passe par un **prédicteur de concepts** $\hat a$, puis seulement ensuite par un prédicteur de classe $\hat y$.

**Deux étapes :**
1. **Encodeur de concepts $g(x)$** : associe les données brutes (pixels, texte) à des **concepts définis par l'humain** $c$. Exemples : « excroissances osseuses » ou « sclérose » en radiographie ; « couleur des ailes » ou « longueur du bec » pour les oiseaux.
2. **Prédicteur $f(c)$** : n'utilise **que les concepts** prédits pour produire l'étiquette $y$.

**Apprentissage.** On suppose disposer d'un ensemble $(x_i, y_i, c_i)_{1\le i\le m}$ où les concepts $c_i$ sont **annotés pour chaque exemple**. Le modèle a la forme $f(g(x))$, avec :
- $g : \mathbb{R}^d \to \mathbb{R}^k$ s'il y a $k$ concepts ;
- $f : \mathbb{R}^k \to \mathbb{R}^n$ s'il y a $n$ classes.

On utilise deux fonctions de coût dédiées :

$$
\hat f = \underset{f}{\mathrm{ArgMin}}\sum_{i=1}^{m}\ell_Y\big(f(c_i),\,y_i\big)
\qquad\qquad
\hat g = \underset{g}{\mathrm{ArgMin}}\sum_{i=1,\,j=1}^{m,\,k}\ell_{C_j}\big(g_j(x_i),\,c_i^{(j)}\big)
$$

- $\ell_Y$ : perte sur la classe ; $\ell_{C_j}$ : perte sur le concept $j$
- $g_j(x_i)$ : concept $j$ prédit pour $x_i$ ; $c_i^{(j)}$ : sa vraie valeur annotée

**Intérêt : les interventions.** On peut **corriger à la main** un concept mal prédit et la classe suit. Par exemple, un goéland est classé à tort *Herring Gull* parce que le concept « dessous de queue noir » vaut 0,82. Si on le met à 0 et « dessous de queue gris » à 1, le modèle prédit correctement *Glaucous Winged Gull*.

> 📝 **Tes notes :** « on force une représentation : plus de connaissance a priori, mais système explicatif par construction ».
> ✅ Bon lien avec le §1.3 (conservation de l'information). Les concepts sont de la **connaissance a priori injectée**, ce qui demande un **coût d'annotation** des $c_i$.

<a id="s4b-3"></a>
#### L'évaluation des systèmes explicatifs (à base de concepts)

1. **Performance de prédiction** : elle ne doit pas être détériorée.
2. **Performance de reconnaissance des concepts intermédiaires.**
3. **Stabilité des poids entre concepts et classes**, d'une exécution à l'autre et selon les conditions initiales.
4. Les **concepts importants dans le codage** sont-ils **importants pour le calcul de la sortie** ?
5. **Sensibilité de la sortie aux interventions** sur les poids des concepts qui codent un exemple.

---

<a id="s5"></a>
## 5. Quel avenir pour l'IA explicable ?

**Qu'est-ce qui manque au ML actuel ?**
- Se concentrer **exclusivement sur la performance prédictive** donne une **formulation incomplète** du problème d'apprentissage.
- On veut aussi l'**interprétabilité des résultats**, l'**interprétabilité du processus**, et **mieux comprendre le monde** en intégrant la fonction apprise dans une théorie existante.
- Il faut donc **changer le critère inductif** utilisé en ML.

**Les explications humaines sont-elles bonnes ?** Expérience de l'**effet du témoin** (*bystander effect*) :
- Des participants remplissent un questionnaire. Dans la pièce voisine, l'assistante tombe de sa chaise et crie de douleur.
- **Condition 1 (seul)** : **70 %** interviennent.
- **Condition 2 (avec un complice impassible)** : **7 %** seulement interviennent.
- Interrogés ensuite, les participants de la condition 2 jugent que ce n'était pas grave. **Aucun ne mentionne l'autre personne**, et ils affirment même que sa présence **n'a eu aucune influence**.

> 💡 **Leçon** : les raisons invoquées **viennent après la décision**, elles sont **manifestement fausses** : **les raisons sont des justifications**. Un parallèle avec les explications post-hoc des modèles est possible.

**Pourquoi cela devient-il important pour les machines ?**

Les IA sont de plus en plus des **sociétés d'agents** qui communiquent et décident en fonction de leurs interactions. **Chaque agent doit pouvoir expliquer les « raisons » de ses sorties aux autres agents.**

- **Accident d'une voiture autonome** (en mode autopilote, collision avec une semi-remorque) :
  - le **radar** a bien détecté la semi-remorque, mais de nombreux panneaux routiers avaient une signature radar similaire ;
  - la **caméra** doutait de sa détection à cause d'un ciel laiteux éblouissant, et la remorque était blanche.
  - Et si chaque capteur avait **justifié son doute** ? Le radar aurait dit « ça ressemble aux panneaux que j'ai souvent vus sur cette route », la caméra « la couleur est proche du ciel laiteux aujourd'hui ». Et si radar et caméra avaient un **long historique d'échanges** et s'influençaient mutuellement ?
- **Interactions entre modules d'apprentissage** : un système publicitaire adaptatif a **deux sous-systèmes**, l'un place les liens, l'autre choisit les annonces. Ils **s'influencent mutuellement** : chacun apprend sur des **données de clics** qui dépendent aussi de l'autre système et de facteurs non contrôlés (prix, requêtes). C'est un cadre de **raisonnement contrefactuel** (Bottou et al.).

**Questions ouvertes :**
- Interprétabilité **vs** explicabilité ?
- Comment **évaluer** une explication ?
- Quel lien entre explications et **causalité** ?
- Une explication est-elle **symbolique** par nature ?
- Explication « **one-shot** » ou par **interactions** ?
- Expliquer donne-t-il la possibilité de **tromper le système** ?
- À qui s'adressent les explications : **experts IA**, **experts du domaine**, **utilisateurs** ?
- Les **« chains of thought »** sont-elles des explications fiables ?
- Quelle place pour la **causalité** ? Est-ce le graal ?

> 🔎 **Complément :** sur les *chains of thought*, l'effet du témoin donne un indice. Comme chez l'humain, le raisonnement affiché par un LLM peut être une **justification a posteriori** plutôt que la cause réelle de sa réponse. Sa fidélité (*faithfulness*) doit donc être vérifiée.

**Ressources** : adataodyssey.com ; C. Molnar, *Interpretable Machine Learning* ; C. Rudin, *Stop explaining black box ML models for high stakes decisions and use interpretable models instead*.

---

<a id="s6"></a>
## 6. Quiz d'auto-évaluation

Barème total : **40 points**.

<details>
<summary><b>Question 1 (2 pts) :</b> Quelle différence entre explication « de surface » et explication « profonde » ? Pour qui chacune est-elle utile ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **De surface** : liée au **fonctionnement du système** (quelles règles ont été déclenchées). Elle sert à **vérifier** ce fonctionnement et s'adresse à l'**expert IA**.
- **Profonde** : la vraie raison causale, **difficile à découvrir et à expliciter**. Elle est nécessaire pour l'**expert du domaine** et les **utilisateurs**.
</details>

<details>
<summary><b>Question 2 (2 pts) :</b> Que dit le « théorème de conservation » du ML, et quel compromis en découle ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

L'apprentissage **ne fait que reformuler** ses entrées : **aucune information n'est ajoutée**, tout vient des **données + connaissances a priori**. Le compromis est : peu de données exige beaucoup de connaissances a priori ; beaucoup de données permet d'en mettre moins. La connaissance a priori peut être dans les **données** (choix des descripteurs, normalisation, augmentation…) ou dans l'**algorithme** (biais de représentation, architecture CNN).
</details>

<details>
<summary><b>Question 3 (3 pts) :</b> Quand l'interprétabilité n'est-elle pas nécessaire ? Citez les 4 cas où elle l'est. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**Pas nécessaire** : faible risque (recommandation de film) ou bonnes garanties de performance (reconnaissance de caractères).

**Nécessaire** :
1. **décisions à haut risque** (chirurgie, centrale nucléaire, véhicule autonome) ;
2. **curiosité scientifique** (résultats surprenants, intégration dans une théorie du domaine) ;
3. **débogage** (contrefactuels, vélo avec sacoches, biais de genre) ;
4. un système interprétable **peut être manipulé**, il doit donc reposer sur des **facteurs causaux**.
</details>

<details>
<summary><b>Question 4 (3 pts) :</b> Qu'est-ce que l'effet « Clever Hans » ? Donnez l'exemple de Pascal VOC et dites en quoi il diffère de la manipulation d'un système. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Le modèle donne la **bonne réponse pour la mauvaise raison** en exploitant une **corrélation accidentelle mais très prédictive**. Sur Pascal VOC, le classifieur reconnaît les chevaux grâce au **filigrane de copyright**. Une voiture artificielle portant ce filigrane est alors classée « cheval ».

La **manipulation**, elle, vient d'un **utilisateur** qui connaît la règle (« plus de 2 cartes de crédit ») et adapte ses entrées. Clever Hans est un défaut du modèle qu'on **détecte** grâce aux explications ; la manipulation est un risque **créé** par l'interprétabilité.
</details>

<details>
<summary><b>Question 5 (2 pts) :</b> Décrivez l'arbre des familles de méthodes d'interprétabilité. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **By-design** (régression linéaire, arbre) : l'importance des descripteurs découle de la nature du modèle.
- **Post-hoc** : l'explication est découplée de l'apprentissage.
  - **Model-agnostic**, soit **local** (une prédiction, ex. LIME, Shapley), soit **global** (stratégie d'ensemble).
  - **Model-specific** : cartes de saillance, concepts.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Quelles sont les 4 hypothèses d'un modèle linéaire correct ? Que se passe-t-il si la normalité est violée ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Fonction cible linéaire (ou proche).
2. Sortie normale **conditionnellement aux descripteurs**. Sinon, les **estimateurs de confiance** (IC, p-valeurs) sont **invalides**.
3. **Homoscédasticité** : variance constante quelle que soit la valeur des descripteurs.
4. Descripteurs **non corrélés**.
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Peut-on dire que le descripteur au plus grand βⱼ est le plus important ? Quel indicateur utilise le cours ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

**Non** : $\beta_j$ dépend de l'**unité** du descripteur. Par exemple, `INCOME` a $\beta\approx 9\times10^{-5}$ mais $t\approx 11$. Le cours utilise la **t-statistique** :

$$t_{\widehat\beta_j} = \frac{\widehat\beta_j}{\mathrm{SE}(\widehat\beta_j)}$$

Plus $|t|$ est grand, plus le coefficient est important. Associée à la p-valeur, une valeur < 0,05 indique un effet significatif. On peut aussi standardiser les descripteurs avant de comparer les coefficients.
</details>

<details>
<summary><b>Question 8 (2 pts) :</b> Pourquoi des descripteurs corrélés posent-ils problème à l'interprétation d'une régression linéaire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Si deux descripteurs sont corrélés et que l'un reçoit un fort coefficient positif, **l'autre peut recevoir un coefficient négatif** alors qu'il est **positivement corrélé à la cible**. Les coefficients ne reflètent plus l'effet propre de chaque variable. On détecte ce problème avec la **matrice de corrélation**.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Les arbres de décision capturent-ils mal les dépendances linéaires ou non linéaires ? Pourquoi ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Ils capturent mal les dépendances **linéaires**. Leurs coupes sont parallèles aux axes ($x_j \le s$), si bien qu'une relation linéaire ou une frontière oblique demande un escalier de nombreuses coupes. La solution est d'utiliser des **arbres obliques**. Leurs autres défauts : ils sont **instables** et deviennent **opaques** si les descripteurs sont mal choisis.
</details>

<details>
<summary><b>Question 10 (4 pts) :</b> Donnez les 6 étapes de LIME et le rôle de π_x et Ω(g) dans sa formulation. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

1. Choisir l'exemple.
2. Générer des exemples virtuels en **permutant des valeurs d'attributs**.
3. **Pondérer** les exemples par leur **distance** à l'exemple.
4. **Les étiqueter avec le modèle** $f$.
5. Entraîner un **modèle surrogé simple** (linéaire).
6. L'interpréter.

$$\xi(x) = \arg\min_{g\in G}\ \mathcal{L}(f,g,\pi_x) + \Omega(g)$$

$\pi_x$ est le **noyau de proximité** (étape 3), un noyau exponentiel de largeur $0{,}75\sqrt{d}$. $\Omega(g)$ **pénalise la complexité** de $g$ pour qu'il reste interprétable. Le choix de la distance est **crucial**.
</details>

<details>
<summary><b>Question 11 (4 pts) :</b> Trois élèves : v(E1)=7500, v(E2)=5000, v(E3)=2500, v(E1E2)=10000, v(E1E3)=7500, v(E2E3)=5000, v(E1E2E3)=10000. Calculez la valeur de Shapley de E3. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

On liste les apports de E3 dans les 6 ordres :
- E3 arrive **dernier** (2 ordres) : $10000 - 10000 = 0$ ;
- E3 arrive en **2e après E1** : $7500 - 7500 = 0$ ;
- E3 arrive en **2e après E2** : $5000 - 5000 = 0$ ;
- E3 arrive **premier** (2 ordres) : $2500 - 0 = 2500$.

$$\mathrm{Val}(E_3) = \frac{0+0+0+0+2500+2500}{6} \approx 833$$

Vérification de l'efficacité : $5833 + 3333 + 833 \approx 10000$.
</details>

<details>
<summary><b>Question 12 (4 pts) :</b> Écrivez la formule de la valeur de Shapley φⱼ et expliquez le poids |S|!(d−|S|−1)!/d! ainsi que le rôle de X_C. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

$$\phi_j(val) = \sum_{S\subseteq\{1,\dots,d\}\setminus\{j\}} \frac{|S|!\,(d-|S|-1)!}{d!}\big(val(S\cup\{j\}) - val(S)\big)$$

$$val_{\mathbf{x}}(S) = \int \hat h(x_1,\dots,x_d)\,\mathrm{dP}_{X_C} - \mathbb{E}[\hat h(\mathbf{X})]$$

Le poids est la proportion des $d!$ ordres d'arrivée où exactement les joueurs de $S$ précèdent $j$ : c'est la **moyenne sur tous les ordres**. $X_C$ désigne les attributs **hors coalition**. On **intègre** sur leur distribution (on « oublie » leur valeur), en pratique en tirant leurs valeurs dans l'ensemble d'apprentissage.
</details>

<details>
<summary><b>Question 13 (3 pts) :</b> Avec h(x₁,x₂) = 200x₁ + 1000x₂, une personne de 20 ans diplômée et P(x₂=1)=½, que vaut la contribution marginale du diplôme à la coalition {âge} ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Si on ignore $x_2$, on intègre sur sa distribution :

$$\mathrm{Val}(x_1) = \tfrac12 h(20,0) + \tfrac12 h(20,1) = \tfrac12(4000) + \tfrac12(5000) = 4500$$

La contribution vaut donc $5000 - 4500 = 500$. C'est aussi la valeur de Shapley complète, puisque pour un modèle linéaire à variables indépendantes $\phi_2 = 1000\,(1 - 0{,}5) = 500$.
</details>

<details>
<summary><b>Question 14 (2 pts) :</b> Comment contourne-t-on le coût exponentiel du calcul des valeurs de Shapley ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Par une **approximation de Monte-Carlo**. À chaque itération, on tire une instance $\mathbf{z}$ et une permutation, puis on construit $\mathbf{x}_{+j}$ (valeurs de $\mathbf{x}$ jusqu'à $j$ inclus, puis $\mathbf{z}$) et $\mathbf{x}_{-j}$ (même chose, mais $j$ pris dans $\mathbf{z}$). On calcule $\phi_j^{(m)} = \hat f(\mathbf{x}_{+j}) - \hat f(\mathbf{x}_{-j})$, puis :

$$\phi_j = \frac1M\sum_{m=1}^{M} \phi_j^{(m)}$$
</details>

<details>
<summary><b>Question 15 (2 pts) :</b> Différence entre ce que fournissent LIME et Shapley, et mise en garde du cours sur SHAP ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

LIME donne l'**importance relative** des attributs. Shapley donne la contribution **« exacte »** de **chaque valeur mesurée**, dont la somme fait prédiction − moyenne.

**Mise en garde** : SHAP mesure l'importance d'un descripteur **pour le modèle**, **pas dans le monde réel**.
</details>

<details>
<summary><b>Question 16 (3 pts) :</b> Avec y = 100x₁ + 5x₂ + 1000 (âge, salaire), pourquoi le gradient seul est-il trompeur ? Quels sont les deux problèmes d'Input×Gradient sur les images ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Les gradients valent 100 et 5, donc le salaire semble peu important. Pour un client de 30 ans gagnant 50 000 €, Input × Gradient donne $3000$ contre $250\,000$ : le salaire domine. Le gradient ignore l'**échelle** des variables.

Problèmes sur les images :
1. les valeurs de pixels n'ont pas de rapport linéaire direct avec la sortie, et le résultat dépend de l'**encodage** (pixels sombres défavorisés en min-max) ;
2. la méthode montre la **structure de l'image**, joli mais pas forcément ce qu'utilise le modèle.
</details>

<details>
<summary><b>Question 17 (3 pts) :</b> Comparez Guided Backprop et Grad-CAM. Pourquoi les combiner ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **GBP** : ne rétropropage que les **gradients positifs**, ce qui donne des cartes nettes et **détaillées**. Mais elle **ne discrimine pas les classes** (toutes les zones de fourrure s'allument) et peut éclairer des contours de régions non pertinentes.
- **Grad-CAM** : s'arrête à la **dernière couche convolutive**. Elle est **discriminante par classe**, rapide, applicable à tout CNN, mais de **basse résolution**.

**Guided Grad-CAM** combine les deux pour obtenir des détails fins **et** spécifiques à la classe.
</details>

<details>
<summary><b>Question 18 (2 pts) :</b> Principe de SmoothGrad et d'Integrated Gradients ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- **SmoothGrad** : on **moyenne les cartes de saillance de copies bruitées** de l'image, ce qui enlève le bruit du gradient.
- **Integrated Gradients** : on part d'une **baseline** (image noire), on interpole vers l'image, on **moyenne les gradients** le long du chemin, puis on **multiplie par (image − baseline)**.
</details>

<details>
<summary><b>Question 19 (2 pts) :</b> Citez les limites générales des méthodes par gradient, dont l'expérience de Kindermans et al. <i>(Cliquez ici pour la réponse)</i></summary>
<br>

- Il est difficile de savoir si l'explication est **causale**.
- On juge une explication bonne **parce qu'elle colle à nos attentes**.
- Les résultats sont **instables**. Si on ajoute une constante à tous les pixels, le biais compense, les poids et la prédiction sont inchangés, **mais les explications diffèrent**.
</details>

<details>
<summary><b>Question 20 (3 pts) :</b> Comment fonctionne un Concept Bottleneck Model ? Quelles données faut-il ? Qu'est-ce qu'une intervention ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

On combine deux étapes : $g:\mathbb{R}^d\to\mathbb{R}^k$ prédit $k$ **concepts humains**, puis $f:\mathbb{R}^k\to\mathbb{R}^n$ prédit la classe **à partir des seuls concepts**. Il faut des données $(x_i, y_i, c_i)$ avec des **concepts annotés**, et on utilise deux pertes, $\ell_Y$ sur la classe et $\ell_{C_j}$ sur les concepts.

Une **intervention** consiste à corriger à la main un concept mal prédit, ce qui corrige la classe (goéland : « dessous de queue noir » → gris).
</details>

<details>
<summary><b>Question 21 (2 pts) :</b> Que montre l'expérience de l'effet du témoin, et quel lien avec l'XAI ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>

Seuls, 70 % des participants aident ; avec un complice impassible, seulement 7 %. Aucun n'attribue sa décision à l'autre personne. Les **raisons invoquées viennent après la décision** : ce sont des **justifications**, souvent fausses.

Pour l'XAI, une explication (humaine, post-hoc, ou *chain of thought*) peut être plausible sans être la vraie cause. Il faut donc **évaluer** les explications, et le cours appelle à changer le **critère inductif**.
</details>

---

<a id="s7"></a>
## 7. Pièges à éviter

> 🚩 **$\beta_j$ grand ≠ descripteur important** : le coefficient dépend de l'unité. Il faut regarder la **t-statistique** ou standardiser les descripteurs.

> 🚩 **Normalité de la régression** : c'est $y\,|\,\mathbf{x}$ (les résidus) qui doit être normal, pas $y$ globalement. Si cette hypothèse est violée, ce sont les **intervalles de confiance et les p-valeurs** qui deviennent invalides.

> 🚩 **Descripteurs corrélés** : un coefficient négatif n'implique pas une corrélation négative avec la cible.

> 🚩 **Arbres** : ils capturent mal les dépendances **linéaires** (pas non linéaires). Pensez aux arbres obliques.

> 🚩 **Clever Hans ≠ manipulation** : le premier est une corrélation fallacieuse apprise par le modèle (à détecter) ; la seconde, un utilisateur qui exploite une règle connue (à prévenir avec des facteurs causaux).

> 🚩 **Shapley = moyenne sur tous les ordres**, pas l'apport dans un seul ordre d'ajout. Et dans la formule, $X_C$ désigne les attributs **hors** coalition, sur lesquels on intègre.

> 🚩 **Shapley explique prédiction − prédiction moyenne**, pas la prédiction brute (appartement : −10 k€ par rapport à 310 k€).

> 🚩 **Shapley est local** et mesure l'importance **pour le modèle**, pas dans le monde réel. Pour sélectionner des attributs, il faut agréger et se méfier des attributs corrélés.

> 🚩 **Coquille de la diapo 70** : Val(E1) = 5833, pas 7500.

> 🚩 **Gradient ≠ importance** : il ignore l'échelle (d'où Input × Gradient), qui dépend à son tour de l'encodage des pixels.

> 🚩 **Une belle carte de saillance n'est pas une preuve.** Elle peut refléter la structure de l'image ou nos attentes (cf. Kindermans : explications différentes pour un modèle équivalent).

> 🚩 **GBP ne discrimine pas les classes** ; c'est **Grad-CAM** qui le fait, en basse résolution. Ne pas inverser les rôles.

> 🚩 **CBM** : il faut des **concepts annotés** pour chaque exemple, et le prédicteur final n'utilise **que** les concepts.
