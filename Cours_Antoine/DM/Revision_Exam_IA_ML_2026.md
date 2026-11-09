# Révision — Examen IA_ML (10 novembre 2026)

## Sommaire

- [Partie A — Questions connues](#partie-a)
  - [1. Question 4 — Classifieur « classe majoritaire »](#q4)
  - [2. Question 5 — Courbe ROC](#q5)
  - [3. Question 6 — Précision et rappel](#q6)
  - [4. Question 7 — Minimisation du risque empirique](#q7)
  - [5. Question 11 — One-hot encoding](#q11)
  - [6. Question 15 — Arbre de décision](#q15)
  - [7. Question 17 — Valeurs de Shapley](#q17)
- [Partie B — Questions types pour les parties cachées (avec corrigés)](#partie-b)
  - [8. Q1 — Espace des versions](#q1)
  - [9. Q2 et Q3 — Questions de cours (2 pts)](#q2q3)
  - [10. Q8 à Q14 — Questions courtes (1 pt)](#qcourtes)
  - [11. Q16 — Causalité](#q16)
  - [12. Q18 — Explicabilité et éthique](#q18)
- [13. Pièges à éviter](#pieges)

---

<a id="partie-a"></a>
# Partie A — Questions connues

<a id="q4"></a>
## 1. Question 4 (1 pt) — Classifieur « classe majoritaire »

**Réponse : le taux d'erreur en test sera presque toujours supérieur à 50 %, donc pire que le hasard.**

Hypothèses : 100 exemples au total, 50 de la classe 1 et 50 de la classe 2 ; 70 en apprentissage, 30 en test.

Si l'ensemble d'apprentissage contient $a$ exemples de la classe 1 avec $a > 35$, l'algorithme prédit toujours « 1 ». Le test contient alors $50 - a < 15$ exemples de la classe 1 sur 30 : la classe prédite est **minoritaire dans le test**.

$$\text{Erreur}_{test} = \frac{30 - (50 - a)}{30} = \frac{a - 20}{30} > 50\ \%$$

où $a$ est le nombre d'exemples de la classe 1 dans l'ensemble d'apprentissage.

Exemple : 36/34 en apprentissage donne 14/16 en test, soit une erreur de 16/30 ≈ 53 %. Plus le déséquilibre en apprentissage est fort, plus l'erreur en test est grande. Le seul cas à 50 % est l'égalité 35/35 : la prédiction est aléatoire, et l'erreur vaut 50 % en espérance.

> 💡 **Leçon :** avec des données parfaitement équilibrées, apprentissage et test sont anti-corrélés : retirer des exemples d'une classe pour l'apprentissage les rend plus rares en test. L'évaluation est biaisée, d'où l'intérêt de la **stratification**. Cas extrême : le leave-one-out donne ici 100 % d'erreur.

<a id="q5"></a>
## 2. Question 5 (2 pts) — Courbe ROC

La courbe ROC mesure la capacité d'un classifieur à **séparer les deux classes**, pour **tous les seuils de décision possibles**. Chaque seuil donne un point de la courbe.

$$FPR = \frac{FP}{FP + TN} \qquad TPR = \frac{TP}{TP + FN}$$

où $TP$, $FP$, $TN$, $FN$ sont les vrais positifs, faux positifs, vrais négatifs et faux négatifs.

- **Abscisse : taux de faux positifs** $FPR$ (= 1 − spécificité).
- **Ordonnée : taux de vrais positifs** $TPR$ (= rappel = sensibilité).

Cas remarquables :

- **Apprentissage parfait** : la courbe monte verticalement de (0,0) à (0,1), puis va horizontalement jusqu'à (1,1). Un seuil détecte 100 % des positifs sans aucun faux positif. AUC = 1.
- **Classifieur aléatoire** : la diagonale de (0,0) à (1,1). AUC = 0,5.
- **Pire apprentissage** : la courbe passe par (1,0). AUC = 0. Le classifieur se trompe systématiquement ; il suffirait d'inverser ses prédictions pour obtenir un classifieur parfait.

> ⚠️ Le pire classifieur réellement *inutile* est celui de la diagonale. Mentionner les deux interprétations du « pire » à l'examen.

Plus la courbe est proche du coin supérieur gauche, meilleur est le classifieur. L'AUC résume la courbe en un nombre.

<a id="q6"></a>
## 3. Question 6 (2 pts) — Précision et rappel

$$\text{Précision} = \frac{TP}{TP + FP} \qquad \text{Rappel} = \frac{TP}{TP + FN}$$

- **Précision** : parmi ce que j'ai prédit « + », quelle proportion est vraiment « + » ? C'est la *fiabilité* des prédictions positives.
- **Rappel** : parmi les vrais « + », quelle proportion ai-je retrouvée ? C'est l'*exhaustivité*.

**Pourquoi un compromis :** les deux dépendent du seuil de décision.

- Seuil bas : on prédit « + » plus souvent. Le rappel augmente, mais les faux positifs aussi, donc la précision baisse.
- Seuil haut : on ne garde que les cas très sûrs. La précision augmente, mais on rate des positifs, donc le rappel baisse.

**Exemple du moteur de recherche :**

- Un moteur qui retourne **toutes les pages du web** a un rappel de 100 %, mais une précision quasi nulle : il est noyé dans le bruit.
- Un moteur qui ne retourne que **le seul lien dont il est certain** a une précision proche de 100 %, mais un rappel minuscule.

Un bon moteur cherche un équilibre, mesuré par exemple par la F-mesure :

$$F_1 = \frac{2PR}{P + R}$$

où $P$ est la précision et $R$ le rappel.

<a id="q7"></a>
## 4. Question 7 (2 pts) — Minimisation du risque empirique (MRE)

**Ce que prescrit le principe :** la distribution réelle des données est inconnue, donc le vrai risque $R(h) = \mathbb{E}[\ell(h(x), y)]$ ne peut pas être calculé. La MRE prescrit de choisir l'hypothèse qui minimise l'erreur moyenne sur l'échantillon d'apprentissage :

$$\hat{h} = \arg\min_{h \in \mathcal{H}} R_{emp}(h) = \arg\min_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^{m} \ell(h(x_i), y_i)$$

où $\mathcal{H}$ est l'espace des hypothèses, $m$ le nombre d'exemples, $\ell$ la fonction de perte et $(x_i, y_i)$ les exemples d'apprentissage.

**Pourquoi régulariser :** si $\mathcal{H}$ est trop riche, on peut trouver une hypothèse d'erreur nulle en apprentissage qui « apprend par cœur », y compris le bruit. Le risque empirique devient un estimateur **trop optimiste** du vrai risque : c'est le **sur-apprentissage**. La théorie (Vapnik, dimension VC) montre que :

$$R(h) \leq R_{emp}(h) + \text{terme de complexité}(\mathcal{H}, m)$$

On minimise donc un risque régularisé :

$$R_{reg}(h) = R_{emp}(h) + \lambda \, \Omega(h)$$

où $\Omega(h)$ mesure la complexité de l'hypothèse (norme L2 ou L1 des poids, profondeur d'un arbre…) et $\lambda \geq 0$ règle le compromis entre ajustement aux données et simplicité, c'est-à-dire le compromis biais-variance.

> 💡 La régularisation rend aussi le problème **bien posé** : la solution devient plus stable vis-à-vis de l'échantillon.

<a id="q11"></a>
## 5. Question 11 (1 pt) — One-hot encoding

Le one-hot encoding transforme une **variable catégorielle à $k$ modalités** en **$k$ variables binaires**, une par modalité. Pour chaque exemple, la variable de sa modalité vaut 1, toutes les autres valent 0.

| color | is_green | is_dark | is_light |
|---|---|---|---|
| green | 1 | 0 | 0 |
| dark | 0 | 1 | 0 |
| light | 0 | 0 | 1 |

**Intérêt :** un codage naïf comme green=1, dark=2, light=3 introduirait un **ordre et des distances artificielles** entre modalités. Le one-hot rend toutes les modalités équidistantes. Il est indispensable pour les modèles qui calculent sur des nombres : régression, réseaux de neurones, k-PPV.

<a id="q15"></a>
## 6. Question 15 (3 pts) — Arbre de décision

> ⚠️ Appliquer d'abord les modifications de l'énoncé : l'exemple 1 devient **false**, l'exemple 9 devient **true**.

- Exemples **true** : {2, 3, 4, 5, 6, 7, 8, 9}, soit **8**.
- Exemples **false** : {1, 10, 11, 12, 13, 14, 15, 16, 17}, soit **9**.

Entropie d'un ensemble $D$ à deux classes :

$$H(D) = -p_+ \log_2 p_+ - p_- \log_2 p_-$$

où $p_+$ et $p_-$ sont les proportions d'exemples true et false dans $D$. Entropie après un test sur un descripteur $A$ :

$$H_A(D) = \sum_{v} \frac{|D_v|}{|D|} H(D_v)$$

où $D_v$ est le sous-ensemble des exemples pour lesquels $A = v$.

### (a) Entropie avant tout test

$$H(D) = -\tfrac{8}{17}\log_2\tfrac{8}{17} - \tfrac{9}{17}\log_2\tfrac{9}{17} = \mathbf{0{,}998}$$

### (b) Après le test sur `texture`

| texture | exemples | true / false | entropie |
|---|---|---|---|
| clear | 1, 2, 3, 4, 5, 6, 8, 10, 15 (9 ex.) | 6 T / 3 F | 0,918 |
| slightly blurry | 7, 9, 13, 14, 17 (5 ex.) | 2 T / 3 F | 0,971 |
| blurry | 11, 12, 16 (3 ex.) | 0 T / 3 F | 0 |

$$H_{texture} = \tfrac{9}{17}(0{,}918) + \tfrac{5}{17}(0{,}971) + \tfrac{3}{17}(0) \approx 0{,}486 + 0{,}286 = \mathbf{0{,}772}$$

Gain d'information : $0{,}998 - 0{,}772 = 0{,}226$.

### (c) Après le test sur `umbilicus`

| umbilicus | exemples | true / false | entropie |
|---|---|---|---|
| hollow | 1, 2, 3, 4, 5, 13, 14 (7 ex.) | 4 T / 3 F | 0,985 |
| slightly hollow | 6, 7, 8, 9, 15, 17 (6 ex.) | 4 T / 2 F | 0,918 |
| flat | 10, 11, 12, 16 (4 ex.) | 0 T / 4 F | 0 |

$$H_{umbilicus} = \tfrac{7}{17}(0{,}985) + \tfrac{6}{17}(0{,}918) + 0 \approx 0{,}406 + 0{,}324 = \mathbf{0{,}730}$$

Gain d'information : $0{,}998 - 0{,}730 = 0{,}268$.

### (d) Meilleur descripteur

**`umbilicus`** : l'entropie après le test est plus faible (0,730 contre 0,772), donc le gain d'information est plus grand. C'est lui qu'on place à la racine.

> 🔎 Les valeurs fournies par l'énoncé (0,985 ; 0,918 ; 0,971) correspondent exactement à ces sous-ensembles, ce qui confirme qu'il fallait appliquer les modifications. $\frac{6}{9} = \frac{2}{3}$ et $\frac{4}{6} = \frac{2}{3}$ donnent tous deux 0,918.

<a id="q17"></a>
## 7. Question 17 (4 pts) — Valeurs de Shapley

La fonction $f$ sera donnée à l'examen. Méthode appliquée à une fonction d'entraînement :

$$f(x_1, x_2, x_3) = 1 + 2x_1 + 4\,x_2 x_3$$

On a $f(0,0,0) = 1$ et $f(1,1,1) = 7$.

### (a) Valeur des 8 coalitions

| S | ∅ | {1} | {2} | {3} | {1,2} | {1,3} | {2,3} | {1,2,3} |
|---|---|---|---|---|---|---|---|---|
| v(S) | 1 | 3 | 1 | 1 | 3 | 3 | 5 | 7 |

### (b) Contributions marginales sur les 6 ordres

Pour chaque ordre, on ajoute les descripteurs un par un et on note de combien chacun fait augmenter $v$ à son entrée.

| ordre | x1 | x2 | x3 |
|---|---|---|---|
| <1,2,3> | 2 | 0 | 4 |
| <1,3,2> | 2 | 4 | 0 |
| <2,1,3> | 2 | 0 | 4 |
| <2,3,1> | 2 | 0 | 4 |
| <3,1,2> | 2 | 4 | 0 |
| <3,2,1> | 2 | 4 | 0 |
| **moyenne φ** | **2** | **2** | **2** |

Formule directe, pour vérifier :

$$\varphi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n - |S| - 1)!}{n!}\,\big[v(S \cup \{i\}) - v(S)\big]$$

où $N = \{1, 2, 3\}$ est l'ensemble des descripteurs, $n = 3$ et $S$ une coalition ne contenant pas $i$. Avec $n = 3$, les poids valent 1/3 si $|S| = 0$, 1/6 si $|S| = 1$, 1/3 si $|S| = 2$.

### (c) Vérification de l'efficacité

$$\varphi_1 + \varphi_2 + \varphi_3 = 6 = f(1,1,1) - f(0,0,0) = 7 - 1 \ ✓$$

### (d) Interprétation

- **$\varphi_1 = 2$ :** $x_1$ intervient de façon **additive**, sans interaction. Il apporte toujours +2 quel que soit l'ordre, donc sa valeur de Shapley est exactement son coefficient.
- **$\varphi_3 \neq 0$ alors que $v(\{3\}) = v(\emptyset)$ :** $x_3$ agit en **interaction** avec $x_2$. Le gain de 4 n'apparaît que quand les deux sont présents ; Shapley le **partage équitablement** entre eux (propriété de symétrie), d'où 2 chacun.

---

<a id="partie-b"></a>
# Partie B — Questions types pour les parties cachées (avec corrigés)

<a id="q1"></a>
## 8. Q1 (3 pts) — Espace des versions

**Énoncé type.** Hypothèses : conjonctions d'attributs. Taille ∈ {petite, grande}, Couleur ∈ {rouge, bleu, vert}, Forme ∈ {rond, carré}. « ? » signifie « n'importe quelle valeur ».

1. (petite, rouge, rond) : **+**
2. (grande, rouge, rond) : **+**
3. (petite, bleu, carré) : **−**

a) Donner S et G après chaque exemple (algorithme d'élimination des candidats).
b) Comment classer (grande, rouge, carré) ? Et (petite, vert, carré) ?

**Corrigé.**

| après l'exemple | S | G |
|---|---|---|
| départ | {∅} | {(?, ?, ?)} |
| 1 (+) | {(petite, rouge, rond)} | {(?, ?, ?)} |
| 2 (+) | {(?, rouge, rond)} | {(?, ?, ?)} |
| 3 (−) | {(?, rouge, rond)} | {(?, rouge, ?), (?, ?, rond)} |

- Exemple positif : on **généralise S** au minimum pour couvrir l'exemple.
- Exemple négatif : on **spécialise G** au minimum pour exclure l'exemple, en ne gardant que les spécialisations plus générales que S. (grande, ?, ?) et (?, vert, ?) excluent bien l'exemple 3, mais sont rejetées car elles ne couvrent pas S.

b)
- (grande, rouge, carré) est **ambigu** : accepté par (?, rouge, ?) mais refusé par S. Les hypothèses de l'espace des versions ne sont pas d'accord.
- (petite, vert, carré) est **sûrement négatif** : refusé par tous les éléments de G, donc par toutes les hypothèses de l'espace des versions.

> 💡 Règle de classement : accepté par tout S ⇒ sûrement positif ; refusé par tout G ⇒ sûrement négatif ; sinon ambigu.

<a id="q2q3"></a>
## 9. Q2 et Q3 (2 pts) — Questions de cours

### 9.1 Compromis biais-variance, bagging et boosting

**Question.** Expliquez le compromis biais-variance. Comment le bagging et le boosting agissent-ils dessus ?

**Corrigé.** L'erreur de généralisation se décompose (en régression avec perte quadratique) en :

$$\text{Erreur} = \text{Biais}^2 + \text{Variance} + \text{Bruit}$$

- **Biais** : erreur due à un modèle trop simple, incapable de représenter la vraie fonction. Associé au sous-apprentissage.
- **Variance** : sensibilité du modèle à l'échantillon d'apprentissage. Associée au sur-apprentissage.
- **Bruit** : erreur irréductible des données.

Un modèle complexe a un biais faible mais une forte variance ; un modèle simple, l'inverse. Le compromis consiste à choisir la complexité qui minimise la somme.

- **Bagging** : on apprend des modèles sur des échantillons bootstrap et on moyenne (ou on vote). Appliqué à des modèles à faible biais et forte variance (arbres profonds), il **réduit la variance** sans augmenter le biais. Les forêts aléatoires ajoutent un tirage aléatoire des descripteurs pour décorréler les arbres.
- **Boosting** : on apprend séquentiellement des apprenants faibles (souches), en repondérant les exemples mal classés. Il **réduit surtout le biais**. Il peut sur-apprendre si les données sont bruitées, car il insiste sur les exemples difficiles, qui peuvent être du bruit.

### 9.2 Validation croisée

**Question.** Pourquoi utilise-t-on la validation croisée ? Décrivez la validation croisée à k plis et son intérêt par rapport à un simple découpage apprentissage/test.

**Corrigé.** Elle sert à **estimer l'erreur de généralisation** quand les données sont peu nombreuses, et à **choisir les hyperparamètres** sans toucher à l'ensemble de test.

Procédure à k plis :

1. Découper les données en $k$ blocs de taille égale.
2. Pour chaque bloc : apprendre sur les $k - 1$ autres, tester sur ce bloc.
3. Moyenner les $k$ erreurs (et calculer leur écart-type).

Intérêts : chaque exemple sert une fois au test et $k - 1$ fois à l'apprentissage ; l'estimation dépend moins d'un découpage particulier ; on obtient une idée de sa variance. Inconvénient : coûte $k$ apprentissages. On la **stratifie** pour garder les proportions de classes dans chaque bloc (voir la question 4).

<a id="qcourtes"></a>
## 10. Q8 à Q14 (1 pt) — Questions courtes

### 10.1 Sur-apprentissage

**Question.** Qu'est-ce que le sur-apprentissage ? Comment le détecte-t-on sur des courbes d'apprentissage ?

**Corrigé.** Le modèle colle aux données d'apprentissage, y compris au bruit, et généralise mal : erreur d'apprentissage faible, erreur de test élevée. Sur les courbes en fonction de la complexité (ou du nombre d'itérations), l'erreur d'apprentissage continue de baisser tandis que l'erreur de validation se met à **remonter** : l'écart se creuse.

### 10.2 Normalisation et k-PPV

**Question.** Pourquoi faut-il normaliser les descripteurs avant les k plus proches voisins ?

**Corrigé.** Les k-PPV reposent sur une distance (souvent euclidienne). Sans normalisation, les descripteurs à grande échelle dominent la distance : un revenu en euros (milliers) écrase un âge en années (dizaines). On centre-réduit ou on ramène sur [0, 1] pour que chaque descripteur pèse de façon comparable.

### 10.3 Erreur d'apprentissage du 1-PPV

**Question.** Quel est le taux d'erreur en apprentissage d'un 1-PPV ? Pourquoi n'est-il pas informatif ?

**Corrigé.** Il est **nul** : chaque exemple est son propre plus proche voisin (sauf doublons de classes différentes). Il mesure la mémorisation et non la généralisation ; il faut évaluer sur des données non vues.

### 10.4 Exactitude sur données déséquilibrées

**Question.** Sur un jeu avec 99 % de négatifs, pourquoi l'exactitude est-elle une mauvaise mesure ?

**Corrigé.** Un classifieur qui prédit toujours « négatif » obtient 99 % d'exactitude alors qu'il ne détecte aucun positif (rappel = 0). Il faut utiliser la matrice de confusion, le rappel, la précision, la F-mesure, l'AUC ou l'exactitude équilibrée.

### 10.5 Biais inductif

**Question.** Qu'est-ce que le biais inductif ? Donnez celui d'un arbre de décision ou d'un modèle linéaire.

**Corrigé.** C'est l'ensemble des hypothèses a priori qui permettent à un algorithme de généraliser au-delà des exemples vus. Sans biais, aucune généralisation n'est possible. On distingue le biais de langage (l'espace $\mathcal{H}$ choisi) et le biais de recherche (les préférences dans $\mathcal{H}$).

- Arbre de décision (ID3) : préférence pour les **arbres courts**, avec les descripteurs à fort gain près de la racine ; frontières parallèles aux axes.
- Modèle linéaire : la frontière de décision est un **hyperplan**.

### 10.6 Perceptron et XOR

**Question.** Pourquoi un perceptron ne peut-il pas apprendre le XOR ?

**Corrigé.** Un perceptron sépare les classes par une droite (un hyperplan). Or le XOR n'est **pas linéairement séparable** : (0,0) et (1,1) sont de classe 0, (0,1) et (1,0) de classe 1, et aucune droite ne sépare ces deux paires. Solutions : ajouter une couche cachée (perceptron multicouche) ou un descripteur non linéaire comme $x_1 x_2$.

<a id="q16"></a>
## 11. Q16 (1 pt) — Causalité

### 11.1 Facteur confondant

**Question.** Une étude montre que les buveurs de café ont plus de maladies cardiaques. Peut-on conclure que le café en est la cause ? Citez un facteur confondant possible.

**Corrigé.** Non : **corrélation n'est pas causalité**. Un facteur confondant possible est le **tabac** : les fumeurs boivent souvent plus de café, et le tabac cause des maladies cardiaques. Il crée une corrélation entre café et maladie sans lien causal direct. Pour conclure, il faudrait une expérience randomisée, ou au moins ajuster sur les facteurs confondants.

### 11.2 Paradoxe de Simpson

**Question.** Qu'est-ce que le paradoxe de Simpson ?

**Corrigé.** Une tendance présente dans **chaque sous-groupe** s'inverse quand on **agrège** les données, à cause d'un facteur confondant inégalement réparti.

Exemple classique des calculs rénaux :

| | petits calculs | gros calculs | total |
|---|---|---|---|
| Traitement A | 81/87 (93 %) | 192/263 (73 %) | 273/350 (78 %) |
| Traitement B | 234/270 (87 %) | 55/80 (69 %) | 289/350 (83 %) |

A est meilleur dans chaque groupe, mais B semble meilleur au total, car A a surtout été donné pour les gros calculs, plus difficiles à traiter.

<a id="q18"></a>
## 12. Q18 (1 pt) — Explicabilité et éthique

### 12.1 Explication locale et globale

**Question.** Quelle est la différence entre une explication locale et une explication globale d'un modèle ?

**Corrigé.**
- **Locale** : explique **une prédiction particulière** (pourquoi ce patient est classé à risque). Exemples : LIME, valeurs de Shapley pour un exemple (question 17).
- **Globale** : explique **le comportement général** du modèle sur l'ensemble des données. Exemples : importance moyenne des descripteurs, arbre de substitution qui imite le modèle.

### 12.2 Discrimination par variables proxy

**Question.** Pourquoi un modèle peut-il être discriminant même si l'attribut sensible (le sexe, par exemple) a été retiré des données ?

**Corrigé.** D'autres descripteurs sont **corrélés** à l'attribut sensible (code postal, prénom, type d'emploi…) : ce sont des **variables proxy**. Le modèle peut reconstruire indirectement l'attribut retiré et reproduire les biais présents dans les données. Retirer l'attribut (« équité par ignorance ») ne suffit donc pas ; il faut mesurer l'équité des prédictions par groupe.

---

<a id="pieges"></a>
# 13. Pièges à éviter

- ⚠️ **Question 15 :** oublier d'appliquer les modifications des exemples 1 et 9. L'entropie initiale ne change pas (toujours 8 true / 9 false), ce qui rend l'erreur invisible à la question (a), mais les sous-ensembles de texture et umbilicus seraient faux.
- ⚠️ **Question 15 :** oublier de **pondérer** les entropies des sous-ensembles par leur taille $\frac{|D_v|}{|D|}$.
- ⚠️ **Question 15 :** le meilleur descripteur est celui dont l'entropie *après* test est la **plus faible** (gain le plus grand), pas la plus forte.
- ⚠️ **Question 17 :** une contribution nulle en isolation ($v(\{i\}) = v(\emptyset)$) ne signifie pas $\varphi_i = 0$ : penser aux interactions.
- ⚠️ **Question 17 :** toujours vérifier $\sum_i \varphi_i = f(x) - f(\text{baseline})$, c'est une vérification gratuite.
- ⚠️ **Question 5 :** ne pas inverser les axes : abscisse = taux de **faux** positifs, ordonnée = taux de **vrais** positifs.
- ⚠️ **Question 6 :** précision = dénominateur « ce que j'ai prédit + » ; rappel = dénominateur « ce qui est vraiment + ».
- ⚠️ **Question 4 :** la réponse n'est pas « 50 % » : le découpage rend la classe majoritaire en apprentissage minoritaire en test.
- ⚠️ **Espace des versions :** une spécialisation de G n'est gardée que si elle reste plus générale qu'un élément de S.
