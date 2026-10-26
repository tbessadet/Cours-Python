# 🎯 Cheat Sheet Sélection de modèle

> Objectif : comprendre **comment** chaque modèle classe/prédit, **comment régler** ses paramètres clés, et **quand** le préférer à un autre — à garder sous la main pendant les TD.
> Convention : `import numpy as np`, imports scikit-learn précisés modèle par modèle.

## Sommaire

**I. [Grille de lecture rapide](#i-grille-de-lecture-rapide)**

**II. Modèles de classification**
1. [k plus proches voisins (k-NN)](#ii1-k-plus-proches-voisins-k-nn)
2. [Naive Bayes (gaussien, Bernoulli, multinomial)](#ii2-naive-bayes-gaussien-bernoulli-multinomial)
3. [Analyse discriminante (LDA / QDA)](#ii3-analyse-discriminante-lda--qda)
4. [Perceptron](#ii4-perceptron)
5. [Régression logistique](#ii5-régression-logistique)
6. [SVM (linéaire et à noyau)](#ii6-svm-linéaire-et-à-noyau)
7. [Arbre de décision](#ii7-arbre-de-décision)
8. [Forêt aléatoire](#ii8-forêt-aléatoire)
9. [Gradient Boosting / XGBoost](#ii9-gradient-boosting--xgboost)
10. [Réseau de neurones (MLP)](#ii10-réseau-de-neurones-mlp)

**III. Modèles de régression**
1. [Régression linéaire (moindres carrés)](#iii1-régression-linéaire-moindres-carrés)
2. [Régularisation : Ridge / Lasso / Elastic Net](#iii2-régularisation--ridge--lasso--elastic-net)
3. [Régression polynomiale](#iii3-régression-polynomiale)
4. [Arbres, forêts et XGBoost en régression](#iii4-arbres-forêts-et-xgboost-en-régression)

**IV. Modèles de clustering (non supervisé)**
1. [K-means](#iv1-k-means)
2. [Clustering hiérarchique](#iv2-clustering-hiérarchique)
3. [DBSCAN](#iv3-dbscan)

**V. [Sélection et comparaison automatique de modèles](#v-sélection-et-comparaison-automatique-de-modèles)**

[Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Grille de lecture rapide

À consulter en premier pendant un TD pour orienter rapidement son choix, avant de creuser la fiche du modèle retenu.

| Modèle | Type | Principe en une ligne | Interprétable ? | Passe à l'échelle ? | À privilégier quand... |
|---|---|---|---|---|---|
| Perceptron | Classification linéaire | Séparateur linéaire, mise à jour en ligne | Oui | Oui | Données linéairement séparables, apprentissage en flux |
| Régression logistique | Classification linéaire probabiliste | Frontière linéaire + probabilité (sigmoïde) | Oui | Oui | Baseline solide, besoin de probabilités calibrées et d'interprétation |
| SVM linéaire | Classification linéaire | Marge maximale entre classes | Moyenne | Oui | Données de grande dimension, bien séparées |
| SVM à noyau | Classification non-linéaire | Marge maximale dans un espace transformé | Faible | Non (données moyennes) | Frontière non-linéaire, jeu de données de taille modérée |
| Naive Bayes | Classification probabiliste | Indépendance des variables + Bayes | Oui | Oui | Beaucoup de variables, texte, baseline très rapide |
| LDA / QDA | Classification probabiliste/générative | Gaussiennes par classe (covariance commune/propre) | Oui | Oui | Variables continues, hypothèses gaussiennes raisonnables |
| k-NN | Classification/régression | Vote des `k` voisins les plus proches | Oui (visuellement) | Non (lent en prédiction) | Peu de données, frontière très irrégulière, baseline rapide |
| Arbre de décision | Classification/régression | Règles de seuils successives | Très forte | Oui | Besoin d'explicabilité, variables mixtes (num+cat) |
| MLP (réseau de neurones) | Classification/régression non-linéaire | Couches de neurones, non-linéarités apprises | Très faible | Oui (avec GPU) | Beaucoup de données, relations complexes, images/signaux |
| Régression linéaire | Régression | Combinaison linéaire des variables | Très forte | Oui | Relation linéaire supposée, besoin d'interprétation |
| Ridge / Lasso / ElasticNet | Régression régularisée | Régression linéaire + pénalité sur les poids | Forte | Oui | Multicolinéarité, trop de variables, besoin de sélection (Lasso) |
| Forêt aléatoire | Ensemble d'arbres | Vote/moyenne de nombreux arbres décorrélés | Faible (mais `feature_importances_`) | Oui | Bonne performance "out of the box", peu de réglage |
| Gradient Boosting / XGBoost | Ensemble d'arbres | Arbres construits pour corriger les erreurs précédentes | Faible | Oui | Données tabulaires, on veut l'état de l'art |
| K-means | Clustering | Partition en `K` groupes autour de barycentres | Moyenne | Oui | Clusters convexes de taille comparable, `K` approximativement connu |
| Clustering hiérarchique | Clustering | Fusions successives, visualisable en arbre | Forte (dendrogramme) | Non (grands jeux) | Petit jeu de données, on veut explorer plusieurs `K` d'un coup |
| DBSCAN | Clustering | Zones denses séparées par du vide | Moyenne | Moyenne | Clusters de forme arbitraire, présence de bruit/anomalies |

---

# II. Modèles de classification

## II.1. k plus proches voisins (k-NN)

**Principe** : pour classer un nouveau point, on regarde ses `k` voisins les plus proches (au sens d'une distance, souvent euclidienne) dans les données d'apprentissage, et on prend la classe **majoritaire** parmi eux. Aucun apprentissage à proprement parler : tout le calcul a lieu au moment de la prédiction ("lazy learning").

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `KNeighborsClassifier(n_neighbors=, weights=, metric=)` | `n_neighbors` (`k`) = nombre de voisins consultés ; `weights` = `'uniform'` (vote égal) ou `'distance'` (les voisins proches pèsent plus) ; `metric` = `'euclidean'` (déf.), `'manhattan'`, `'minkowski'` | Import : `sklearn.neighbors`. |

**Comment régler `k`** : `k` petit (1-3) → frontière très irrégulière, forte variance, sensible au bruit (sur-apprentissage). `k` grand → frontière lisse, biais plus fort, risque de sous-apprentissage (peut même ignorer des petites classes). Se règle **par validation croisée**, en balayant plusieurs valeurs de `k` et en traçant la courbe d'erreur (voir §V.1).

**Avantages** : simple, aucune hypothèse sur la forme des données, s'adapte à des frontières complexes.
**Limites** : coûteux en prédiction sur de grands jeux de données (distance à tous les points d'apprentissage) ; **très sensible à l'échelle des variables** → normalisation quasi obligatoire ; performance dégradée en grande dimension (fléau de la dimensionnalité, les distances perdent leur sens discriminant).

👉 **À privilégier** : jeu de données de taille raisonnable, frontière de décision très irrégulière, bonne baseline rapide à mettre en place.

---

## II.2. Naive Bayes (gaussien, Bernoulli, multinomial)

**Principe** : modèle **génératif** — on estime la loi de probabilité de chaque variable *à l'intérieur de chaque classe* (en supposant les variables **indépendantes** entre elles), puis on classe un nouveau point dans la classe qui **maximise sa vraisemblance** (éventuellement pondérée par la fréquence a priori de chaque classe, cf. maximum a posteriori). Voir la cheat sheet *Classification probabiliste* pour l'implémentation "à la main".

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `GaussianNB()` | `var_smoothing` (évite les variances nulles) | Variables **continues**, supposées gaussiennes par classe. Import : `sklearn.naive_bayes`. |
| `BernoulliNB(alpha=)` | `alpha` = lissage de Laplace (évite les probabilités 0/1 exactes) | Variables **binaires** (présent/absent). |
| `MultinomialNB(alpha=)` | `alpha` idem | Variables de **comptage** (ex. fréquence de mots dans un texte). |

**Comment régler les paramètres** : `alpha`/`var_smoothing` se règlent rarement finement — leur seul rôle est d'éviter les probabilités exactement nulles (division par zéro / `log(0)`), une petite valeur par défaut suffit dans la majorité des cas.

**Avantages** : extrêmement rapide à l'entraînement et à la prédiction, fonctionne bien même avec peu de données, robuste en grande dimension (texte, bag-of-words).
**Limites** : l'hypothèse d'indépendance des variables est presque toujours fausse en pratique (d'où le nom "naive") — la performance peut en pâtir si les variables sont fortement corrélées.

👉 **À privilégier** : classification de texte, baseline très rapide, grande dimension avec peu de données, quand on a besoin d'un modèle simple à comprendre et à déployer.

---

## II.3. Analyse discriminante (LDA / QDA)

**Principe** : comme Naive Bayes, un modèle **génératif** gaussien par classe — mais sans l'hypothèse d'indépendance des variables : chaque classe est modélisée par une gaussienne **multivariée** complète (avec covariances entre variables). La **LDA** (Linear Discriminant Analysis) suppose une matrice de covariance **commune** à toutes les classes (→ frontière linéaire) ; la **QDA** (Quadratic) laisse une covariance **propre à chaque classe** (→ frontière quadratique, plus flexible).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `LinearDiscriminantAnalysis(solver=)` | `solver` = `'svd'` (déf., stable), `'lsqr'`, `'eigen'` (permet aussi la réduction de dimension via `.transform()`) | Import : `sklearn.discriminant_analysis`. |
| `QuadraticDiscriminantAnalysis(reg_param=)` | `reg_param` = régularisation (évite le sur-apprentissage si peu de données par classe) | Plus flexible que LDA mais plus de paramètres à estimer. |

**Comment choisir entre les deux** : LDA si peu de données par classe (moins de paramètres à estimer, plus stable) ou si les classes ont vraiment une dispersion comparable ; QDA si les classes ont des formes/dispersions clairement différentes et qu'on a assez de données pour estimer une covariance par classe de façon fiable.

**Avantages** : rapide, solution analytique (pas d'optimisation itérative), donne aussi des scores probabilistes, la LDA peut servir de méthode de **réduction de dimension supervisée**.
**Limites** : hypothèse gaussienne à vérifier, QDA peut sur-apprendre si peu de données par classe (trop de paramètres de covariance à estimer).

👉 **À privilégier** : variables continues, hypothèse gaussienne plausible, besoin d'un modèle probabiliste rapide et interprétable — bonne alternative à la régression logistique quand les classes sont bien séparées.

---

## II.4. Perceptron

**Principe** : le plus simple des classifieurs linéaires — trouve une droite (ou hyperplan) séparant les classes, mise à jour **en ligne** point par point : à chaque erreur de classification, on ajuste le poids dans la direction qui corrige cette erreur.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `Perceptron(alpha=, max_iter=, eta0=)` | `alpha` = régularisation, `max_iter` = nb de passages sur les données, `eta0` = pas d'apprentissage initial | Import : `sklearn.linear_model`. |

**Comment régler** : `max_iter` doit être assez grand pour converger si les données sont séparables (sinon l'algorithme ne s'arrête jamais de "corriger" → limiter via `max_iter`/`tol`). Peu utilisé avec un réglage fin en pratique : c'est avant tout un modèle **pédagogique**, souvent supplanté par la régression logistique ou le SVM linéaire qui apportent en plus une notion de marge/probabilité.

**Avantages** : très simple, rapide, apprentissage possible en flux (nouvelles données une par une).
**Limites** : ne converge **que** si les données sont linéairement séparables (sinon oscille indéfiniment) ; ne fournit pas de probabilité ; pas de notion de marge (contrairement au SVM), donc frontière pas nécessairement "la meilleure possible".

👉 **À privilégier** : cas pédagogique, données linéairement séparables, apprentissage incrémental en flux. En pratique, préférer la régression logistique ou le SVM linéaire.

---

## II.5. Régression logistique

**Principe** : modèle linéaire discriminant qui, contrairement au perceptron, prédit une **probabilité** d'appartenance à une classe via une fonction sigmoïde appliquée à une combinaison linéaire des variables. La frontière de décision reste linéaire.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `LogisticRegression(C=, penalty=, solver=, max_iter=)` | `C` = **inverse** de la force de régularisation (`C` petit = régularisation forte) ; `penalty` = `'l2'` (déf.), `'l1'` (sélection de variables), `'elasticnet'` ; `solver` doit être compatible avec `penalty` (ex. `'liblinear'` ou `'saga'` pour `'l1'`) | Import : `sklearn.linear_model`. |

**Comment régler `C`** : balayer plusieurs valeurs en échelle **logarithmique** (`[0.001, 0.01, 0.1, 1, 10, 100]`) via validation croisée — `C` trop grand → sur-apprentissage (poids non contraints) ; `C` trop petit → sous-apprentissage (poids écrasés vers 0).

**Avantages** : rapide, coefficients interprétables (`.coef_`, transformables en odds ratios via `exp()`), probabilités bien calibrées, bonne baseline quasi systématique.
**Limites** : frontière strictement linéaire (sauf à enrichir les variables manuellement), performance limitée si la vraie frontière est fortement non-linéaire.

👉 **À privilégier** : quasiment toujours comme **première baseline** en classification, quand on a besoin de probabilités et/ou d'interprétabilité des coefficients.

---

## II.6. SVM (linéaire et à noyau)

**Principe** : cherche l'hyperplan séparateur qui **maximise la marge** (distance aux points les plus proches de chaque classe, les "vecteurs de support"). Avec un **noyau** (kernel trick), les données sont implicitement projetées dans un espace de plus grande dimension où elles deviennent linéairement séparables, sans jamais calculer explicitement cette projection.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `SVC(C=, kernel=, gamma=, degree=)` | `C` = tolérance aux erreurs (grand `C` = marge stricte, risque de sur-apprentissage ; petit `C` = marge souple, plus de tolérance) ; `kernel` = `'linear'`, `'rbf'` (gaussien, déf.), `'poly'` ; `gamma` = "portée" du noyau RBF (grand `gamma` = influence très locale de chaque point, risque de sur-apprentissage ; petit `gamma` = influence large, frontière plus lisse) ; `degree` = degré du noyau polynomial | Import : `sklearn.svm`. `LinearSVC` = version optimisée pour `kernel='linear'` sur beaucoup de données. |

**Comment régler `C` et `gamma`** : les deux se règlent **ensemble** par grid search (ils interagissent), en balayant chacun sur une échelle logarithmique. Un bon réflexe : d'abord fixer un `gamma` raisonnable (ex. `'scale'`, la valeur par défaut, qui s'adapte à la variance des données), puis affiner `C`.

**Avantages** : efficace en grande dimension, marge maximale = bonne généralisation théorique, le noyau RBF capture des frontières très non-linéaires.
**Limites** : coûteux en temps de calcul sur de grands jeux de données (plusieurs dizaines de milliers de points et plus) ; réglage de `C`/`gamma` sensible ; peu interprétable avec un noyau non-linéaire ; sensible à l'échelle des variables (normalisation nécessaire).

👉 **À privilégier** : jeu de données de taille modérée, frontière non-linéaire complexe (noyau RBF), ou grande dimension avec frontière linéaire plausible (`LinearSVC`, ex. texte).

### `LinearSVC()`

**Principe** : version spécialisée du SVM pour une **frontière linéaire**, sans noyau. Cherche l'hyperplan séparateur qui maximise la marge tout en pénalisant les erreurs de classification. Contrairement à `SVC(kernel='linear')`, `LinearSVC` est optimisé pour être plus rapide sur les jeux de données comportant beaucoup d'observations et/ou de variables.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `LinearSVC(C=, loss=, penalty=)` | `C` = compromis entre largeur de la marge et pénalisation des erreurs (grand `C` = erreurs fortement pénalisées, marge plus stricte, risque de sur-apprentissage ; petit `C` = marge plus souple, davantage de tolérance) ; `loss` = fonction de perte, `'squared_hinge'` par défaut ; `penalty` = type de régularisation, `'l2'` par défaut | Import : `from sklearn.svm import LinearSVC`. Contrairement à `SVC`, pas de paramètre `kernel` : la frontière est nécessairement linéaire. `LinearSVC` est particulièrement adapté aux jeux de données de grande dimension. |

**Comment régler `C`** : régler principalement `C` par grid search sur une **échelle logarithmique**. Un petit `C` augmente la régularisation et produit une marge plus souple ; un grand `C` cherche davantage à classer correctement les observations d'entraînement, avec un risque accru de sur-apprentissage.

**Avantages** : rapide et peu coûteux en mémoire, efficace en grande dimension, particulièrement adapté aux données comportant beaucoup de variables (ex. texte), marge maximale = bonne généralisation théorique.

**Limites** : ne peut apprendre qu'une **frontière linéaire**, pas de noyau RBF ou polynomial ; sensible à l'échelle des variables (normalisation nécessaire) ; les véritables vecteurs de support ne sont pas directement accessibles via un attribut `support_` contrairement à `SVC`.

👉 **À privilégier** : jeu de données de grande taille ou de grande dimension lorsque l'on suppose qu'une **frontière linéaire** est suffisante.

---

## II.7. Arbre de décision

**Principe** : construit récursivement une séquence de règles de type "si variable X < seuil, alors...", en choisissant à chaque étape la coupure qui réduit le plus l'impureté des groupes résultants (indice de Gini ou entropie).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `DecisionTreeClassifier(max_depth=, min_samples_leaf=, criterion=)` | `max_depth` = profondeur maximale (limite le sur-apprentissage) ; `min_samples_leaf` = nb minimal d'individus par feuille (idem) ; `criterion` = `'gini'` (déf.) ou `'entropy'` | Import : `sklearn.tree`. `plot_tree(model)` pour visualiser l'arbre. |

**Comment régler `max_depth`** : arbre **trop profond** → sur-apprentissage quasi garanti (mémorise le bruit, une feuille par exemple à la limite) ; arbre **trop peu profond** → sous-apprentissage (règles trop grossières). Se règle par validation croisée, en traçant la courbe de performance train/test en fonction de la profondeur — le point où les deux courbes divergent indique le début du sur-apprentissage.

**Avantages** : **le plus explicable** de tous les modèles (règles directement lisibles), gère nativement les variables catégorielles et numériques mélangées, aucune normalisation nécessaire (les seuils sont invariants par changement d'échelle monotone).
**Limites** : très instable (un arbre seul change beaucoup si on modifie légèrement les données) — d'où l'intérêt des méthodes d'ensemble (§II.8, II.9) qui moyennent plusieurs arbres pour stabiliser la prédiction.

👉 **À privilégier** : besoin d'explicabilité forte (règles métier), variables mixtes, comme brique de base des forêts/boosting.

---

## II.8. Forêt aléatoire

**Principe** : entraîne un grand nombre d'arbres de décision, chacun sur un **échantillon aléatoire** de données (bootstrap) et un **sous-ensemble aléatoire** de variables à chaque coupure, puis **vote** (classification) ou **moyenne** (régression) sur l'ensemble des arbres. La décorrélation entre arbres réduit fortement la variance par rapport à un arbre seul.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `RandomForestClassifier(n_estimators=, max_depth=, max_features=, n_jobs=)` | `n_estimators` = nombre d'arbres (plus = mieux, jusqu'à un plateau de performance, coût de calcul croissant) ; `max_depth` = profondeur de chaque arbre (souvent laissé libre, l'aspect ensembliste limite déjà le sur-apprentissage) ; `max_features` = nb de variables testées à chaque coupure (`'sqrt'` = classification, déf.) ; `n_jobs=-1` = parallélisation | Import : `sklearn.ensemble`. |

**Comment régler `n_estimators`** : augmenter jusqu'à ce que la performance en validation croisée se stabilise (rendements décroissants, pas de sur-apprentissage supplémentaire lié à ce paramètre contrairement aux arbres seuls). `max_depth`/`min_samples_leaf` peuvent être resserrés si malgré tout du sur-apprentissage est observé.

**Avantages** : excellente performance "out of the box" avec peu de réglage, robuste au bruit et aux valeurs aberrantes, fournit une mesure d'importance des variables (`.feature_importances_`), peu sensible au sur-apprentissage grâce à l'agrégation.
**Limites** : moins interprétable qu'un arbre seul, plus lente à l'entraînement et à la prédiction qu'un modèle linéaire, modèle volumineux en mémoire.

👉 **À privilégier** : quasiment toujours une **valeur sûre** sur données tabulaires, quand on veut de bonnes performances rapidement sans réglage fin poussé.

---

## II.9. Gradient Boosting / XGBoost

**Principe** : comme la forêt aléatoire, un ensemble d'arbres — mais construits **séquentiellement** plutôt qu'indépendamment : chaque nouvel arbre est entraîné pour corriger les erreurs (résidus) laissées par les arbres précédents. Cette construction itérative permet en général d'atteindre une performance supérieure à la forêt aléatoire, au prix d'un réglage plus délicat.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `xgb.XGBClassifier(n_estimators=, max_depth=, learning_rate=, subsample=)` | `n_estimators` = nombre d'arbres séquentiels ; `max_depth` = profondeur de chaque arbre (souvent **faible**, 3-8, contrairement aux forêts) ; `learning_rate` = poids de contribution de chaque nouvel arbre (petit = apprentissage plus lent mais plus stable) ; `subsample` = fraction des données utilisées par arbre (< 1 = régularisation supplémentaire) | Import `xgboost`. Interface `fit`/`predict` identique à scikit-learn. |

**Comment régler ensemble `n_estimators` et `learning_rate`** : ils s'équilibrent — `learning_rate` petit nécessite **plus** d'arbres (`n_estimators` grand) pour converger, et inversement. Stratégie courante : fixer un `learning_rate` modéré (0.05-0.1), puis choisir `n_estimators` par **early stopping** (arrêter dès que la performance sur un jeu de validation cesse de s'améliorer, plutôt que de fixer une valeur a priori).

**Avantages** : très souvent **l'état de l'art** sur données tabulaires (compétitions Kaggle, etc.), gère nativement les valeurs manquantes, très flexible (nombreux hyperparamètres de régularisation).
**Limites** : plus de hyperparamètres à régler que la forêt aléatoire (risque de sur-apprentissage si mal réglé), plus sensible aux données bruitées que la forêt (chaque arbre corrige les erreurs, y compris le bruit, des précédents), temps de réglage plus long.

👉 **À privilégier** : données tabulaires, quand on cherche la meilleure performance possible et qu'on est prêt à investir du temps de réglage (grid search / Optuna).

---

## II.10. Réseau de neurones (MLP)

**Principe** : empile plusieurs couches de neurones (combinaisons linéaires + non-linéarité), chaque couche apprenant une représentation de plus en plus abstraite des données. Entraîné par descente de gradient (rétropropagation) — voir la cheat sheet *Régression & descente de gradient* pour les bases de l'optimisation.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `MLPClassifier(hidden_layer_sizes=, activation=, alpha=, learning_rate_init=)` | `hidden_layer_sizes` = tuple donnant le nombre de neurones par couche cachée (ex. `(100,)`, `(64,32)`) ; `activation` = `'relu'` (déf., quasi toujours un bon choix), `'tanh'` ; `alpha` = régularisation L2 ; `learning_rate_init` = pas d'apprentissage | Import : `sklearn.neural_network`. Pour des architectures plus riches (CNN, RNN...), voir PyTorch. |

**Comment régler l'architecture** : commencer **simple** (1-2 couches cachées) et complexifier seulement si nécessaire — un MLP trop profond sur peu de données sur-apprend très vite. `learning_rate_init` se règle comme pour toute descente de gradient (voir les pièges de la cheat sheet dédiée : trop grand = divergence, trop petit = convergence lente).

**Avantages** : capture des relations très complexes et non-linéaires, seul type de modèle réellement adapté aux données non-tabulaires "brutes" (images, texte, son) via des architectures spécialisées (CNN, transformers...).
**Limites** : nécessite **beaucoup** de données pour bien généraliser, quasiment aucune interprétabilité, sensible au réglage (architecture, learning rate, normalisation des entrées quasi obligatoire), coûteux à entraîner.

👉 **À privilégier** : gros volumes de données, relations fortement non-linéaires, données non-tabulaires (avec les architectures adaptées). Sur données tabulaires de taille modeste, XGBoost/forêt aléatoire sont en général plus performants et plus simples à régler.

---

# III. Modèles de régression

## III.1. Régression linéaire (moindres carrés)

**Principe** : trouve les coefficients qui minimisent la somme des carrés des écarts entre valeurs observées et prédites — solution **analytique** exacte (voir cheat sheet *Régression & descente de gradient*, §2), pas d'optimisation itérative nécessaire pour le cas linéaire simple.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `LinearRegression()` | pas d'hyperparamètre à régler | Import : `sklearn.linear_model`. `.coef_`, `.intercept_` pour l'interprétation. |

**Avantages** : rapide, coefficients directement interprétables, aucun réglage nécessaire.
**Limites** : suppose une relation linéaire, sensible à la multicolinéarité (variables très corrélées entre elles → coefficients instables) et aux valeurs aberrantes.

👉 **À privilégier** : relation linéaire plausible, besoin d'interprétation directe des coefficients, comme baseline systématique en régression.

---

## III.2. Régularisation : Ridge / Lasso / Elastic Net

**Principe** : ajoute une pénalité sur l'amplitude des coefficients à la fonction de coût des moindres carrés, pour limiter le sur-apprentissage — voir la cheat sheet *Scikit-learn — Prétraitement* §4 pour le détail mathématique et la notion de parcimonie.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `Ridge(alpha=)` | `alpha` = force de régularisation L2 | Réduit les poids sans jamais les annuler — stabilise en cas de multicolinéarité. |
| `Lasso(alpha=)` | `alpha` idem | Annule exactement certains poids → **sélection de variables** automatique. |
| `ElasticNet(alpha=, l1_ratio=)` | `l1_ratio` = proportion de pénalité L1 vs L2 | Compromis entre les deux, utile si plusieurs variables corrélées doivent être sélectionnées **ensemble** (Lasso seul n'en garde qu'une arbitrairement parmi un groupe corrélé). |

**Comment régler `alpha`** : balayer en échelle logarithmique par validation croisée (`RidgeCV`, `LassoCV` automatisent directement ce balayage). `alpha=0` = régression linéaire classique ; `alpha` très grand = tous les poids écrasés vers 0.

👉 **À privilégier** : Ridge si multicolinéarité sans besoin de sélection ; Lasso si on veut identifier un sous-ensemble restreint de variables pertinentes ; ElasticNet si plusieurs variables corrélées doivent survivre ensemble à la sélection.

---

## III.3. Régression polynomiale

**Principe** : pas un modèle à part entière, mais un **enrichissement des variables** (ajout de puissances $x^2, x^3,...$) avant d'appliquer une régression linéaire classique — voir cheat sheet *Régression & descente de gradient* §4.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `PolynomialFeatures(degree=)` | `degree` = degré maximal des termes ajoutés | Import : `sklearn.preprocessing`. À chaîner avec `LinearRegression` dans une `Pipeline`. |

**Comment régler `degree`** : degré trop faible → sous-apprentissage (courbe pas assez flexible) ; degré trop élevé → sur-apprentissage spectaculaire, en particulier aux bords de l'intervalle des données (oscillations). Se règle par validation croisée, jamais en ne regardant que l'erreur d'apprentissage.

👉 **À privilégier** : relation clairement non-linéaire mais "simple" (une seule courbure), petite dimension (explose combinatoirement en grande dimension avec les termes croisés).

---

## III.4. Arbres, forêts et XGBoost en régression

Mêmes principes exactement que leurs équivalents en classification (§II.7-II.9), à la différence que chaque feuille d'arbre prédit une **valeur moyenne** plutôt qu'une classe majoritaire.

| Fonction | Explication |
|---|---|
| `DecisionTreeRegressor(max_depth=, min_samples_leaf=)` | Mêmes réglages et compromis biais/variance que `DecisionTreeClassifier`. |
| `RandomForestRegressor(n_estimators=, max_depth=)` | Moyenne des prédictions de chaque arbre plutôt qu'un vote. |
| `xgb.XGBRegressor(n_estimators=, learning_rate=, max_depth=)` | Idem `XGBClassifier`, souvent l'état de l'art sur régression tabulaire également. |

👉 **À privilégier** : relation non-linéaire complexe entre variables et cible, données tabulaires, peu d'hypothèses fortes à poser sur la forme de la relation (contrairement à la régression linéaire/polynomiale).

---

# IV. Modèles de clustering (non supervisé)

> Détail complet (formules, code, sélection du nombre de clusters) dans la cheat sheet *Clustering* dédiée — résumé ici pour la comparaison rapide.

## IV.1. K-means

**Principe** : partitionne les données en `K` groupes en alternant affectation au centre le plus proche et recalcul des centres (barycentres).

| Paramètre clé | Comment le régler |
|---|---|
| `n_clusters` (`K`) | Inconnu a priori en général : balayer plusieurs valeurs et utiliser le score silhouette / méthode du coude sur l'inertie (voir §V et cheat sheet Clustering §6). |

👉 **À privilégier** : clusters à peu près convexes (sphériques) et de taille comparable ; rapide, passe à l'échelle.
**Limites** : suppose des clusters convexes, sensible à l'initialisation et aux valeurs aberrantes, nécessite de fixer `K` à l'avance.

## IV.2. Clustering hiérarchique

**Principe** : fusionne progressivement les clusters les plus proches, produit un dendrogramme exploitable pour choisir `K` *a posteriori*.

| Paramètre clé | Comment le régler |
|---|---|
| `linkage` | `'ward'` = choix par défaut robuste (minimise la variance intra-cluster) ; `'single'` = sensible au bruit (effet de chaîne), à éviter sauf cas particulier. |

👉 **À privilégier** : petit jeu de données, besoin d'explorer visuellement plusieurs découpages possibles (dendrogramme) avant de choisir `K`.
**Limites** : ne passe pas à l'échelle (coût quadratique ou pire en nombre de points).

## IV.3. DBSCAN

**Principe** : regroupe les zones denses de points, sépare les zones creuses ; les points isolés sont étiquetés comme bruit plutôt que forcés dans un cluster.

| Paramètre clé | Comment le régler |
|---|---|
| `eps` | Rayon de voisinage : trop petit → tout en bruit ; trop grand → un seul cluster géant. Balayer plusieurs valeurs et observer la distribution des tailles de clusters obtenues. |
| `min_samples` | Nombre minimal de voisins pour former une zone dense — augmenter réduit la sensibilité au bruit mais peut fragmenter de petits clusters légitimes. |

👉 **À privilégier** : clusters de forme arbitraire (non convexe), présence d'anomalies/bruit à isoler explicitement, nombre de clusters inconnu et non imposé.
**Limites** : sensible au réglage de `eps`, moins efficace si les clusters ont des densités très différentes.

---

# V. Sélection et comparaison automatique de modèles

## 1. Validation croisée et courbes d'apprentissage/validation

| Outil | Import | Usage |
|---|---|---|
| `cross_val_score(mod, X, y, cv=)` | `sklearn.model_selection` | Score moyen et variance d'un modèle sur plusieurs découpages — la base de toute comparaison sérieuse (voir cheat sheet *Scikit-learn — Modèles & Évaluation* §7). |
| `validation_curve(mod, X, y, param_name=, param_range=, cv=)` | idem | Trace performance train/test **en fonction d'un seul hyperparamètre** (ex. `max_depth`) — permet de repérer visuellement le sur/sous-apprentissage et la valeur optimale. |
| `learning_curve(mod, X, y, train_sizes=, cv=)` | idem | Trace performance train/test **en fonction de la quantité de données** — diagnostique si le modèle manque de données (les deux courbes ne se sont pas encore rejointes) ou si ajouter des données ne servirait à rien (courbes déjà stabilisées). |

```python
from sklearn.model_selection import validation_curve

train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(), X, y, param_name="max_depth", param_range=range(1, 20), cv=5
)
plt.plot(range(1, 20), train_scores.mean(axis=1), label="train")
plt.plot(range(1, 20), test_scores.mean(axis=1), label="test")   # l'écart croissant = sur-apprentissage
```

## 2. Recherche d'hyperparamètres

| Outil | Import | Usage |
|---|---|---|
| `GridSearchCV(estimator, param_grid, cv=)` | `sklearn.model_selection` | Exhaustif, teste toutes les combinaisons — coûteux dès que la grille grandit (voir cheat sheet Prétraitement §10). |
| `RandomizedSearchCV(estimator, param_distributions, n_iter=)` | idem | Tire aléatoirement `n_iter` combinaisons plutôt que toutes — bien plus efficace dès que l'espace de recherche est grand (souvent aussi bon résultat pour un coût bien moindre). |
| `optuna` | `import optuna` | Recherche **intelligente** (bayésienne) qui concentre l'exploration sur les zones prometteuses — à privilégier dès que le budget de calcul est limité et l'espace de recherche large (voir cheat sheet Modèles & Évaluation §8). |

## 3. Choisir la bonne métrique selon le problème

| Contexte | Métrique(s) recommandée(s) |
|---|---|
| Classes équilibrées | `accuracy` |
| Classes déséquilibrées | `f1_score` (par classe), `precision`/`recall` ciblés selon le coût métier des faux positifs/négatifs, courbe ROC-AUC ou précision/rappel |
| Régression | `RMSE`/`MSE` (pénalise fort les grosses erreurs), `MAE` (plus robuste aux valeurs aberrantes), `R²` (variance expliquée) |
| Clustering | `silhouette_score`, `davies_bouldin_score` (pas de vérité terrain), `adjusted_rand_score` (si vérité terrain disponible pour évaluation externe) |

👉 **La métrique de sélection doit refléter l'objectif réel** : un modèle avec la meilleure accuracy n'est pas forcément le meilleur choix si le coût d'un faux négatif est très supérieur à celui d'un faux positif (ex. diagnostic médical, détection de fraude) — toujours partir du besoin métier avant de choisir la métrique de comparaison.

## 4. Comparaison statistique de modèles

Un score de validation croisée plus élevé pour le modèle A que pour le modèle B ne garantit pas que A est réellement meilleur — l'écart peut être dû au hasard du découpage.

| Outil | Usage |
|---|---|
| Test t apparié sur les scores de CV | Compare les scores obtenus par deux modèles **sur les mêmes plis** de validation croisée. |
| `paired_ttest_kfold_cv` (librairie `mlxtend`) | Version corrigée du test t, adaptée au fait que les plis de CV ne sont pas complètement indépendants (le test t naïf a tendance à sous-estimer la variance). |

👉 Sur peu de données (<50), l'incertitude statistique est grande — un test formel est presque indispensable. Sur beaucoup de données (>10 000), presque toute différence devient "significative" au sens statistique, même minime en pratique — relativiser l'importance du test face à l'ampleur réelle du gain.

## 5. AutoML (aperçu)

Pour explorer rapidement un grand nombre de modèles/hyperparamètres sans tout coder à la main :

| Outil | Principe |
|---|---|
| `LazyPredict` | Entraîne et compare des dizaines de modèles scikit-learn par défaut en une commande — excellent pour un premier tour d'horizon rapide, jamais pour le modèle final. |
| `TPOT` | Recherche automatique (algorithmes génétiques) d'une pipeline complète (prétraitement + modèle + hyperparamètres). |
| `auto-sklearn` | Recherche automatique combinée à de l'ensembling final des meilleurs modèles trouvés. |

⚠️ Ces outils sont utiles pour **dégrossir** rapidement le choix, mais ne remplacent pas la compréhension du problème (métrique adaptée, prétraitement pertinent, interprétabilité recherchée) — à utiliser en première approche, jamais comme solution finale sans validation critique.

---

## 📌 Récapitulatif des pièges classiques

1. **Comparer des modèles avec des métriques inadaptées au problème** (ex. accuracy sur données très déséquilibrées) fausse complètement le classement — toujours partir du besoin métier pour choisir la métrique (§V.3).
2. **Régler les hyperparamètres en ne regardant que la performance en apprentissage** mène droit au sur-apprentissage — toujours utiliser une validation croisée, jamais le score train seul.
3. **k-NN et SVM sont très sensibles à l'échelle des variables** — normalisation quasi obligatoire, contrairement aux arbres/forêts/XGBoost qui n'en ont pas besoin.
4. **XGBoost et forêt aléatoire ne se réglent pas pareil** : la forêt tolère de nombreux arbres profonds sans sur-apprendre, alors que le gradient boosting sur-apprend si `n_estimators`/`learning_rate` sont mal équilibrés — ne pas transposer les réglages de l'un à l'autre.
5. **Confondre corrélation de variables et sélection de variables** : Lasso élimine arbitrairement une variable parmi un groupe corrélé, ElasticNet est plus adapté si on veut les garder ensemble.
6. **Un score de CV plus élevé ne signifie pas toujours une différence significative** entre deux modèles — sur peu de données en particulier, un test statistique dédié est nécessaire avant de trancher.
7. **Choisir un modèle uniquement pour sa réputation ("XGBoost est toujours le meilleur")** sans tester d'alternatives plus simples (régression logistique, arbre seul) — sur un petit jeu de données ou une relation simple, un modèle plus simple peut égaler voire dépasser un modèle complexe, avec bien plus d'interprétabilité.
8. **AutoML donne un point de départ, pas une réponse finale** : toujours valider les résultats et comprendre le modèle retenu avant de le déployer.
