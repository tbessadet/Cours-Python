<style>
/* Texte agrandi (x1.25) ; les lignes "Montrer" en bleu gardent leur taille d'origine */
p, li, td, th { font-size: 2em; }
li p { font-size: inherit; }
span[style*="1e6fd9"] { font-size: 0.5em; }
</style>

# Plan de l'oral (≈ 8 min)

| Partie | Durée |
|---|---|
| 1. Introduction : le dataset | 45 s |
| 2. Nettoyage des données | 1 min 30 |
| 3. Détection des fuites | 1 min |
| 4. Élaboration du modèle | 4 min |
| 5. Test final et conclusion | 1 min 30 |

Les numéros entre crochets [n] sont les numéros des cellules du notebook (en comptant à partir de 1).

---

## 1. Introduction : le dataset (≈ 45 s)

<span style="color:#1e6fd9">**Montrer** : la cellule markdown du titre [1].</span>

<img src="screen_add/0.png" alt="Présentation du dataset" style="border: 3px solid white; padding: 4px;">

**À dire** :
- « Dataset Kaggle synthétique : 100 000 patients et 209 variables (biologie, mode de vie, environnement, antécédents…). »
- « La cible est `diabetes_status`. Il y a 28 % de diabétiques, donc les classes sont déséquilibrées. J'ai écarté `diabetes_type` : Type 1 et gestationnel sont presque absents. »
- « **Objectif** : un outil de **dépistage** qui utilise seulement des analyses sanguines et des données simples, sans avis d'expert, pour être appliqué au plus grand nombre de patients. »

---

## 2. Nettoyage des données (≈ 1 min 30)

<span style="color:#1e6fd9">**Montrer** : le markdown 1.1 [6], puis le graphique de l'entonnoir [8].</span>

![Entonnoir des colonnes et lignes retirées](save/entonnoir_colonnes_lignes.png)

- « J'ai retiré trois familles de colonnes : les fuites de données (diagnostic, traitement), les scores "experts" dont on ne connaît pas le calcul, et les doublons. On passe de 209 à 130 colonnes et à 97 980 lignes. »

<span style="color:#1e6fd9">**Montrer** : le tableau croisé `sex` × `pregnancy_status` [24].</span>

<img src="screen_add/23.png" alt="Tableau croisé sex × pregnancy_status" style="border: 3px solid white; padding: 4px;">

- « Avant d'imputer, je remplis d'abord les NaN "logiques". Par exemple, `pregnancy_status = Not Applicable` implique un homme. »

<span style="color:#1e6fd9">**Montrer** : le tableau des 4 groupes de la fonction `Clean` [22]. Survole-le, ne le lis pas.</span>

<img src="screen_add/21.png" alt="Tableau des 4 groupes de Clean" style="border: 3px solid white; padding: 4px;">

- « Le pipeline encode les colonnes (binaires, ordinales, OneHot), puis impute avec `IterativeImputer`. Il est ajusté **sur le train uniquement**. »

<span style="color:#1e6fd9">**Montrer** : le tableau SMD / KS [35]. Une phrase suffit.</span>

<img src="screen_add/34.png" alt="Tableau SMD / KS" style="border: 3px solid white; padding: 4px;">

- « Je vérifie que l'imputation ne déforme pas les distributions : SMD ≈ 0 et KS faible, sur le train comme sur le test. »

---

## 3. Détection des fuites (≈ 1 min)

<span style="color:#1e6fd9">**Montrer** : la courbe de retrait glouton [38].</span>

![Courbe de détection des fuites](save/courbe_dataleak.png)

- « Pour vérifier qu'il ne reste pas de fuite, je retire à chaque étape la colonne la plus importante pour XGBoost. L'AUC reste à 1 tant qu'il reste une colonne qui contient la réponse. Elle chute à 0,88 sans `chronic_condition_count`, puis se stabilise vers 0,83. Ces 7 colonnes en rouge sont mes fuites. »

<span style="color:#1e6fd9">**Montrer** (rapidement) : le récapitulatif des 4 jeux [38].</span>

![Récapitulatif des étapes XGBoost](save/recap_etapes_xgboost.png)

- « Récapitulatif : avec les fuites on obtient un score parfait, mais faux. Sans elles, on a le vrai niveau du problème. »

---

## 4. Élaboration du modèle (≈ 4 min) : le cœur de l'oral

### 4a. Premiers modèles (20 s) [43]

![Effet du StandardScaler](save/effet_standardscaler.png)

- « Premier essai avec des modèles naïfs. La standardisation aide nettement la régression logistique. »

### 4b. Filtrage avancé (20 s) [48]

![Écart d'AUC des filtres](save/filtres_dauc.png)

- « Les filtres simples (variance, corrélation) retirent des colonnes sans perte d'AUC. »

### 4c. Sélection de variables (1 min 15)

<span style="color:#1e6fd9">**Montrer** : le tableau des familles de méthodes [49].</span>

<img src="screen_add/48.png" alt="Tableau des familles de méthodes de sélection" style="border: 3px solid white; padding: 4px;">

- « 7 méthodes de 4 familles : filtres, L1, sondes aléatoires (Boruta, SHAP), RFECV. Toutes sont évaluées sur les mêmes 10 plis. »

<span style="color:#1e6fd9">**Montrer** : la stabilité des méthodes [66].</span>

![Stabilité de Jaccard](save/stabilite_jaccard.png)

- « Une méthode qui choisit des colonnes différentes à chaque pli n'est pas fiable. Je mesure donc sa stabilité avec l'indice de Jaccard. »

<span style="color:#1e6fd9">**Montrer** : le choix du consensus [71].</span>

![Choix du consensus](save/choix_consensus.png)

- « Je garde les colonnes choisies par au moins 5 méthodes. On passe de **178 colonnes à 27, sans perte d'AUC** (dAUC ≈ 0). »

### 4d. Choix du modèle (30 s) [75]

![Comparaison des modèles](save/boxplot_modeles.png)

- « Logistique, XGBoost et forêt aléatoire font tous environ 0,80 d'AUC, bien au-dessus du Dummy. Je garde la logistique et XGBoost. »

### 4e. Hyperparamètres (45 s)

<span style="color:#1e6fd9">**Montrer** : le markdown 6.2 [76], puis le graphique avant / après réglage [78].</span>

![Performances avant / après réglage](save/reglage_avant_apres.png)

- « J'ai utilisé `RandomizedSearchCV` dans une **validation croisée imbriquée**, pour que le réglage ne gonfle pas les scores. Le critère est l'AP. Le gain est de quelques millièmes seulement : c'est la limite des données, pas du modèle. »

### 4f. Vote et seuil (50 s)

<span style="color:#1e6fd9">**Montrer** : le poids du vote [81].</span>

![Poids du vote](save/vote_poids.png)

- « Je moyenne les probabilités des deux modèles. Le meilleur poids est 0,8 pour XGBoost. »

<span style="color:#1e6fd9">**Montrer** : le choix du seuil [84].</span>

![Choix du seuil de décision](save/seuil_decision.png)

- « En dépistage, rater un diabétique coûte cher. Je fixe donc le seuil pour détecter 90 % des diabétiques : seuil = 0,11, précision = 0,34. »

---

## 5. Test final et conclusion (≈ 1 min 30)

<span style="color:#1e6fd9">**Montrer** : la sortie [87] (intervalles de confiance), puis les graphiques [88] et [89].</span>

![ROC, précision-rappel et matrice de confusion sur le test](save/test_roc_pr_confusion.png)

![Test vs validation croisée](save/test_vs_cv.png)

- « Le test est utilisé **une seule fois**, tout étant figé. AUC entre 0,79 et 0,81, rappel 0,90, précision 0,34. C'est cohérent avec la validation croisée : le modèle généralise, et la sélection n'a pas surappris. »

<span style="color:#1e6fd9">**Montrer** : le bilan [94], puis le markdown 7.3 [93].</span>

![Bilan de l'AUC à chaque étape](save/bilan_auc.png)

- « Plafond autour de 0,80 quel que soit le modèle. Limite : 2 alertes sur 3 sont fausses, et **74 % des patients sont signalés**. C'est un outil d'orientation vers un examen, pas un diagnostic. Pistes : régler le compromis rappel/précision selon le coût d'une fausse alerte, ou trouver de meilleures variables. »

---

## Si je suis en retard, couper dans cet ordre

1. 4a (premiers modèles)
2. 4b (filtres)
3. Le tableau SMD / KS [35]
4. Le récapitulatif `recap_etapes_xgboost`

## À corriger avant l'oral

- [ ] **Markdown 3.2 [39]** : il reste « QUElle métrique ??? » dans le tableau. Indiquer la métrique ou retirer la ligne.
- [ ] **Le 74 % d'alertes** : préparer une réponse (question probable du jury). Pour atteindre 90 % de rappel avec 28 % de diabétiques et une AUC de 0,80, il faut alerter très large.

## Graphiques en réserve (si le jury pose une question)

| Question possible | Graphique |
|---|---|
| Valeurs aberrantes ? | ![](save/aberrantes_asymetrie.png) |
| Corrélations avec la cible ? | ![](save/correlation_cible_avant_apres.png) |
| Nombre de colonnes vs performance par méthode ? | ![](save/selection_ncols_dauc.png) |
| Hyperparamètres stables d'un pli à l'autre ? | ![](save/reglage_params_par_pli.png) |
| Quelles variables comptent le plus ? | ![](save/importances_modele_final.png) |
