# Partie A — Questions connues

## Question 4 (1 pt) — Classifieur « classe majoritaire »
- train : $x_1 = x_2$ = 35 $\to$ prédiction aléatoire ; 15/15 en test $\to$ taux d'erreur = 50% en moyenne  
- train : $x_1 > 35$, $x_2 = 70 - x_1$ $\to$ prédiction de 1 ; 50-$x_1$ / 50-$x_2$=$x_1$-20 $\to$ $\displaystyle E=\frac{30-(50-x_1)}{30}=\frac{x_1-20}{30} > 50\%$ allant jusqu'à 100% pour $x_1$=50

## Question 5 (2 pts) — Courbe ROC
$\displaystyle TPR = \frac{TP}{TP+FN}$  et  $\displaystyle FPR = \frac{FP}{FP+TN}$  
avec :
- TPR (taux de vrai positif): rappel          | combien de vrai positif sont trouvés
- FPR (taux de faux positif): fausses alertes | sur tout les néatif, cbm passe en +   
  
ROC $\to$ mesure la capacité d'un classifieur a bien séparer les 2 classes, $\forall$ seuil décision  
AUC $\to$ aire sous la coubre :
- Parfaite : AUC = 1 |on classe bien toutes les sorties
- Pire : AUC = 0.5 |au niveau du hasard
- AUC = 0 ? C'est en réalité une prédiction parfaite, juste les labelles de sorti sont inversés (appliquer XOR)  

## Question 6 (2 pts) — Précision et rappel  
- TPR (taux de vrai positif): rappel          | combien de vrai positif sont trouvés
- precision : vrai alertes | $\displaystyle \frac{TP}{TP+FP}$ = $\forall$ détectés +, cbm sont de vrai +  

Comprois précision rappel repose sur le fait que si le classifier n'est pas capable de séparer en 2 les données, il faudra poser une limite arbitraire qui laissera forcement des exemples mal placés. Dans ce cas, on fait un compromis, soit on veut avoir un max de positif bien placé, soit des négatifs.  
- moteur de rechere qui rend tt internet : contient forcement toutes les solutions (rappel = 1), mais aussi tout les lien inutiles (precision très faible)
- moteur de recherche qui rend 1/2 bonnes solutions : precision = 1 (que des bonnes réponses) mais precision très faible car fait l'impasse sur bcp de bonnes solutions.  
  
## Question 7 (2 pts) — Minimisation du risque empirique (MRE)
Le risque réel n'étant pas atteignable (car on ne connait pas la vrai distribution des donénes), il faut l'approximer. Le MRE permet de l'estimer avec précaution : on utilise le risque empirique calculé sur nos données, en ajoutant un terme de régulation limittant le sur-apprentissage (souvent limiter la complexité du modèle), et on cherche à minimiser ce risque.  
$R(h) = R_{emp}(h)+ term.complexité(H,m)$  
On va donc minimiser le risque régularisé :  
$R_{reg}(h) = R_{emp}(h) + \lambda.\Omega(h)$  
avec :  
- $\lambda$   : importance de la régularisation
- $\Omega(h)$ : fonction de régularisation ($L_1, L_2, depth...$)  

## Question 11 (1 pt) — One-hot encoding  
Coder une variable catégorielle (n catégories numérique ou caractérielles) dans n (ou n-1) colonnes contennant des 0|1.  

| color | is_green | is_dark | is_light |
|---|---|---|---|
| green | 1 | 0 | 0 |
| dark | 0 | 1 | 0 |
| light | 0 | 0 | 1 |  

Permet de renter dans des modèles qui n'acceptent pas les caractères, ou s'abstraire d'un ordre/distance introduit arbitrairement par une transformation green=1, dark=2, light=3. Indispensable pour les modèles qui calculent sur des nombres (régression, réseau de neuronnes, k-PPV).  

## Question 15 (3 pts) — Arbre de décision
Entropie d'un ensemble D à 2 classes, n exemples au total :  
$\displaystyle H(D) = - \sum{\frac{n_i}{n}ln\frac{n_i}{n}}$  

Entropie après un test sur un descripteur A :  
$\displaystyle H_A(D) = \sum{\frac{|D_v|}{|D|}H(D_v)}$  
ou $D_v$ sous-ensemble des exemples pour lesquels $A = v$  

Gain d'information :  
$\displaystyle G(A)=H(D) - H_A(D)$  
Meilleur descripteur est celui qui a le plus gros gain, donc plus petite entropie.  

## Question 17 (4 pts) — Valeurs de Shapley  
- Si $\lambda.x_1$ : paramètre intervenant de facon additive, sans interaction (linéaire). Il apporte toujours $\lambda$ au résultat.  
- Si $\lambda.x_1.x_2$ : paramètres intervenant en interactions : n'apportent rien seul, mais le gain $\lambda$ lorsqu'ils sont les deux présents - Shapley le partage équitablement entre les deux (symétrie).  


#  Partie B - Questions bonus
## Espace des versions :  
$G_{set}$ : ensemble le plus général = incluant tout sauf les exemples négatifs et les plus générales que S  
$S_{set}$ : ensemble le plus spécifique = incluant uniquement les exemples positifs et les plus spécifiques que G  

## Compromis biais-variance :  
L'erreur de généralisaton se décompose en :  
$Erreur = Biais² + Variance + Bruit$  
- Erreur approximative (Biais) : le meilleur $h \in H$ loin de la cible f
- Erreur d'estimation (variance) : le $h_{step} \in H$ trouvé est loin du meilleur $h_{best} \in H$  
$\to$ un modèle comple à un biais faible mais forte variance ; un modèle simple l'inverse. Le compromis consiste à choisir la compléxité qui minimise la somme  

- bagging : ensemble de modèles instables, entrainer sur des échantillons bootstrap (aléaoitre avec remise, taille fixe) et on moyenne. Permet de réduire variance globale sans augmenter biais.  
- boosting : ensembles d'apprenant faibles, entrainé succéssivement sur les points faibles des précédents. Réduit surtout le biais. (peut sur apprendre si les données sont bruités, car insiste bcp sur les données les plus complexes qui peuvent etre du bruit).  

## Les types de biais :  
Biais inductif : ensemble des hypothèses a priori qu'un algo d'apprentisssage utilise pour généraliser à des exemples jamais vu $\to$
- Biais déclaratif / representatif : restreindre l'espace des H : on écarte certaines hypothèses a priori (qui nous semble impossible ...)
- Biais de recherche / procedural : restreindre les méthodes d'exploration de H : définir une méthode d'exploration (descente de gradient) et un critère d'arret  

autre
- biais de cursus : on choisit une méthode/ un ordre pour présenter les hypothèses  