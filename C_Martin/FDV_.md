# Fiche de cours — Introduction à la fouille de données (data mining)

*UE Fouille de données du Vivant — DA IODAA, Master BIBS-IA — Christine Martin (AgroParisTech)*
*Objectif de l'UE : découvrir, rappeler, approfondir les concepts et méthodes de la fouille de données et de l'apprentissage non supervisé. 7 séances, contrôle continu, examen de 2 h.*

## Sommaire

1. [Définitions de la fouille de données](#1-définitions)
2. [Positionnement : bases de données, OLAP, disciplines](#2-positionnement)
3. [Fouille de données aujourd'hui](#3-aujourdhui)
4. [Application : relation client (CRM)](#4-crm)
5. [Domaines d'application](#5-domaines)
6. [Processus d'ECBD](#6-ecbd)
7. [Familles de méthodes](#7-familles)
8. [Critères de réussite](#8-réussite)
9. [Les données (représentation, types, qualité, transformation)](#9-données)
10. [Outils](#10-outils)
11. [Pièges à éviter](#11-pièges)
12. [Quiz / auto-test](#12-quiz)

---

<a id="1-définitions"></a>
## 1. Définitions

- Domaine « récent » : **mélange d'idées et d'outils** venant de la **statistique**, de l'**intelligence artificielle** et de l'**informatique**.
- La définition exacte reste **peu claire** et la terminologie encore **floue** (en particulier : *fouille de données* vs *science des données*).

| Définition | Contenu |
|---|---|
| **Déf. 1** | **Processus de découverte** de règles, relations, corrélations et/ou dépendances **à travers une grande quantité de données**, grâce à des méthodes **statistiques, mathématiques et de reconnaissance de formes**. |
| **Déf. 2** | **Processus d'extraction automatique d'informations prédictives** à partir de **grandes bases de données**. |
| **Déf. 3** (ensemble de techniques) | Aide à la **découverte de régularités** (règles, relations, corrélations, dépendances) pour **expliquer et prédire**, dans de **grandes bases de données** (accent sur la **complexité des algorithmes**), fondées sur des méthodes **statistiques, d'apprentissage artificiel, de visualisation**, etc. |

Caractéristiques complémentaires :
- Accent mis sur l'**objectif** (traitement d'applications) plutôt que sur les justifications théoriques.
- Plutôt associée aux problématiques **non supervisées**.
- Nombreuses applications.

> 💡 **Idée clé** : on cherche des **régularités** (règles, corrélations…) dans de **gros volumes** de données, pour **expliquer et prédire**, avec une approche **pragmatique** (l'application prime sur la théorie).

---

<a id="2-positionnement"></a>
## 2. Positionnement

### 2.1 Évolution des technologies de bases de données

La fouille de données est la **suite logique de l'évolution des techniques de bases de données**.

| Technologie | Rôle | Caractéristiques |
|---|---|---|
| **OLTP** (On-Line Transaction Processing) | Gérer des requêtes **simultanées** | Requêtes « simples », « à faible impact » ; données souvent **dispersées, détaillées mais de faible qualité** |
| **Entrepôt de données** (Data Warehouse) | Stocker des informations **synthétiques « nettoyées »** | Nouvelle base de données dédiée |
| **OLAP** (On-Line Analytical Processing) | **Analyser en « temps réel »** l'entrepôt | **Objectif défini** ; données en **tableaux multidimensionnels** = **cubes de données** (*data cube*) |
| **Fouille de données** (Data mining) | **Explorer** les données à la recherche de « connaissances » | Règles, tendances, régularités, modèles… |

### 2.2 OLAP vs data mining

| | OLAP | Data mining |
|---|---|---|
| Démarche | **Vérification d'hypothèses** guidée par l'**utilisateur** | **Construction d'hypothèses** basée sur les **données** |

> 💡 OLAP : *« je teste ce que je pense déjà »* ; data mining : *« les données me suggèrent des hypothèses »*.

### 2.3 Positionnement parmi les disciplines

La fouille de données se situe au croisement de : **Statistiques**, **Analyse de données**, **Apprentissage artificiel**.

| | **Statistiques** | **Analyse de données** | **Fouille de données** |
|---|---|---|---|
| Nb d'individus | quelques **centaines** | quelques **dizaines de milliers** | plusieurs **millions** |
| Nb de variables | quelques variables | quelques **dizaines** | plusieurs **centaines** |
| Type de variables | recueillies avec un **protocole spécial** (échantillonnage, plan d'expérience…) | tableaux **« Individus × Variables »** | nombreuses variables **non numériques** (textuelles…) |
| Hypothèses sur les lois | **fortes** hypothèses sur les lois suivies | — | **faibles** hypothèses |
| Origine des modèles | issus de la **théorie**, confrontés aux données | importance du **calcul** et de la **représentation visuelle** (ACP, ACM…) | issus des **données** (+ éléments théoriques) |
| Méthodes / usage | probabilistes et statistiques ; **laboratoire** | — | **entreprise** |

#### Particularités de la fouille de données

**Les données**
- Recueillies **avant l'étude**, et souvent **à d'autres fins**.
- **Imparfaites** : erreurs de saisie, de codification, valeurs manquantes, aberrantes.
- **Population constamment évolutive** (difficulté d'échantillonner).

**Les besoins**
- **Calculs rapides**, parfois en **temps réel**.
- On ne cherche **pas toujours l'optimum mathématique**, mais un **modèle facile à appréhender et à interpréter** par des utilisateurs souvent **non-statisticiens** → importance de l'**explicabilité**.

**Les méthodes exploitées** : issues des statistiques, de l'analyse de données, de l'IA, de l'apprentissage artificiel (notamment **supervisé** : réseaux de neurones, arbres de décision, SVM… mais aussi **non supervisé**).

> ⚠️ **Attention** : en fouille de données, le modèle vient **des données** avec peu d'hypothèses ; en statistique classique, il vient **de la théorie** avec de fortes hypothèses.

> 🔎 **Complément** : cette distinction « statistique vs fouille » est un ordre de grandeur donné au cours, pas une frontière stricte (les chiffres sont indicatifs).

---

<a id="3-aujourdhui"></a>
## 3. Fouille de données aujourd'hui

- La fouille de données **est entrée dans les entreprises** ; on parle de **science des données** (« en fait pas tout à fait » : les deux termes ne sont pas équivalents).
- Elle **exploite** :
  - des techniques **pas toujours récentes** issues de la **recherche en IA** et de la **théorie de l'apprentissage** ;
  - les **capacités de stockage et de calcul** du matériel et des techniques informatiques modernes ;
  - la constitution de **giga-bases de données** pour les besoins de gestion des entreprises.
- **But** : traiter de **grands volumes de données**.
- Elle se répand surtout dans les secteurs détenant de nombreuses **données terrain, économiques et comportementales individualisées** → désormais « partout ».
- Selon le **MIT** : l'une des **10 technologies émergentes** qui « changeront le monde » au XXIᵉ siècle.

### Raisons du développement
- **Intérêts économiques**
- Progrès des **technologies de l'information**
- **Faible coût de stockage** des bases de données
- **Saisie automatique** de transactions (codes-barres, clics, internet…)
- **Objets connectés**
- Augmentation de la **puissance de calcul**

> 💡 Résultat : extraire de l'information de **très grandes bases de données** est devenu **possible**.

---

<a id="4-crm"></a>
## 4. Application : relation client (CRM)

**GRC / CRM** = *Gestion de la Relation Client* / *Customer Relationship Management*.

- **Richesse des entreprises : leurs clients.**
- **Objectifs** : augmenter la **rentabilité** et la **fidélité** des clients, en **maîtrisant les risques**, en utilisant **les bons canaux, au bon moment, pour vendre le bon produit**.
- Moyen : le CRM, en **2 volets** ; matière première précieuse : **les données sur les clients**.

| | **CRM opérationnel** | **CRM analytique** |
|---|---|---|
| **Objectif** | **Mise en œuvre optimale** des stratégies identifiées par le CRM analytique | Fournir une **vision complète et unifiée** du client, mieux comprendre son **profil et ses besoins** |
| **Moyens** | **Gestion des canaux** (forces commerciales, centres d'appels, serveurs vocaux, centres d'appel web, bornes interactives, téléphonie mobile, TV interactive…) et **gestion des campagnes marketing** | **Extraction, stockage, analyse et restitution** des informations pertinentes |
| **Composants** | Outils interfacés avec le *back-office*, les **ERP**, les outils de *workflow*, de gestion des agendas et des alertes commerciales | **Entrepôt de données**, **analyse multidimensionnelle (OLAP)**, **fouille de données**… |

Schéma du cours (cycle) : **Collecte des informations clients → Analyse des informations clients** (analytique) **→ Gestion des campagnes → Gestion des canaux** (opérationnel).

### Ce que l'on veut savoir
- Avant : « Combien de clients ont acheté tel produit pendant telle période ? »
- Maintenant : « **Quel est leur profil ?** », « **Quels autres produits les intéresseront ?** », « **Quand** seront-ils intéressés ? »

> 💡 **Data mining ≠ statistiques descriptives** : les profils de clientèle sont **complexes** — pas seulement des oppositions « jeunes/seniors » ou « citadins/ruraux », mais des **combinaisons complexes qui ne seraient pas découvertes par hasard**. Le data mining permet des **analyses exploratoires** avec une **ambition d'exhaustivité**.

### Utilité
- Mieux connaître le client **pour mieux le servir**, **augmenter sa satisfaction et sa fidélité** (**acquérir un client coûte plus cher que le conserver**).
- Encore plus utile dans le **secteur tertiaire** : produits similaires entre établissements, prix pas toujours déterminant → **le service et la relation client font la différence**.

### Applications au CRM
| Application | Description |
|---|---|
| **Étude d'appétence** | Concentrer mailings et phoning sur les clients **les plus susceptibles de répondre favorablement** |
| **Prédiction de l'attrition** | **Attrition = départ d'un client pour un concurrent** |
| **Analyse du ticket de caisse** | Déterminer les produits **souvent achetés simultanément** → agencer les rayons, organiser les promotions |

---

<a id="5-domaines"></a>
## 5. Domaines d'application

| Domaine | Exemples vus en cours |
|---|---|
| **Entreprise / relation client** | Création de **profils clients**, ciblage de clients potentiels / nouveaux marchés (« quel produit ce client sera-t-il tenté d'acheter ? », promotions ciblées — pas de promo tondeuse à gazon pour quelqu'un en appartement) ; **agencement de magasins** (regrouper des produits dans un même rayon) |
| **Banques / finances** | **Minimisation des risques financiers** (fraudes, optimisation des prêts) ; **détection de fraudes** (« quelle déclaration de revenus a le plus de chances d'être malhonnête ? ») |
| **Bioinformatique** | Analyse du **génome**, mise au point de **médicaments** |
| **Internet** | **Spam**, **e-commerce** |
| **Sécurité** | **Détection d'intrusions** ; quels événements sont des indices de menaces pour la sécurité nationale ? |

**Étendue des domaines** (liste de contrastes du cours) :
- de l'**infiniment petit** (génomique) à l'**infiniment grand** (astrophysique) ;
- du plus **quotidien** (reconnaissance de l'écriture manuscrite sur les enveloppes) au moins quotidien (aide au pilotage aéronautique) ;
- du plus **ouvert** (e-commerce) au plus **sécuritaire** (fraude en téléphonie mobile, cartes bancaires) ;
- du plus **industriel** (contrôle qualité) au plus **théorique** (sciences humaines, biologie) ;
- du plus **alimentaire** (agronomie, agroalimentaire) au plus **divertissant** (prévisions d'audience TV).

### Exemples travaillés

**E-commerce**
- **Dell** — *Problème* : 50 % des clients achètent via le site Web, mais seulement **0,5 % des visiteurs** deviennent clients. *Solution* : **stocker les séquences de clics** des visiteurs, analyser les caractéristiques des acheteurs et, lors de la visite d'un client potentiel, **adapter le contenu du site** pour maximiser la probabilité d'achat.
- **Amazon** — *Opportunité* : les listes d'achats sont stockées et les utilisateurs **notent** les produits. Comment tirer profit des choix d'un utilisateur pour proposer des produits à un autre ? *Solution* : le **filtrage collaboratif**, qui **regroupe les clients ayant les mêmes goûts**.

**Analyse des risques**
- **Fraude à l'assurance** : (1) un **expert** analyse les déclarations pour identifier les fraudes ; (2) **extraction de caractéristiques** (type d'accident, de blessures…) ; (3) **méthodes statistiques** pour identifier les caractéristiques **fortement corrélées à la fraude**.
- **Prêt bancaire** : objectif = **réduire le risque** ; on crée un **modèle à partir des caractéristiques des clients** pour **discriminer les clients à risque** des autres.

**Commerce — organisation du rayonnage**
- *Objectif* : identifier les produits **achetés conjointement**. *Données* : **codes-barres**. *Méthode* : **extraction de règles**.
- Résultat « logique » : boissons alcoolisées et biscuits apéritifs souvent proches.
- Résultat « étrange » (étude américaine) : la vente de **bière augmente si le rayon couches n'est pas trop loin**, et si des **chips** sont sur le chemin, les ventes des 3 produits augmentent.

**Sciences du vivant** (problématiques complexes à toutes les échelles)
- **Agronomie** : sélection variétale, estimation de rendement, protection des cultures…
- **Environnement** : effets des modifications climatiques, estimation de la diversité de la faune…
- **Santé** : médecine ciblée, aide à la rééducation…

> ⚠️ **Attention (conclusion du cours)** :
> - Avoir un **regard critique sur les données** (souvent récoltées **sans connaître les objectifs visés**).
> - **Ne pas réinventer la roue** : certains résultats sont **triviaux**, ce sont **les autres** qui sont intéressants.
> - **L'expert a une place centrale** pour limiter les risques.

---

<a id="6-ecbd"></a>
## 6. Processus d'ECBD (Extraction de Connaissances dans les Bases de Données)

Processus en **7 étapes**, avec une **validation par un expert** (schéma du cours) :

| # | Étape | But |
|---|---|---|
| 1 | **Nettoyage** des données | Éliminer le **bruit** et les données **incohérentes** |
| 2 | **Intégration** des données | Quand elles proviennent de **sources différentes** |
| 3 | **Sélection** des données | Rechercher les données **pertinentes** dans la base |
| 4 | **Transformation** des données | Les mettre sous une **forme appropriée** pour la recherche de régularités (peut inclure **résumé** ou **agrégation**) |
| 5 | **Fouille de données** | **Extraction de régularités** |
| 6 | **Évaluation** des régularités | Ne garder que les régularités **intéressantes** |
| 7 | **Représentation** des connaissances extraites | **Visualisation** / représentation pour transmettre clairement les régularités aux utilisateurs finaux |

> 💡 **Ce cours se concentre sur l'étape 5** (fouille de données proprement dite) et sur **les méthodes à employer**.
>
> 💡 La fouille de données est un **processus automatique, centré sur l'expert** (dernière diapo).

---

<a id="7-familles"></a>
## 7. Familles de méthodes

| Famille | Données | Objectif | Algorithmes / exemples |
|---|---|---|---|
| **Supervisée** | Ensemble d'exemples **et leur « étiquette »** | Trouver des **règles de prédiction** | **Arbres de décision**, **réseaux de neurones**, **SVM**… ; applications : détection de fraude, marketing téléphonique, changement d'opérateur téléphonique |
| **Non supervisée** | Ensemble d'exemples / d'objets **sans étiquette** | **Identifier des régularités d'un type cherché** : **catégories**, **tendances**, **exemples atypiques** | *Catégorisation* : **k-moyennes**, **clustering hiérarchique**, **cartes auto-organisées de Kohonen**. *Caractérisation* : recherche de **motifs intéressants** (fréquents / **règles d'association**). Applications : **segments de marché**, **documents similaires** |
| **Semi-supervisée** | Observations $\{x_1,\dots,x_n\}\in\mathcal{X}^d$, **seul un petit nombre** a un label $\{y_i\}$ | **Même objectif que le supervisé**, mais en **tirant profit des observations non labélisées** | Méthodes bayésiennes, **SVM** (Séparateur à Vastes Marges), etc. |

Légende de la notation (semi-supervisé) :
- $x_i$ : la $i$-ème observation ; $n$ : nombre d'observations ;
- $\mathcal{X}^d$ : espace des observations, de dimension $d$ (nombre d'attributs) ;
- $y_i$ : le label (étiquette), connu seulement pour un petit sous-ensemble.

> 💡 **Exemple semi-supervisé (cours)** : en discrimination d'images, le nombre d'exemples peut être très grand mais **leur associer un label est coûteux**.

---

<a id="8-réussite"></a>
## 8. Critères de réussite d'un projet de fouille de données

1. **Identifier et énoncer clairement les besoins.**
2. **Créer ou obtenir des données de qualité**, **représentatives du problème**.
3. **Identifier la ou les méthodes adaptées.**
4. **Analyser et réduire la dimension** des données.
5. **Choisir un algorithme et/ou un espace d'hypothèses.**
6. **Valider les performances** de la méthode.

---

<a id="9-données"></a>
## 9. Les données

### 9.1 Représentation : forme attribut–exemple

| Terme | Définition | Synonymes |
|---|---|---|
| **Attribut** | **Descripteur d'une entité** | variable, champ, caractéristique, observation |
| **Exemple** | **Entité caractérisant un objet**, constituée d'attributs | point, vecteur (surtout si les exemples sont dans $\mathbb{R}^n$) |

### 9.2 Types de données

| Type | Exemple du cours |
|---|---|
| **Numérique continue** (dans $\mathbb{R}$) | montant d'un compte en banque |
| **Numérique discrète** (dans $\mathbb{Z}$ ou $\mathbb{N}$) | âge d'un client |
| **Binaire** | — |
| **Catégorielle** (avec ou sans relation d'ordre) | $\{$rouge, vert, bleu$\}$ |
| **Chaîne de caractères** | un texte |
| **Données structurées** | arbre (page XML), graphe… |
| **Séquences d'événements** | séquences de clics, de commandes d'ordinateur |
| **Données spatiales** | images satellites + positionnement GPS |

### 9.3 Qualité des données

Qualité = **combinaison subtile** de propriétés : **précision, cohérence, complétude, fraîcheur, unicité**.

C'est un problème :
- **fondamental / omniprésent** (toutes les applications) ;
- **complexe** : intrinsèque à toute BD, entrepôt ou système d'information ; **frontière floue** entre bonnes et mauvaises données ;
- **critique** : coûte énormément.

### 9.4 Intégration et transformation

- **Identifier les entités** (ex. `client_id` et `cl_nr` désignent la même chose).
- **Uniformiser les unités** (ex. dollar et euro).
- **Convertir les adresses en coordonnées.**
- **Agréger** (ex. calculer la vente quotidienne à partir des ventes individuelles).
- **Normaliser** les variables.

Formule du cours : pour un attribut $A$, remplacer chaque valeur $x$ par

$$x' = \frac{x-\mu}{\sigma}$$

Hypothèses : $A$ est numérique, de moyenne $\mu$ et de variance $\sigma^2$ (donc d'écart-type $\sigma>0$).

- $x$ : valeur initiale de l'attribut $A$ ;
- $\mu$ : moyenne de $A$ ; $\sigma$ : écart-type de $A$ ;
- $x'$ : valeur transformée, de **moyenne 0** et de **variance 1**.

> ⚠️ **Incohérence dans la diapo** : le titre dit « normaliser entre 0 et 1 » mais la formule $(x-\mu)/\sigma$ donne une variable **centrée-réduite** (moyenne 0, variance 1), **pas** bornée dans $[0,1]$.
>
> 🔎 **Complément** : ce sont deux transformations différentes — **standardisation** (centrage-réduction, $\frac{x-\mu}{\sigma}$) et **normalisation min-max** ($\frac{x-x_{\min}}{x_{\max}-x_{\min}}$, qui borne dans $[0,1]$). Utile pour que des variables d'échelles différentes pèsent de façon comparable dans les calculs de distance (k-moyennes, etc.).

---

<a id="10-outils"></a>
## 10. Outils

| Type | Outils |
|---|---|
| **Commerciaux** | **Clementine** (SPSS), **Enterprise Miner** (SAS), **Intelligent Miner** (IBM) |
| **Libres** | **Weka** (universitaire), **Orange** (prototypage rapide), (Tanagra), **Scikit-Learn**, bibliothèques **Python** ciblées |

Site de référence cité : [kdnuggets.com](http://www.kdnuggets.com/).

---

<a id="11-pièges"></a>
## 11. Pièges à éviter

> 🚩 **Confondre OLAP et data mining** : OLAP = on **vérifie** des hypothèses de l'utilisateur ; data mining = les hypothèses **viennent des données**.

> 🚩 **Confondre fouille de données et statistiques descriptives** : le data mining cherche des **combinaisons complexes**, pas de simples oppositions (jeunes/seniors…).

> 🚩 **Confondre « fouille de données » et « science des données »** : le cours précise que ce n'est « pas tout à fait » la même chose.

> 🚩 **Confondre standardisation et normalisation [0,1]** : la formule $(x-\mu)/\sigma$ ne borne pas dans $[0,1]$.

> 🚩 **Confondre supervisé / non supervisé / semi-supervisé** : présence d'étiquettes pour tous, aucun, ou seulement un petit nombre des exemples.

> ⚠️ **Faire confiance aveuglément aux données** : elles sont souvent récoltées **sans connaître l'objectif** et sont **imparfaites** (erreurs, manquants, aberrants).

> ⚠️ **Chercher l'optimum mathématique à tout prix** : on préfère un modèle **explicable**, interprétable par des non-statisticiens.

> ⚠️ **Oublier l'expert** : il valide le processus d'ECBD et limite les risques ; certains résultats sont **triviaux** (« ne pas réinventer la roue »).

> ⚠️ **Se focaliser sur toutes les étapes** : ce cours porte surtout sur l'**étape 5** (fouille) de l'ECBD.

---

<a id="12-quiz"></a>
## 12. Quiz / auto-test

*Barème indicatif : ⭐⭐⭐ = très important (3 pts), ⭐⭐ = important (2 pts), ⭐ = complément (1 pt).*

<details>
<summary><b>Question 1 (⭐⭐⭐) :</b> Donne une définition de la fouille de données. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Processus de <b>découverte</b> (ou d'<b>extraction automatique</b>) de règles, relations, corrélations, dépendances, informations prédictives, à travers une <b>grande quantité de données</b>, grâce à des méthodes statistiques, mathématiques, d'apprentissage artificiel, de reconnaissance de formes et de visualisation, pour <b>expliquer et prédire</b>. L'accent est mis sur l'objectif applicatif et la complexité des algorithmes.
</details>

<details>
<summary><b>Question 2 (⭐⭐⭐) :</b> Quelle est la différence entre OLAP et data mining ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>OLAP</b> : vérification d'hypothèses <b>guidée par l'utilisateur</b>, sur des données en cubes multidimensionnels, avec un objectif défini. <b>Data mining</b> : <b>construction d'hypothèses</b> à partir des données, exploration à la recherche de connaissances (règles, tendances, régularités, modèles).
</details>

<details>
<summary><b>Question 3 (⭐⭐) :</b> Que sont OLTP, entrepôt de données et cube de données ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>OLTP</b> : gestion de requêtes simultanées simples et à faible impact, sur des données dispersées, détaillées, de faible qualité. <b>Entrepôt</b> : base contenant des informations synthétiques « nettoyées ». <b>Cube de données</b> : tableau multidimensionnel utilisé par OLAP pour analyser l'entrepôt en « temps réel ».
</details>

<details>
<summary><b>Question 4 (⭐⭐⭐) :</b> Compare statistiques, analyse de données et fouille de données (volumes, hypothèses, origine des modèles). <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Statistiques</b> : quelques centaines d'individus, peu de variables recueillies avec un protocole, fortes hypothèses sur les lois, modèles issus de la théorie, usage en laboratoire. <b>Analyse de données</b> : dizaines de milliers d'individus, dizaines de variables, tableaux Individus × Variables, calcul et visualisation (ACP, ACM). <b>Fouille de données</b> : millions d'individus, centaines de variables (dont non numériques), données collectées à d'autres fins et imparfaites, faibles hypothèses, modèles issus des données, calcul rapide, explicabilité.
</details>

<details>
<summary><b>Question 5 (⭐⭐) :</b> Quelles sont les imperfections typiques des données en fouille de données ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Erreurs de saisie, erreurs de codification, valeurs manquantes, valeurs aberrantes ; en plus : données recueillies avant l'étude et à d'autres fins, population constamment évolutive (échantillonnage difficile).
</details>

<details>
<summary><b>Question 6 (⭐⭐) :</b> Pourquoi cherche-t-on un modèle « facile » plutôt que l'optimum mathématique ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Les utilisateurs sont souvent non-statisticiens : le modèle doit être facile à appréhender et à interpréter (importance de l'<b>explicabilité</b>). Il faut aussi des calculs rapides, parfois en temps réel.
</details>

<details>
<summary><b>Question 7 (⭐⭐) :</b> Pourquoi la fouille de données s'est-elle développée ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Intérêts économiques ; progrès des technologies de l'information ; faible coût de stockage ; saisie automatique de transactions (codes-barres, clics, internet) ; objets connectés ; augmentation de la puissance de calcul. Extraire de l'information de très grandes bases est devenu possible.
</details>

<details>
<summary><b>Question 8 (⭐⭐⭐) :</b> Différence entre CRM opérationnel et CRM analytique ? Où se place la fouille de données ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Analytique</b> : vision complète et unifiée du client, compréhension de son profil et de ses besoins (collecte, stockage, analyse, restitution) ; composants : entrepôt, OLAP, <b>fouille de données</b>. <b>Opérationnel</b> : mise en œuvre optimale des stratégies identifiées (gestion des canaux et des campagnes marketing), interfacé avec back-office, ERP, workflow, agendas. La fouille de données fait partie du CRM <b>analytique</b>.
</details>

<details>
<summary><b>Question 9 (⭐⭐) :</b> Quelles questions pose-t-on aujourd'hui en plus de « combien de clients ont acheté tel produit ? » Pourquoi ce n'est pas de la statistique descriptive ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Quel est leur profil ? Quels autres produits les intéresseront ? Quand ? Les profils sont des combinaisons complexes, non découvrables par hasard (au-delà de « jeunes/seniors », « citadins/ruraux ») : le data mining apporte des analyses exploratoires avec une ambition d'exhaustivité.
</details>

<details>
<summary><b>Question 10 (⭐⭐) :</b> Définis étude d'appétence, attrition et analyse du ticket de caisse. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Appétence</b> : cibler mailings/phoning sur les clients les plus susceptibles de répondre favorablement. <b>Attrition</b> : départ d'un client vers un concurrent (on cherche à la prédire). <b>Ticket de caisse</b> : déterminer les produits souvent achetés ensemble pour agencer les rayons et organiser les promotions.
</details>

<details>
<summary><b>Question 11 (⭐) :</b> Pourquoi la connaissance du client est-elle encore plus utile dans le tertiaire ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Les produits se ressemblent entre établissements, le prix n'est pas toujours déterminant : ce sont le service et la relation client qui font la différence. De plus, acquérir un client coûte plus cher que le conserver.
</details>

<details>
<summary><b>Question 12 (⭐⭐) :</b> Explique les cas Dell et Amazon. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Dell</b> : 50 % des clients achètent via le site mais seulement 0,5 % des visiteurs deviennent clients → stocker les séquences de clics, analyser les acheteurs, adapter le contenu du site au visiteur pour maximiser la probabilité d'achat. <b>Amazon</b> : listes d'achats et notes des utilisateurs → <b>filtrage collaboratif</b> regroupant les clients aux goûts similaires pour recommander des produits.
</details>

<details>
<summary><b>Question 13 (⭐⭐) :</b> Décris les applications en détection de fraude d'assurance et en prêt bancaire. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Assurance</b> : un expert identifie des cas de fraude dans les déclarations ; extraction de caractéristiques (type d'accident, de blessures) ; méthodes statistiques pour repérer celles fortement corrélées à la fraude. <b>Prêt bancaire</b> : réduire le risque en construisant un modèle à partir des caractéristiques clients qui discrimine les clients à risque.
</details>

<details>
<summary><b>Question 14 (⭐) :</b> Quel est l'exemple du rayonnage ? Quelles données et méthode ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Objectif : identifier les produits achetés conjointement pour organiser les rayons. Données : codes-barres des produits. Méthode : extraction de règles. Résultat logique : alcool et biscuits apéritifs proches. Résultat étrange : plus de bière vendue si le rayon couches est proche et que des chips sont sur le chemin.
</details>

<details>
<summary><b>Question 15 (⭐) :</b> Cite des applications en sciences du vivant. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Agronomie (sélection variétale, estimation de rendement, protection des cultures), environnement (effets climatiques, diversité de la faune), santé (médecine ciblée, aide à la rééducation) ; bioinformatique (analyse du génome, mise au point de médicaments).
</details>

<details>
<summary><b>Question 16 (⭐⭐⭐) :</b> Quelles sont les 7 étapes du processus d'ECBD ? Sur laquelle porte le cours ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
1) Nettoyage, 2) Intégration, 3) Sélection, 4) Transformation, 5) <b>Fouille de données</b>, 6) Évaluation des régularités, 7) Représentation des connaissances. Avec validation par un expert. Le cours se concentre sur l'étape <b>5</b>.
</details>

<details>
<summary><b>Question 17 (⭐⭐⭐) :</b> Distingue méthodes supervisées, non supervisées et semi-supervisées, avec algorithmes et applications. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Supervisées</b> : exemples + étiquettes, on cherche des règles de prédiction (arbres de décision, réseaux de neurones, SVM ; fraude, marketing téléphonique, changement d'opérateur). <b>Non supervisées</b> : exemples sans étiquette, on cherche des régularités (catégories, tendances, exemples atypiques) : k-moyennes, clustering hiérarchique, cartes de Kohonen (catégorisation), motifs fréquents / règles d'association (caractérisation) ; segments de marché, documents similaires. <b>Semi-supervisées</b> : seul un petit nombre d'observations est labélisé, même objectif que le supervisé en exploitant aussi les non labélisées (méthodes bayésiennes, SVM) ; utile quand l'étiquetage est coûteux (images).
</details>

<details>
<summary><b>Question 18 (⭐⭐) :</b> Quels sont les critères de réussite d'un projet de fouille de données ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Énoncer clairement les besoins ; obtenir des données de qualité et représentatives ; identifier les méthodes adaptées ; analyser et réduire la dimension ; choisir un algorithme et/ou un espace d'hypothèses ; valider les performances.
</details>

<details>
<summary><b>Question 19 (⭐⭐) :</b> Définis attribut et exemple, et cite des types de données. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
<b>Attribut</b> : descripteur d'une entité (variable, champ, caractéristique). <b>Exemple</b> : entité caractérisant un objet, composée d'attributs (point, vecteur dans ℝⁿ). Types : numérique continue (ℝ), discrète (ℤ, ℕ), binaire, catégorielle (ordonnée ou non), chaîne de caractères, données structurées (arbre, graphe), séquences d'événements, données spatiales.
</details>

<details>
<summary><b>Question 20 (⭐⭐) :</b> Quelles propriétés définissent la qualité des données ? Pourquoi est-ce un problème important ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Précision, cohérence, complétude, fraîcheur, unicité. Problème fondamental (omniprésent), complexe (frontière floue entre bonnes et mauvaises données) et critique (coûte énormément).
</details>

<details>
<summary><b>Question 21 (⭐⭐) :</b> Donne la formule de normalisation du cours, ses propriétés et sa limite par rapport au titre « entre 0 et 1 ». <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La valeur x d'un attribut de moyenne μ et de variance σ² est remplacée par (x − μ) / σ. On obtient moyenne 0 et variance 1 (centrage-réduction).

Cette transformation ne borne <b>pas</b> dans $[0,1]$ : c'est une standardisation, différente de la normalisation min-max $\frac{x-x_{\min}}{x_{\max}-x_{\min}}$.
</details>

<details>
<summary><b>Question 22 (⭐) :</b> Cite des opérations d'intégration et de transformation des données. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Identifier les entités (client_id / cl_nr), uniformiser les unités (dollar / euro), convertir les adresses en coordonnées, agréger (ventes quotidiennes à partir des ventes individuelles), normaliser les variables.
</details>

<details>
<summary><b>Question 23 (⭐⭐) :</b> Quelles précautions le cours recommande-t-il face à un projet de fouille de données ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Garder un regard critique sur les données (souvent récoltées sans connaître les objectifs), ne pas réinventer la roue (certains résultats sont triviaux, ce sont les autres qui intéressent), et placer l'expert au centre pour limiter les risques.
</details>

<details>
<summary><b>Question 24 (⭐) :</b> Cite quelques outils de fouille de données. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Commerciaux : Clementine (SPSS), Enterprise Miner (SAS), Intelligent Miner (IBM). Libres : Weka, Orange, Tanagra, Scikit-Learn, bibliothèques Python ciblées.
</details>
