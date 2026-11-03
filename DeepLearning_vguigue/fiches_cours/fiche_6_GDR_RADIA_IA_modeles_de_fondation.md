# 🤖 Fiche de cours — De l'intelligence artificielle aux modèles de fondation (GDR RADIA)

> Source : `GDR RADIA.pdf` (conférence de 125 pages, 64 diapos numérotées, par V. Guigue). **Format conférence généraliste** : la fiche suit l'ordre des diapos. Les frises historiques et dates ont été omises (consigne). Légende : 💡 idée clé · ⚠️ vigilance · 🚩 piège fréquent · 🔎 complément (hors cours) · 📝 mes notes (aucune fournie).

## Sommaire
1. [De l'IA au machine learning](#1-de-lia-au-machine-learning)
2. [Deep learning et texte (NLP)](#2-deep-learning-et-texte-nlp)
3. [Transformer et modèles pré-entraînés](#3-transformer-et-modèles-pré-entraînés)
4. [Les ingrédients de ChatGPT](#4-les-ingrédients-de-chatgpt)
5. [Utilisations](#5-utilisations)
6. [Limites](#6-limites)
7. [Risques](#7-risques)
8. [De l'IA générative aux modèles de fondation](#8-de-lia-générative-aux-modèles-de-fondation)
9. [Outils, métiers, souveraineté, conclusion](#9-outils-métiers-souveraineté-conclusion)
10. [Quiz d'auto-test](#10-quiz-dauto-test)
11. [🚩 Pièges à éviter](#11--pièges-à-éviter)

---

## 1. De l'IA au machine learning

**Ingrédients de l'IA** : **modèles**, **logiciels**, **données**, **capteurs**, **stockage et calcul**.

**Définitions**
- **IA** : programmes informatiques qui s'adonnent à des tâches qui sont, pour l'instant, accomplies de façon plus satisfaisante par des êtres humains car elles demandent des processus mentaux de haut niveau.
- **N-AI** (*Narrow AI*, dédiée à **une** tâche) ≠ **G-AI** (*General AI*, qui remplace l'humain dans des systèmes complexes).

**Place de l'IA dans le numérique** : Numérique ⊃ Données ⊃ **Machine learning** ⊃ **Deep learning / réseaux de neurones**. Exemples : caisse automatique du supermarché, Google Maps, **système prédictif** (marché immobilier), **recommandation**.

**Cadres en ML** : apprentissage **supervisé**, **semi-supervisé**, **non supervisé**, **par renforcement** (décision ↔ récompense). Modalités de données variées (images, textes, données numériques…), étiquetages variés. **Types de prédictions** : interpolation/prévision (*forecasting*), classification, régression, *ranking* ; + **IA générative**.

**Chaîne de traitements supervisée** : jeu de données (ex. objets vert/rouge) → **extraction de caractéristiques** (diamètre, texture, niveau de rouge/bleu…) → **apprentissage** → **classifieur** → **inférence**. **Promesse = construire un modèle uniquement à partir des observations.**

**Deep learning** : on part de l'**image brute**, avec des **paramètres de combinaison** initialisés **aléatoirement** (donc décision aléatoire au début) ; **rétropropagation du gradient** : comparaison avec la **vérité terrain**, **mise à jour des poids pas à pas**, nombreuses itérations sur les données.
> 💡 **Apprentissage lent et coûteux ; inférence (beaucoup plus) rapide.** Training = *intensive computing* ; exploitation = calcul limité.

**Étapes en ML** : données → **sélection de modèle** / **réglage des paramètres** (*parameter tuning*) / optimisation → **meilleur modèle** → **industrialisation**.

> ⚠️ **Mesurer les performances** (estimer la performance **en généralisation**) est **aussi important que l'apprentissage du modèle lui-même** : comparer prédiction et vérité terrain sur des **nouveaux documents** pour estimer les performances réelles.

---

## 2. Deep learning et texte (NLP)

**Du tabulaire au texte** : tabulaire = **dimension fixe**, **valeurs continues** ; texte = **longueur variable**, **valeurs discrètes** (« *this new iPhone, what a marvel* » vs « *An iPhone? What a scam!* »).

**NLP** = la plus grande communauté scientifique en IA.

| Approche | Caractéristiques |
|---|---|
| **Systèmes à base de règles** (linguistique) | **expertise** requise ; extraction de règles ⇔ **données très propres** ; **précision très élevée**, **rappel faible**, **système interprétable** (ex. {like, love, appreciate} + #produit ; négations {didn't, not, …}) |
| **Machine learning** | peu d'expertise ; extraction statistique ⇔ **robuste au bruit** ; **≈ moins interprétable**, **précision plus faible**, **meilleur rappel** |

> 💡 **La précision est le critère d'acceptation par l'industrie** → lien avec les métriques.

### Représentation : des sacs de mots aux vecteurs
- **Sac de mots** one-hot : *car*, *vehicle*, *cat* → **même distance** entre tous.
- **Espace vectoriel continu** : **similarité ++, distance ++**. Exemple : à partir de phrases contenant *cat* (« The fluffy cat napped… », « My cat loves to chase… »), le vecteur de *cat* est construit à partir des **mots de contexte** (fish, mouse, garden, fur, birds…), à l'écart de mots sans lien (*pineapple*, *bicycle*, *telescope*).
- **Espace sémantique** : sens proche → position proche (cats/dogs, good/bad/better/worse) ; **espace structuré** : **régularités grammaticales**, connaissances de base (Italy–Roma, France–Paris, man–woman, king–queen).

### Des mots aux tokens
**Découpage statistique en sous-mots (*word piece*)** : représentation des **mots inconnus**, **adaptation aux domaines techniques**, **résistance aux fautes d'orthographe**.

### Agrégation des représentations de mots : vers l'IA générative
**Génération et représentation** : prédire un mot à partir de la phrase (*The fluffy cat napped **lazily** in the sunbeam*) : couche de représentation → couche cachée → **couche de prédiction sur le vocabulaire** → **perte** par rapport à la vérité terrain → **rétropropagation**. Entraînement sur un **corpus** de phrases.

**Inférence et beam search** : traduction *« It's raining cats and dogs »* → encodeur (RNN/Transformer) → décodeur qui prédit **token par token** (« pleut »…). **Coût élevé : ≈ 1 appel par token.** Principe du **maximum de vraisemblance** ; la tâche NLP historique = architecture de classification/scoring spécifique ; ici **contrainte et/ou post-traitement sur une architecture générative**.

**Cas d'usage : traduction automatique** — critère 1 : **régénérer la phrase** ; au-delà du mot à mot : **représentation multilingue des phrases** ; critère 2 : **aligner les représentations** (*« It's raining cats and dogs »* ≈ *« Es regnet Bindfäden »* ≈ *« Il tombe des cordes »*).

---

## 3. Transformer et modèles pré-entraînés

**Transformer, agrégation à l'état de l'art** : RNN : $h_{t+1}=h_tW_1+x_{t+1}W_2$ (dépendance séquentielle) vs **Transformer** : **matrice de self-attention** + couche **fully connected** sur des **embeddings de tokens** (*« it's raining cats and dogs »*), en **blocs empilés** ; décodeur avec **cross-attention** vers l'encodeur et **tête de prédiction** du mot.

**Nouveau paradigme de développement** : *énorme jeu de données + énorme architecture ⇒ coût d'entraînement déraisonnable* → **architecture pré-entraînée + 0-shot / fine-tuning**.
- **Pré-entraînement** (corpus massif, ex. tâche « *It's raining MASK and PRED* ») → **modèle de langue pré-entraîné** → représentations de mots et de textes.
- **Fine-tuning** sur **vos (petites) données** : le modèle adapté vise la cible attendue (la diapo indique quelques pour cent du corpus pour l'adaptation).
- **Modèle de langue multi-tâches** : entraîner sur plusieurs tâches NLP (corpus étiquetés) ; pour une **nouvelle tâche** : **0/few-shot learning** (0 à 10 exemples).

---

## 4. Les ingrédients de ChatGPT

**0. Transformer + données massives (GPT)** : **pré-entraînement causal** (*JFK died in…* → *1963, he was assassinated in Dallas*) sur un **énorme corpus filtré** ; acquisition de **compétences grammaticales** (accord singulier/pluriel, concordance des temps) et de **connaissances**.

**1. *More is better !*** (taille du modèle) : plus de mots en entrée (500 → 2k, 32k, 100k), plus de **dimensions** dans l'espace des mots (500–2k → 12k), plus de **têtes d'attention** (12 → 96), plus de **blocs/couches** (5–12 → 96).
> **175 milliards de paramètres** : $1{,}75\cdot10^{11}$ ⇒ ≈ **300 Go** (+ 100 Go de stockage de données pour l'inférence) ≈ **400 Go** ; une **GPU NVIDIA A100 = 80 Go** (≈ 20 k€) ; **coût d'entraînement ≈ 4,6 M€**.

**2. Suivi de dialogue** : entraînement spécifique sur un **corpus de dialogues** (suivi de dialogue, résolution de coréférences, façon de parler) ; **données très propres**, **générées/validées/classées par des humains**.

**3. Fine-tuning sur des tâches de raisonnement plus ou moins complexes** (*instruction finetuning*) : ~**1 800 tâches** étiquetées, regroupées par catégories (classification d'opinion, génération de texte, résumé, raisonnement arithmétique, questions-réponses, traduction, génération de code, …) ; **CoT** (*chain of thought*, *« let's reason step by step »*) ; plus il y a de tâches de fine-tuning, plus la performance sur des tâches non vues augmente (modèles 8B/62B/540B).

**4. Instructions + classement de réponses (RLHF)** : le modèle génère **plusieurs réponses** à une question ; des **humains les classent** (base de données créée par des humains) ; un **modèle de score** est appris puis utilisé pour **améliorer le modèle par apprentissage par renforcement (PPO)**.
> ⚠️ C'est aussi un moyen d'**éviter les sujets critiques = censure**.

**Étapes et performance** : *données massives ⇒ données de qualité (dialogue) ⇒ tâches ⇒ RLHF*.

### Usage de ChatGPT et prompting
**Demander à ChatGPT = compétence à acquérir ⇒ le *prompting*** :
- bien poser la question : *en détail*, *étape par étape* ; **préciser le nombre d'éléments** (3 qualités pour…) ; **fournir le contexte** (*cellule* pour un biologiste / assistant juridique) ;
- **ne pas s'arrêter à la première question** : détailler des points, rediriger la recherche (dialogue) ; **reformuler** : *explique comme si j'avais 5 ans*, *comme un article scientifique*, *style bro* ; **résumer, développer** ; **ajouter des erreurs (!)** ;
- **besoin de pratique (1 à 2 heures)**, discuter avec des collègues.

**Vers le few-shot learning** : apprendre **sans modifier le modèle** = **exemples dans le prompt**.

**GPT-4 et multimodalité** : fusion de l'information texte + image ; apprendre à exploiter conjointement l'information ; exemple de la **VQA** (*Visual Question Answering*) ⇒ **rétropropager l'erreur** ⇒ modifier les représentations de mots **et** l'analyse d'image.

**Pourquoi tant de controverses ?** Nouvel outil, **vitesse d'adoption sans précédent** ; forces et faiblesses **mal comprises** des utilisateurs ; **gains de productivité** significatifs ; usages surprenants/parfois absurdes ; **biais**, usages dangereux, risques ; retours mal interprétés ; **anthropomorphisation** de l'algorithme et de ses erreurs ; **coût prohibitif** : quel modèle économique, écologique, sociétal ?

---

## 5. Utilisations

### LLM et recherche d'information
- Moteur classique : **requête** (vocabulaire) → **index** → métriques (ex. **BM25**) → **liste classée de documents**.
- **Interroger ChatGPT pour de l'information** : usage surprenant, mais **est-ce raisonnable ?** (*vraie question ouverte*) : modèle **hors ligne**, **pas d'index ni de sourçage**, **génération mot à mot**, **aucune garantie, aucune source**.
- **+ documents à analyser (= contexte)** : requête web + analyse, résumé automatique, reformulation, comptes rendus de réunion… **Limite actuelle** de taille d'entrée (2k puis 32k tokens) ; équivalent d'un LLM pour la **question-réponse** d'avant ChatGPT.
- **RAG (*Retrieval Augmented Generation*)** : **mix extraction/génération** : Intranet/Internet + **sourçage** comme en QA.

### Autres usages de l'IA générative
- **Outil de mise en forme** : formatage, langue… *pas de nouvelles idées* ; **assistant personnel** (lettres types, de recommandation, de motivation, de licenciement), traductions, comptes rendus, mise en forme de notes, **rédaction d'articles scientifiques** (idées, en français/anglais), **analyse de documents** (extraction d'information, QA).
- **Outil de réflexion** : accès à l'information (risqué mais pratique), **brainstorming** (développement d'arguments, recherche de contradictions), **assistant de développement** (génération de code, recherche d'erreurs, documentation), **assistant pédagogique** (Wikipédia++, propositions de plans, explication/correction de code).
- **Enseignement** : un **enseignant disponible 24h/24**, en particulier pour le code (apprendre Python, le machine learning) : 1) générer un petit programme, 2) poser des questions sur les fonctions. *Le LLM peut faire vos devoirs… mais il peut aussi vous les expliquer, répondre aux questions sur la solution, vous enseigner !*

### IA générative vs IA classique
L'IA **classique** reste pertinente (et souvent supérieure) pour : **séries temporelles / maintenance prédictive**, **prédiction de prix** (voitures, immobilier), **diagnostic médical sur données numériques** (EEG, ECG), **systèmes de recommandation** (rappel : les modèles à arbres surpassent souvent le deep learning sur les **données tabulaires**).

### IA embarquée
Bracelet connecté/vêtements/lunettes (séries temporelles, diagnostic, anomalies : médecine ou gadget ?), **assistant intelligent / chatbot**, **compteur intelligent**, **cabine de télémédecine**, **robotique**, smartphone…

### Jumeaux numériques et PINNs
- **Jumeau numérique** : optimiser en **temps réel** les décisions de gestion du jumeau réel (lien capteurs/actionneurs) ; réaliser des **expériences numériques** pour tester les conséquences avant de les mettre en œuvre.
- **Types de modèles** : **mécaniste/simulation** (conditions aux limites, calibration) vs **piloté par les données** (entraînement).
- **PINNs** (*Physics-Informed Neural Networks*) : combiner modèles mécanistes et données ; **contraintes physiques / équations différentielles modélisées dans l'architecture neuronale** (ex. *Neural ODE*).
- **Architectures hybrides** : modèle mécaniste composite géant (lent/coûteux, précis) + **composant faible** (hypothèses insuffisantes) ; approximation **rapide** ; **données de simulation, réelles, générées**.

---

## 6. Limites

**ChatGPT et le rapport à la vérité**
1. **Plausibilité** = grammaire, accords, concordance des temps, enchaînements logiques ⇒ **connaissances répétées** ;
2. **Prédit le mot le plus plausible** ⇒ produit des **hallucinations** ;
3. Fonctionnement **hors ligne** ;
4. ChatGPT ≠ **graphes de connaissances** ;
5. **Réponses brillantes… et bêtises !** + on **ne peut pas prédire les erreurs** (ex. produire une bibliographie).

**Évaluer l'IA générative** (*le point critique aujourd'hui*) : comment évaluer **par rapport à une vérité terrain** ? comment évaluer la **confiance / plausibilité** de la génération ? Mesures de **plausibilité** (« les LLM savent-ils ce qu'ils ne savent pas ? » ; train vs test).

**Stabilité / prévisibilité** : difficile de **borner un comportement** ; impossible de prédire les bonnes/mauvaises réponses ⇒ **peu/pas d'usage dans les jeux vidéo** ; **stabilité, explicabilité… et complexité** : *système simple* (capteurs → règles simples → feu clignotant) = **test exhaustif** des entrées/sorties, **prévisible et explicable** ; *système LLM* (vocabulaire énorme → séquence de mots (combinaison) → agrégation → prédiction de mot) = **grande dimension**, **combinaisons non linéaires complexes**, **non prévisible et non explicable**.

**Interprétabilité vs explication post-hoc** : réseaux de neurones = **non interprétables** (presque toujours) — trop de combinaisons pour les anticiper ; mais **explicables a posteriori** (presque toujours) — (exemple d'accident cité : véhicule autonome, 2018).

**Transparence** : poids du modèle (*open-weight*) **mais pas seulement** : **données d'entraînement** (ex. BLOOM) + distribution + instructions ; **techniques d'apprentissage** ; **évaluation**. (Indice de transparence des modèles de fondation : dimensions *données, calcul, main-d'œuvre, capacités, risques, atténuations, usage…*.)

---

## 7. Risques

**Typologie des risques de l'IA en NLP** (Weidinger).

- **Accès à l'information** : informations **dangereuses/interdites** ; **données personnelles** (**droit à l'oubli numérique**) ; **autorités d'information** : *nature* (l'image = vérité), *source* (journaux, réseaux sociaux), *volume* (nombre de variantes, citations, pagerank) ; **génération de texte** : harcèlement ; **anthropomorphisation** ; distinguer humain / machine.
- **Biais** : **biais dans les données ⇒ biais dans les réponses** ; le ML repose sur l'**extraction de biais statistiques** ⇒ **lutter contre le biais = ajuster manuellement l'algorithme** (ex. « moustache, oreilles triangulaires, fourrure » → chat ; « plus de 40 ans, blanc, rasé, costume » → cadre dirigeant) ; stéréotypes (genre, couleur de peau, posture…).
- **Correction des biais et ligne éditoriale** : sélection de données, rééquilibrage ; **censure** de certaines informations/résultats ⇒ **travail éditorial** — fait par qui ? (experts du domaine/spécifications, ingénieurs, groupe éthique, communication) ⇒ **quelle légitimité, quelle transparence, quelle efficacité ?**
- **Le ML n'est jamais neutre** : (1) **sélection des données** (sources, équilibre, filtrage) ; (2) **transformation des données** ; (3) **connaissances a priori** (équilibre, perte, choix d'opérateurs) ; (4) **filtrage des sorties** (post-traitement) ⇒ choix qui influencent les résultats.
- **Fuites de données** : requête (+ documents) **enregistrée** ; transfert de **données sensibles** vers un serveur US, exploitation par l'éditeur, **fuite dans les modèles futurs**.
- **Sécurité** : **plug-ins** (accès aux emails, transfert d'informations sensibles) ; gestion en entreprise (sécuriser de très gros fichiers) ; **malware** : signatures facilement contournées par **reformulation** (génération directe de malware).
- **Questions juridiques** : collecte/lecture/formatage ; stockage (temporaire ou permanent) de **documents, données personnelles, données de santé** ; **entraînement** (droit d'utiliser les données dans un algorithme, opt-out, consentement) ; **modèle = fonction mathématique** ⇒ *émanation des données ?* (droit d'auteur, droit des bases de données) ; **inférence** (responsabilité des erreurs, reproductions d'extraits intraçables).
- **Questions économiques** : financement/publicité ⇔ **visites des internautes** ; ChatGPT = **encodage de l'information du web** ⇒ beaucoup moins de visites ? **Quel modèle économique pour les sources d'information ?** Qui bénéficie des retours (ex. StackOverflow) ?
- **Risques de généralisation de l'IA** : *l'IA partout = perte de sens ?* — **éducation** (rédaction, plans, idées / évaluation automatique, résumé / plans, quiz, illustrations ⇒ **vérification de l'usage de l'IA**) ; transposition aux **RH** et aux **financements sur projets**.
- **Éthique** (parallèle **médecine ↔ IA**, 8 principes) :
| Principe | En médecine | En IA |
|---|---|---|
| Autonomie | patient éclairé | **les humains contrôlent le processus** |
| Bienfaisance | faire le bien | y compris pour **l'environnement** ? |
| Non-malfaisance | éviter de nuire | humains + environnement / **durabilité** / usages malveillants |
| Justice | équité d'accès aux soins | **accès à l'IA** et égalité des chances |
| Confidentialité | des patients | **que dire du modèle économique de Google/Facebook ?** |
| Vérité et transparence | information honnête | « **la tragédie de l'IA moderne** » |
| Consentement éclairé | des patients | **des cookies aux algorithmes** : savoir quand on interagit avec une IA |
| Respect de la dignité | traiter avec respect | (à discuter) |

---

## 8. De l'IA générative aux modèles de fondation

### Origine de la modélisation statistique
1. **Observer** les données (et le contexte) ; 2. **Modéliser** = choisir un modèle probabiliste / réseau bayésien ; 3. **Optimiser** les paramètres (maximum de vraisemblance, EM, BFGS…) ; 4. **Échantillonner / inférer** et **évaluer les distances** existant vs échantillonné.

### Origine du deep learning
- **Disparition du gradient** dans les architectures profondes (le gradient s'affaiblit en remontant) ;
- **Auto-encodeur** : face à un jeu **non supervisé**, **réduction de dimension**/apprentissage de représentation (≈ PCA, SVD) ; **débruitage** (*denoising*) ;
- **Stacked Denoising Auto-Encoder** : entraînement **itératif / pré-entraînement** couche par couche ;
- **Variational Auto-Encoder (VAE)** : ajoute du **bruit**, paramètres de **distribution**, **échantillonnage**, un **a priori sur la distribution** : **structuration de l'espace latent** = « IA générative pour statisticiens ».

### Différentes formes d'IA générative
**Encodeur → vecteur compact (représentation) → décodeur** : 1) **encoder** une entrée = construire un vecteur ; 2) **décoder** un vecteur = générer une sortie.

| Média | Architecture / problème |
|---|---|
| **Textes** | **problème de classification** (token par token) : RNN/Transformer |
| **Images** | **régression multivariée** : **U-Net** (réseaux convolutifs pour la segmentation) |
| **GAN** (*Generative Adversarial Networks*) | **détecter les échantillons générés** (générateur vs discriminateur) |
| **Processus physiques** | **modèles de diffusion** (débruitage progressif), génération d'images conditionnée par du texte (CLIP) |
| **Structures complexes (3D, graphes)** | problème séquentiel ; mélange approches mécanistes/données (équations différentielles dans le réseau : Neural ODE, PINNs) |
| **Apprentissage par renforcement** | action/récompense (ex. structure des protéines) |

### Multimodalité
**Construction d'espaces de représentation multimodaux = *grounding*** : **image ⇒ texte** (légendage, VQA), **texte ⇒ image** (Midjourney, DALL-E) ; deux **encodeurs** (image, texte) + **alignement des espaces de représentation** ; **inpainting** d'image (masque), **question answering**, prédiction de mots masqués.
- **Vers des modèles de fondation plus larges** : laisser les modalités **s'enrichir mutuellement**.
- **Principaux défis** : **nouvelles applications** à l'interface texte/image/musique/voix ; **amélioration des performances** (meilleur encodage, désambiguïsation, contexte) ; **explicabilité** (par le dialogue) ; IoT / systèmes de recommandation / véhicule intelligent.

---

## 9. Outils, métiers, souveraineté, conclusion

**Outils et questions** : de **nouveaux outils** : nouvelles façons de traiter des problèmes existants, **adresser de nouveaux problèmes** — **mais ça ne marche pas toujours !** L'IA fait souvent des erreurs (**assistant vs remplacement**) ; **apprendre à utiliser** un système d'IA ; IA **inadaptée** à de nombreux problèmes ; **l'IA = une partie du problème** (+ interface, usage, acceptation).

**Maturité des outils et environnements**
| Domaine | Outils | Formation | Réalisation d'un projet |
|---|---|---|---|
| **ML** | scikit-learn : **blocs à assembler** (Jupyter, VS Code) | ≈ 1 semaine | quelques heures à quelques jours |
| **Deep learning** | PyTorch, TensorFlow : briques mais **plus complexes** ; **indispensable pour texte et image** | 2 à 5 semaines | quelques jours à quelques mois |

Un projet de données = **10 à 100× moins de temps qu'en 2005** ; accessible aux **non-informaticiens**.

**Trois niveaux d'accès à l'IA** : **1)** utilisateur via une interface (ChatGPT) — formation de 2–4 h ; **2)** utilisation de **bibliothèques Python** — bases sur les protocoles, chaînes de traitement standard, 1 semaine à 3 mois (ML/DL) ; **3) développeur d'outils** — adapter à un cas, intégrer des contraintes métier, **systèmes hybrides** (mécaniste/symbolique), mélanger texte et images — ≥ 1 an.

**Souveraineté numérique : toute la chaîne**
1. **Construction du modèle pré-entraîné** : maîtrise des **données** (collecte/équilibre, nettoyage), **entraînement** (puissance de calcul ×1000 GPU, architecture ML) ;
2. **Fine-tuning** : maîtrise et construction de **données** (interactions humaines +++, coût du jeu de données, adaptation au domaine), **structuration**, suivi de dialogue ;
3. **Exploitation** : optimisation / réduction des coûts (**MLOps**, déploiement local), **industrialisation**.

**Une multitude de métiers** : **data architect/manager** (stockage, réseau) · **data engineer** (mise à jour et requêtes) · **data analyst** (visualisation, tendances statistiques) · **prompt engineer** (requêtes sur LM/modèles de fondation) · **data scientist** (requêtes, sélection critique/équilibre, développement/adaptation/évaluation d'algorithmes, visualisation avancée) · **MLOps engineer** (optimisation, industrialisation) · + **DPO** (*Data Protection Officer*).

**Facteurs d'acceptabilité de l'IA générative** : 1) **utilitarisme** (performance, fiabilité/auto-évaluation) ; 2) **non-dangerosité** (biais/correction, transparence — ligne éditoriale, confusion humain/machine —, mise en œuvre fiable, **souveraineté**, **régulation** (AI Act), éviter les applications dangereuses) ; 3) **savoir-faire** (formation à l'usage et au développement).

**ChatGPT : une étape simple** : coûts d'entraînement/réglage **4–5 M€ par entraînement** (⇒ « ChatGPT est mal entraîné ! ») ; **efficacité des données** : ChatGPT lit **> 1000 fois** ce qu'un humain lit dans sa vie ; **identifier des entités, citer des sources** (ancrer les réponses dans des bases de connaissances et dans des sources) ; **multiplication d'initiatives** (GPT, LaMDA, PaLM, BARD, BLOOM, Gopher, Megatron, OPT, Ernie, Galactica…) ; **implication du public**, impact sur l'accès à l'information.

---

## 10. Quiz d'auto-test

> Barème indicatif sur 30 : Q1–Q4 (4 pts) · Q5–Q8 (3 pts) · Q9–Q12 (2 pts).

<details>
<summary><b>Question 1 (4 pts) :</b> Quels sont les quatre grands ingrédients de ChatGPT, dans l'ordre ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
0) Transformer + données massives (pré-entraînement causal, GPT) ; 1) *More is better* (plus de paramètres/mots/têtes/couches : 175 Mds de paramètres) ; 2) entraînement au **suivi de dialogue** (données propres, validées par des humains) ; 3) **fine-tuning sur des tâches** de raisonnement (instructions, CoT) ; 4) **instructions + classement de réponses (RLHF/PPO)**.
</details>

<details>
<summary><b>Question 2 (4 pts) :</b> Pourquoi un LLM produit-il des hallucinations ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Il **prédit le mot le plus plausible** (grammaire, accords, enchaînements logiques) à partir de connaissances répétées dans le corpus, fonctionne **hors ligne**, sans index ni source, et n'est pas un graphe de connaissances : plausible ≠ vrai ; les erreurs ne sont pas prévisibles.
</details>

<details>
<summary><b>Question 3 (4 pts) :</b> Qu'est-ce que la RAG et pourquoi ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
*Retrieval Augmented Generation* : on **récupère des documents** (Intranet/Internet) puis on génère une réponse à partir de ce contexte, avec **sourçage** (mix extraction/génération) ; cela contourne l'absence d'index/de sources du LLM seul et la limite de taille d'entrée.
</details>

<details>
<summary><b>Question 4 (4 pts) :</b> Pourquoi dire que « le ML n'est jamais neutre » ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Quatre niveaux de choix : sélection des données (sources, équilibre, filtrage), transformation des données, connaissances a priori (perte, opérateurs), filtrage des sorties. Les biais des données se retrouvent dans les réponses ; les corriger = un travail éditorial (qui, avec quelle légitimité ?).
</details>

<details>
<summary><b>Question 5 (3 pts) :</b> Comparer systèmes à règles et ML en NLP. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Règles : expertise, données propres, précision très élevée, rappel faible, interprétable. ML : peu d'expertise, robuste au bruit, moins interprétable, précision plus faible, meilleur rappel. La précision est le critère d'acceptation industriel.
</details>

<details>
<summary><b>Question 6 (3 pts) :</b> Que signifient 0-shot, few-shot et fine-tuning ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
0-shot : tâche nouvelle sans exemple ; few-shot : quelques exemples **dans le prompt** (sans modifier le modèle) ; fine-tuning : on ré-entraîne (un peu) le modèle pré-entraîné sur ses données.
</details>

<details>
<summary><b>Question 7 (3 pts) :</b> Interprétabilité vs explication post-hoc ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Les réseaux de neurones sont presque toujours non interprétables (trop de combinaisons à anticiper) mais explicables **a posteriori** (saillance, LIME…). Un système simple à règles est prévisible et testable exhaustivement.
</details>

<details>
<summary><b>Question 8 (3 pts) :</b> Donner les ordres de grandeur d'un modèle de 175 Mds de paramètres. <i>(Cliquez ici pour la réponse)</i></summary>
<br>
$1{,}75\cdot10^{11}$ paramètres ⇒ ≈ 300 Go (+100 Go) ≈ 400 Go ; GPU A100 = 80 Go (≈ 20 k€) ; entraînement ≈ 4,6 M€.
</details>

<details>
<summary><b>Question 9 (2 pts) :</b> Pourquoi l'IA classique reste-t-elle utile malgré l'IA générative ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Pour les séries temporelles/maintenance prédictive, prix, diagnostic sur signaux (EEG, ECG), recommandation : l'IA classique (souvent à base d'arbres sur le tabulaire) est souvent meilleure.
</details>

<details>
<summary><b>Question 10 (2 pts) :</b> Qu'est-ce qu'un PINN ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
*Physics-Informed Neural Network* : un réseau où les **contraintes physiques / équations différentielles** sont modélisées dans l'architecture, combinant modèle mécaniste et données.
</details>

<details>
<summary><b>Question 11 (2 pts) :</b> Les trois niveaux d'accès à l'IA ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Utilisateur via interface (2–4 h de formation) ; utilisateur de bibliothèques Python (1 semaine–3 mois) ; développeur d'outils (≥ 1 an).
</details>

<details>
<summary><b>Question 12 (2 pts) :</b> Pourquoi parler de souveraineté « sur toute la chaîne » ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
Parce que la maîtrise doit couvrir la construction du modèle pré-entraîné (données, calcul), le fine-tuning (données humaines, domaine) et l'exploitation (MLOps, déploiement local, coût).
</details>

---

## 11. 🚩 Pièges à éviter

- **Plausible ≠ vrai** : hallucinations ; toujours vérifier et exiger des sources.
- **Anthropomorphiser** l'algorithme et ses erreurs.
- **Fuite de données** : ne pas envoyer de données sensibles à un service externe.
- **Biais** : les données biaisées produisent des réponses biaisées ; corriger = acte éditorial.
- **Évaluer** : l'estimation de la performance **en généralisation** est aussi importante que l'apprentissage.
- **Deep learning ≠ solution universelle** : tabulaire/séries/maintenance → méthodes classiques souvent meilleures.
- **Coût** (entraînement, écologique, économique) et **dépendance** : souveraineté.
