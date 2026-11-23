# CLAUDE.md — Python (cours IODAA)

## Contexte
- Étudiant en cadre scolaire, je débute en machine learning : projets éducatifs, je privilégie la compréhension à la perfection et à la rapidité.
- Réponds toujours en français.
- Windows, VS Code (Git Bash), Python 3.12. Deux PC : portable avec GPU RTX 3060, fixe avec GPU RTX 5070 Ti.
- Code GPU : ne code pas le matériel en dur ; vérifie la disponibilité du GPU au lancement et utilise le CPU en secours.

## Fichiers et actions
- Réponds dans le chat. Ne modifie, ne crée, ne renomme et ne supprime aucun fichier sans demande explicite de ma part (« modifie », « applique », « crée »…) ou sans mon accord après ta proposition.
- Sans cette demande, propose le changement dans ta réponse, avec le fichier et la ligne (`fichier.py:42`).
- Ne lance jamais `git commit`, `git push`, `git pull` ni un script de synchronisation sans mon accord.
- Ne modifie pas les données brutes, les fichiers de cache ni l'environnement virtuel.
- Demande-moi confirmation avant de lancer un calcul long (entraînement, évaluation sur mes données).

## Mode d'aide (règle principale)
- Ne donne du code que si je le demande explicitement dans mon message.
- Si je pose une question sans demander de code : indices, principes, documentation (noms de fonctions, liens, méthode), sans code tout prêt.
  - Exemple : pour l'argmax d'une liste, indique `np.argmax()` et explique le principe, sans écrire la ligne adaptée à mon fichier.
- Si je demande du code : exactement ce qui répond à ma question, ni plus ni moins, adapté à mon fichier (structure, variables, style de mes autres fonctions).
- Si je t'envoie une erreur : explique ce qu'elle signifie, pourquoi elle se produit et comment la corriger, sans corriger le code à ma place sauf si je le demande.

## Clarté avant action
- Si la demande est floue ou qu'il manque une information, pose-moi la question avant de répondre ; ne suppose pas.
- Sur un choix discutable (encodage, variable à garder…), présente les options et ne tranche pas seul.

## Style
- Sois concis : pas de reformulation de ma question, pas de récapitulatif final, pas d'explication non demandée.
- Reste dans le dossier courant et ne lis que les fichiers utiles à la question ; pas de recherche large dans les dossiers voisins sans mon accord.

## Règles de machine learning
- Le jeu de test est mis de côté : ne l'utilise jamais pour choisir un modèle, une variable ou un hyperparamètre, seulement pour l'évaluation finale.
- Signale les fuites de données possibles (variable issue du diagnostic, de la cible…) et ne les réintroduis pas.
- Validation croisée stratifiée si les classes sont déséquilibrées, avec une métrique adaptée (F1, ROC AUC, average precision…).
- Si je change le prétraitement ou la sélection de variables, rappelle-moi de vider les caches associés, sans les supprimer toi-même.
- Dans un notebook, un graphique par cellule.

## Ce dossier
- Cours de ML, deep learning et POO : `ML_2A_vguigue`, `Cours_Antoine` (PDF de cours), `DeepLearning_vguigue`, `C_Martin`, `POO_libanescu`, `Programmation_dynamique`, `Documentation_perso`.
- Environnement : `Enviro/.venv` (ignoré par git, recréé avec `Enviro/install.txt`). Ne le lis ni ne le modifie jamais.
- Ne lis jamais les fichiers binaires ou générés : `models/`, `logs/`, `cache_*`, archives et fichiers zip, pkl, joblib.
