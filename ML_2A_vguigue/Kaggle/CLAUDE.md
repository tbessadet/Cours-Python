# CLAUDE.md (projet Kaggle : diabète)

## Contexte
- Projet académique, je débute en ML. L'objectif est de comprendre la démarche (nettoyage, sélection de variables, validation), pas d'avoir le meilleur score.
- Source : https://www.kaggle.com/datasets/datascikhan/diabetes?resource=download
- Dataset : environ 100 000 patients, plus de 180 variables regroupées en 22 catégories (démographie, biologie, mode de vie, alimentation, environnement, antécédents, scores).
- Cible principale : `diabetes_status` (0 = non, 1 = oui). Environ 28 % de diabétiques, donc classes déséquilibrées.
- Cible secondaire : `diabetes_type` (No Diabetes, Type 1, Type 2, Gestational, Other). Très déséquilibrée : Type 1 et Gestational sont quasi absents.
- Métriques utilisées : F1, ROC AUC, average precision (AP).
- Objectif actuel : prédire le diabète sans diagnostic, uniquement avec des analyses sanguines et des données simples, sans intervention d'expert, pour l'appliquer au plus grand nombre de patients.

## Fichiers
- `kaggle_clean.ipynb` : notebook principal (préparation, sélection de variables, modèles, test final). `kaggle_clean_avant_revue.ipynb` : sauvegarde avant la revue du 29/10/2026.
- `diabetes_large_dataset.csv` : données brutes. Ne pas modifier.
- `archive.zip` : archive d'origine. Ne pas lire.
- `annexes.py` : importé par le notebook (`ArrondirModalites`, `ColumnFilter`). Ne pas le lire. Ne pas le supprimer sans accord : le notebook ne tourne plus sans lui.
- `notes_kaggle.md` : mes questions et mes notes. Je les complète moi-même.
- `cache_prep/`, `cache_selection/`, `__pycache__/` : caches générés par le notebook. Ne pas lire ni modifier.
- `save/` : graphiques sauvegardés.

## Règles du notebook
- Me demander avant de lancer un calcul long (sélection, réglage des hyperparamètres).
- Le jeu de test (20 %, `random_state=42`, fonction `split_train_test`) est mis de côté. Ne jamais l'utiliser pour choisir un modèle, une variable ou un hyperparamètre : il ne sert qu'en partie 7.
- Validation : `StratifiedKFold` (`N_SPLITS = 10` dans la partie sélection et la suite).
- Cache des plis : `cache_selection/folds_*.joblib` doit être supprimé si je modifie `to_drop`, `binaires`, `ordres`, `onehot` ou `Clean`.
- Fuites de données : les variables de diagnostic, de traitement et de suivi du diabète sont retirées en partie 1.1 (dictionnaire `motifs`). Ne pas les réintroduire.
- Graphiques : chaque graphique est dans sa propre cellule et passe par `montrer(fig, nom)`, qui l'affiche et l'enregistre dans `save/`.

## Questions ouvertes (à discuter, ne pas trancher seul)
- Encodage de `diabetes_type` : LabelEncoder (utilisé dans `Clean`) ou OrdinalEncoder ?
- `smoking_status` : "Former" entre "Never" et "Current" est-il un ordre valide, ou faut-il un OneHot ?
- Redondances : `smoking_frequency` / `smoking_status`, et `gender` / `sex`.
- Variables fortement corrélées : laquelle garder, et faut-il toujours en supprimer une ?

## Workflow
1. Filtrage initial : retrait des doublons, des fuites de données et des variables non disponibles au moment de la prédiction.
2. Filtrage avancé : retrait des variables constantes, des variables trop corrélées entre elles, ou peu corrélées à la cible.
3. Comparaison de plusieurs sélections : méthodes statistiques, régularisation, importances des modèles.
4. Validation des performances : validation croisée, avec une métrique adaptée au problème.
5. Évaluation finale : jeu de test, utilisé uniquement après les choix de modélisation.

## Avancement
- Préparation : valeurs manquantes traitées (partie 2.1), encodage et imputation (partie 2.2, fonction `Clean`).
- Premiers modèles (AUC en validation croisée, après réduction des variables) : LogisticRegression 0,74 ; LinearSVC 0,74 ; XGBoost 0,74 ; GaussianNB 0,63 (abandonné).
- Sélection de variables (partie 5) : 7 méthodes lancées par `executer` (résultats dans `cache_selection/resultats_full_*.joblib`), puis comparaison des consensus en 5.8. Consensus retenu : `choix` (« Méthodes >= 5 », 27 colonnes), repris par `COLS_FINALES`.
- Prochaines étapes : choisir le meilleur modèle (ou un vote), régler les hyperparamètres, présenter les performances finales avec points forts et faiblesses.
- Décisions prises et pourquoi : [à compléter]
