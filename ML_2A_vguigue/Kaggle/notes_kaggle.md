



  
### Problèmes

Le fait d'encoder `diabete_type` avec label encoder qui transforme le type en chiffre, avec une "hierarchie" vu que 0<1<...<n-1 ne biaise pas le résultat en forcant un ordre dans les targets ? Vaut-il mieux utiliser LabelEncoder ou OrdinalEncoder (différence et le plus adapté a ma situation)

J'ai mis seulement 2000 lignes uniquement pour le debug, car sinon les cellules sont longues a tournées. Mais je le fait bien sur l'entiereté quand je veux voir les performances et comparer, les 2000 lignes sont juste pour voir si les cellules tournent bien.

Est-ce qu'il faut toujours supprimer les variables fortement corélées ou elles peuvent avoir parfois du sens de garder les deux ? Dans mon cas lesquels sont interessantes a suppirmer ? 
Est-ce qu'il est judicieux de supprimer toutes les colonnes "risk_score" car elles ne sont pas très "cohérente" avec l'objectif du modèle ? C'est à dire donne des infos qu'on aurait pas forcement dans la vie ?

Comment savoir quel variable represente la separation d'un nuage de point ?

Pour les variables quasi pas corélées, estce qu'on rajoute dans le clean un point spécifique au modèle. Par exemple on laisse les très peu dans les modeles a arbres mais on enlever pour les autres ? ou on regarde du coté des poids, si c'est quasi nul on considere que le modele ne s'appuie pas dessus donc on enleve la colonne car non utilisée ?

## Workflow

**Workflow récurent en pro et sérieux**  

1. Filtrage initial  
Retrait des variables doublons, fuites de données et variables indisponibles au moment de la prédiction.  

2. Filtrage avancé  
Retrait des variables constantes, trop corrélation entre X ou faible corr en X et y ...

3. Comparaison de plusieurs sélections  
Méthodes statistiques, régularisation et importances de modèles.  

4. Validation des performances  
Comparaison avec validation croisée, selon une métrique adaptée au besoin métier.  

5. Évaluation finale indépendante  
Évaluation sur le jeu de test, utilisé uniquement après les choix de modélisation.