# Mission

Je te fournis un ou plusieurs notebooks de TD (cours de data science / machine learning). À partir d'eux, fabrique-moi une ou plusieurs **cheat sheets** en Markdown.

## Ce que je veux (et ce que je ne veux PAS)

- ✅ Une **fiche de référence sur les bibliothèques et fonctions utilisées**, appliquées aux sujets que j'étudie. Elle me sert à **revenir rapidement** pour retrouver une fonction, comprendre **quels paramètres comptent**, **comment les régler** pour résoudre mon problème, et éviter les pièges.
- ❌ Pas un résumé du TD, pas un cours, pas un tutoriel pour apprendre à coder. Ne raconte pas les exercices un par un, ne reprends pas les énoncés.

## Méthode de travail

1. **Lis tous les notebooks en entier** (cellules de code ET de texte). Si une version corrigée existe (`-corr`), appuie-toi dessus en priorité : elle contient les vraies solutions.
2. **Dégage l'essence de chaque notebook** : de quoi ça parle, quelle compétence/notion le prof veut qu'on acquière, quelles bibliothèques et fonctions sont mobilisées.
3. **Construis la fiche autour de ça** : reprends principalement ce qui est dans les TD (fonctions, paramètres, démarches), mais **complète** avec ce qui manque s'il est important ou très répandu en milieu professionnel (autres paramètres utiles, fonctions voisines, bonnes pratiques courantes).
4. **Si les sujets sont très différents**, sépare en plusieurs fichiers thématiques (ex. un par bibliothèque ou par grand thème) plutôt qu'un fichier fourre-tout.
5. Si une information du notebook est ambiguë ou incomplète (TODO non résolu, formule seulement décrite), construis la solution correcte toi-même à partir de la description, sans inventer.

## Contenu attendu pour chaque fonction / outil

Pour chaque élément important : la **fonction ou classe**, ses **paramètres essentiels et fréquemment utiles** (valeurs possibles, valeur par défaut, effet concret), et une **explication** : ce qu'elle fait, quand l'utiliser, **comment bien régler les paramètres**, comment lire le résultat. Sois précis sur les paramètres : jamais juste le nom de la fonction.

À inclure aussi :
- **Les pièges** (formes/dimensions, ordre des arguments, fuite de données, valeurs par défaut trompeuses, confusions entre fonctions proches...).
- **Les choses importantes à retenir** et **la manière de faire courante** (le réflexe standard en pratique).
- Quand c'est pertinent : comment choisir entre plusieurs outils proches, et les formules/définitions de base (avec équations en LaTeX, noms en français et en anglais).

## Forme (à respecter strictement)

- Format **Markdown**, rédigé en **français**.
- Titre avec emoji, puis **sommaire cliquable** en haut (liens d'ancre vers chaque section).
- Sections numérotées. Si le document a plusieurs grandes parties, utilise des parties clairement étiquetées (ex. `I. Matplotlib`, `II. Seaborn`) avec leur propre numérotation.
- Tableaux à **trois colonnes** : `Fonction | Paramètres essentiels | Explication` (ajoute une colonne `Import` si utile). Les options secondaires (ex. options de grille) vont **dans la cellule de paramètres** de la fonction concernée, pas dans une section à part.
- **Blocs de code réalistes** illustrant un usage concret, avec commentaires courts.
- Avertissements ⚠️ pour les pièges, 💡 pour les astuces, 👉 pour les conseils d'usage.
- Une section **« 📌 Récapitulatif des pièges classiques »** à la fin de chaque fiche.
- Ne dépasse pas une taille raisonnable par section : dense mais lisible, pas de blocs interminables.

## Livraison

- Livre chaque fiche en **fichier `.md` téléchargeable**.
- Dans ta réponse, indique en quelques lignes ce que contient chaque fiche, sans la recopier.
- Si je te demande ensuite des modifications, donne-moi **uniquement le bloc à ajouter ou modifier** et dis-moi précisément où le placer (et mets le fichier à jour, sommaire et renvois de sections compris).