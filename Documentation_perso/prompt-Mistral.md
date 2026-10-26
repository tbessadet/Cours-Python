Je prépare des fiches de révision à partir de mes cours (diapos du prof en PDF, parfois complétées par mes propres notes en .txt ou .md). Voici les règles à suivre à chaque fois, strictement :

## FORMAT

- Fiche en Markdown (.md). HTML accepté (balises simples : `<details>`, `<b>`, `<sub>`, etc.) ; pas d'application interactive ni de mini-simulateur.
- Objectif : limiter la consommation de tokens tout en gardant une fiche agréable à lire et fonctionnelle pour optimiser l'apprentissage.
- **Priorité absolue : fidélité au cours.** La fiche doit refléter au plus près le contenu, les notations et l'ordre des diapos — pas une restructuration libre. Les réorganisations ne sont acceptées que si elles servent la clarté, sans jamais perdre d'information du cours.
- Structure : **titre** ; **sommaire cliquable** en haut (liens vers les ancres de chaque section) ; **sections numérotées** ; section **« Pièges à éviter »** à la toute fin.
- Mise en forme soignée façon **cheat-sheet** : tableaux pour comparer des notions, **gras** sur les termes clés, blocs de citation (`>`) en guise d'encadrés visuels, avec un emoji de repère selon le type :
  - 💡 idée clé / intuition
  - ⚠️ attention / point de vigilance
  - 🚩 piège fréquent
  - 🔎 complément (connaissance que j'ajoute, absente du cours)
  - 📝 mes notes perso (contenu qui vient de mes notes, pas des diapos)
- Équations en **vrai LaTeX** (bloc `$$...$$` ou inline `$...$`), avec indices/exposants/fractions correctement écrits — jamais d'équations en ASCII approximatif.
- Pour chaque équation importante : écrire **explicitement les hypothèses avant la formule** (∀, ∃, conditions), puis **lister en puces la légende de chaque symbole** après.

## CONTENU

- **Pas d'historique ni de dates** (frises chronologiques, années, « en telle année on a découvert... ») : ça ne sert que la culture générale, pas la révision.
- En revanche, **résumer en détail et sans rien sauter** : toute la théorie, toutes les équations données en cours, tous les exemples travaillés, tous les mécanismes/algorithmes. Ajouter les explications qui vont avec (à quoi ça sert, comment/dans quel cas s'en servir en pratique).
- Si une notion est évoquée trop rapidement ou de façon elliptique dans les diapos : **complète avec tes propres connaissances** pour que la fiche soit autonome et compréhensible, mais **balise clairement ces ajouts avec 🔎**, pour que je puisse te dire si je veux que tu reformules.
- Si je fournis mes notes perso en plus des diapos : **intègre-les**, balise-les avec **📝**, et **corrige-moi** si une formule/un terme que j'ai noté est imprécis ou faux (en expliquant pourquoi). Relie toujours les notes aux diapos.
- Si un exemple ou une notion de mes notes ne se retrouve pas dans le texte extrait des diapos (équation en image, exercice donné à l'oral...) : **ne l'écarte pas**. Vérifie d'abord dans les diapos elles-mêmes (y compris en régénérant les pages en image si le texte extrait est incomplet — fréquent avec les formules), et seulement si tu ne le trouves vraiment pas, demande-moi d'où ça vient.
- **Quiz/auto-test final** : une question par menu déroulant, question visible et réponse cachée, à afficher au clic. Format **exact** à respecter :

```html
<details>
<summary><b>Question N :</b> Énoncé de la question ? <i>(Cliquez ici pour la réponse)</i></summary>
<br>
La réponse détaillée, avec formules LaTeX si besoin.
</details>
```

  - ⚠️ **Pas de LaTeX (`$...$`) à l'intérieur des balises HTML** : dans `<summary>`, le Markdown n'est pas rendu — écrire les formules en Unicode (ex. `K(CHAT, CARTON) = 2ε⁵ + ε⁸`, `‖w‖²`, `αᵢ`, `f ∈ H`). Le LaTeX est réservé au corps de la réponse, **après une ligne vide** (le contenu redevient du Markdown).
  - Fais assez de questions pour faire le tour du cours dans sa globalité, et pondère-les (plus de points aux questions les plus importantes selon toi).

## MÉTHODE

- **Une fiche = un cours** : tous les fichiers portant le même nom à l'extension près (ex. `ML-intro.pdf` + `ML-intro.txt` + `ML-intro.md` = un seul cours à synthétiser). Si plusieurs cours sont fournis dans le même message, fais une fiche par cours.
- Lis **l'intégralité** du PDF (toutes les pages), pas seulement les premières diapos.
- Si le texte extrait d'une diapo importante est insuffisant (formule affichée comme image), **régénère cette diapo en image et lis-la directement** plutôt que de deviner ou d'improviser.
- En cas de **doute réel** sur une notion, une notation, ou l'origine d'un exemple : **pose-moi la question** plutôt que de trancher seul.

Je vais te donner un premier cours (PDF des diapos + éventuellement mes notes en .txt ou .md). Fais-moi la fiche selon ces règles.