# 📝 Fiche rapide — Markdown

## Sommaire
1. [Titres](#1-titres)
2. [Mise en forme du texte](#2-mise-en-forme-du-texte)
3. [Listes](#3-listes)
4. [Liens et images](#4-liens-et-images)
5. [Code](#5-code)
6. [Citations et séparateurs](#6-citations-et-séparateurs)
7. [Tableaux](#7-tableaux)
8. [Divers utiles](#8-divers-utiles)
9. [Pièges classiques](#-pièges-classiques)

---

## 1. Titres

| Syntaxe | Rendu | Explication |
|---|---|---|
| `# Titre` | Titre niveau 1 (H1) | Un seul `#` par document en général (le titre principal). |
| `## Titre` | Titre niveau 2 (H2) | Sections principales. |
| `### Titre` | Titre niveau 3 (H3) | Sous-sections. |

```markdown
# Titre du document
## Section 1
### Sous-section 1.1
```

---

## 2. Mise en forme du texte

| Syntaxe | Rendu | Explication |
|---|---|---|
| `**gras**` ou `__gras__` | **gras** | Les deux syntaxes sont équivalentes. |
| `*italique*` ou `_italique_` | *italique* | idem. |
| `***gras italique***` | ***gras italique*** | Combinaison des deux. |
| `~~barré~~` | ~~barré~~ | Texte barré. |
| `` `code inline` `` | `code inline` | Pour un bout de code ou un nom de fonction dans une phrase. |

---

## 3. Listes

| Syntaxe | Rendu | Explication |
|---|---|---|
| `- item` ou `* item` | liste à puces | Indenter de 2-4 espaces pour imbriquer un sous-niveau. |
| `1. item` | liste numérotée | Les numéros n'ont pas besoin d'être corrects (`1.` répété partout fonctionne aussi), le rendu se renumérote automatiquement. |
| `- [ ] tâche` / `- [x] tâche faite` | ☐ tâche / ☑ tâche | Liste de tâches (case à cocher), supportée sur GitHub et la plupart des viewers modernes. |

```markdown
- Élément 1
  - Sous-élément 1.1
- Élément 2

1. Premier
2. Deuxième

- [x] Fait
- [ ] À faire
```

---

## 4. Liens et images

| Syntaxe | Rendu | Explication |
|---|---|---|
| `[texte](url)` | lien cliquable | Lien simple. |
| `[texte](url "titre")` | lien avec info-bulle | `titre` s'affiche au survol. |
| `![texte alternatif](url)` | image affichée | Même syntaxe qu'un lien, précédée de `!`. |
| `[texte][ref]` ... `[ref]: url` | lien via référence | Pratique pour réutiliser une même URL plusieurs fois dans le document. |
| `#ancre-du-titre` (dans une URL locale) | lien vers une section | L'ancre = titre en minuscules, espaces remplacés par `-`, ponctuation supprimée. |

```markdown
[Documentation Python](https://docs.python.org)
![logo](./images/logo.png)
Voir la [section 3](#3-listes) plus haut.
```

---

## 5. Code

| Syntaxe | Rendu | Explication |
|---|---|---|
| `` `code` `` | `code` | Code inline, dans une phrase. |
| ```` ```langage ... ``` ```` | bloc de code coloré | Bloc multi-lignes ; préciser le langage (`python`, `bash`, `r`...) active la coloration syntaxique. |
| 4 espaces d'indentation | bloc de code (ancienne syntaxe) | Équivalent aux triples backticks, mais sans coloration ni info de langage — préférer les triples backticks. |

````markdown
```python
def f(x):
    return x**2
```
````

---

## 6. Citations et séparateurs

| Syntaxe | Rendu | Explication |
|---|---|---|
| `> texte` | bloc de citation | `>>` pour une citation imbriquée. |
| `---` ou `***` ou `___` (seul sur une ligne) | ligne horizontale | Sépare visuellement deux sections. ⚠️ Laisser une ligne vide avant et après. |

```markdown
> Ceci est une citation.
> Sur plusieurs lignes.

---
```

---

## 7. Tableaux

| Syntaxe | Explication |
|---|---|
| `\| Col1 \| Col2 \|` puis `\|---\|---\|` | Ligne d'en-tête, puis ligne de séparation obligatoire (les tirets définissent le tableau). |
| `:---` / `:---:` / `---:` | Alignement à gauche / centré / à droite dans la ligne de séparation. |

```markdown
| Nom     | Âge | Ville      |
|---------|:---:|-----------:|
| Alice   | 30  |      Paris |
| Bob     | 25  |       Lyon |
```

---

## 8. Divers utiles

| Syntaxe | Rendu | Explication |
|---|---|---|
| `texte  ` (2 espaces en fin de ligne) puis retour à la ligne | saut de ligne simple | Une ligne vide seule crée un nouveau **paragraphe**, pas juste un retour à la ligne. |
| `\*texte échappé\*` | \*texte échappé\* | `\` échappe un caractère spécial pour l'afficher tel quel. |
| `$x^2 + y^2$` | rendu LaTeX inline | Formule mathématique (supporté par de nombreux moteurs Markdown, dont GitHub). |
| `$$ ... $$` | formule sur sa propre ligne | Version bloc du LaTeX. |
| `[^1]` ... `[^1]: note` | appel de note[^ex] + note en bas de page | Notes de bas de page (support variable selon le moteur). |

[^ex]: Exemple de note de bas de page.

---

## 📌 Pièges classiques

1. **Une ligne vide sépare les paragraphes** ; un simple retour à la ligne sans ligne vide est souvent ignoré au rendu (sauf 2 espaces en fin de ligne).
2. **La ligne de séparation `|---|---|` est obligatoire** dans un tableau, sinon il ne s'affiche pas comme un tableau.
3. **Les ancres de sommaire (`#titre`)** doivent correspondre exactement au titre en minuscules, espaces → tirets, ponctuation supprimée — une faute de frappe casse le lien silencieusement.
4. **Ne pas mélanger indentation par tabulation et par espaces** dans les listes imbriquées — le rendu peut casser selon le moteur.
5. **`*` et `_` pour l'italique/gras peuvent entrer en conflit** avec du texte contenant des astérisques ou underscores légitimes (ex. `variable_nom`) — échapper avec `\` si besoin.
