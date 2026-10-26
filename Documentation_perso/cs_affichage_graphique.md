# 📊 Cheat Sheet Affichage graphique (Matplotlib & Seaborn)

> Basée sur les notebooks *Affichage matplotlib* et *Affichage avancé*.
> Convention : `import matplotlib.pyplot as plt` (et souvent `import numpy as np`) ; `import seaborn as sns` pour la partie II.

## Sommaire

**I. Matplotlib**
1. [Structure de base d'un graphique](#i1-structure-de-base-dun-graphique)
2. [Nuages de points et courbes](#i2-nuages-de-points-et-courbes)
3. [Sous-graphiques (`subplot`)](#i3-sous-graphiques-subplot)
4. [Affichage de matrices (images)](#i4-affichage-de-matrices-images)
5. [Histogrammes](#i5-histogrammes)
6. [Remplissage entre courbes et intervalles de confiance](#i6-remplissage-entre-courbes-et-intervalles-de-confiance)
7. [Boîtes à moustaches (boxplot)](#i7-boîtes-à-moustaches-boxplot)
8. [Affichage 3D et grilles (`meshgrid`)](#i8-affichage-3d-et-grilles-meshgrid)
9. [Interactivité (usage avancé)](#i9-interactivité-usage-avancé)

**II. Seaborn**
1. [Style général](#ii1-style-général)
2. [Distributions (1D / 2D)](#ii2-distributions-1d--2d)
3. [Relations entre variables](#ii3-relations-entre-variables)
4. [Variables catégorielles](#ii4-variables-catégorielles)
5. [Matrices et évaluation de modèles (ML/DL)](#ii5-matrices-et-évaluation-de-modèles-mldl)

[Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Matplotlib

## I.1. Structure de base d'un graphique

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.figure(figsize=[l,h])` | `figsize` optionnel (largeur, hauteur en pouces) | Crée une nouvelle fenêtre/figure. **À appeler avant chaque nouveau graphique** si on veut plusieurs figures séparées (sinon les tracés s'accumulent sur la dernière figure ouverte). |
| `plt.show()` | — | Affiche la fenêtre. Parfois nécessaire selon l'environnement (souvent implicite dans Jupyter). |
| `plt.title(texte)` | `texte` | Titre du graphique. |
| `plt.xlabel(texte)` / `plt.ylabel(texte)` | `texte` | Légende des axes. |
| `plt.legend(loc=n)` | `loc` = position (1=haut droite, 2=haut gauche, 3=bas gauche, 4=bas droite...) | Affiche la légende, à condition d'avoir donné un `label=` à chaque tracé. |
| `plt.axis([xmin,xmax,ymin,ymax])` | liste des 4 bornes | Force manuellement les limites des axes (utile pour éviter les effets d'optique liés à un zoom automatique). |
| `plt.savefig(fichier)` | `fichier` (ex `'fig.pdf'`) | Sauvegarde la figure **courante**. Préférer un format **vectoriel** (`.pdf`, `.svg`) pour un zoom sans perte de qualité, notamment pour un rapport. |
| `plt.grid(visible, which=, axis=, color=, linestyle=, linewidth=, alpha=)` | `visible` (`True`/`False`) ; `which`=`'major'`/`'minor'`/`'both'` ; `axis`=`'both'`/`'x'`/`'y'` ; `color` (ex `'gray'`) ; `linestyle`(ou `ls`) = `'-'`/`'--'`/`'-.'`/`':'` ; `linewidth`(ou `lw`) ; `alpha` (0 à 1) | Affiche une grille en fond, avec un contrôle fin du style. Équivalent objet : `ax.grid(...)`. |

```python
plt.figure()
plt.plot(x, y)
plt.title('Mon titre')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()
```

---

## I.2. Nuages de points et courbes

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.scatter(x, y, c=..., s=...)` | `x`,`y` = vecteurs de même taille, `c` = couleur (peut être un vecteur de valeurs → dégradé), `s` = taille des points | Affiche un **nuage de points** : chaque couple `(x_i, y_i)` est un point isolé, non relié. |
| `plt.plot(x, y, style, label=...)` | `style` = code couleur+forme (ex `'r+-'`, `'b--'`), `label` pour la légende | Trace une **ligne** reliant les points `(x_i, y_i)` dans l'ordre. |

**Codes de style courants pour `plot`** : couleur (`r`=rouge, `b`=bleu, `g`=vert, `k`=noir...) + marqueur (`+`, `*`, `o`) + trait (`-` continu, `--` pointillé).

```python
plt.plot(x, y1, 'r+-', label='$y=x^2+2x$')  # rouge, croix, ligne continue
plt.plot(x, y2, 'b--')                       # bleu, pointillés
plt.legend(loc=4)
```

⚠️ **`scatter` vs `plot`** : `scatter` = nuage de points indépendants (pas de lien visuel entre eux) ; `plot` = ligne qui relie les points dans l'ordre où ils sont donnés. Pour tracer une fonction `y=f(x)`, `plot` est en général préférable dès que les points sont ordonnés selon `x`.

**Colorer un nuage de points selon une catégorie / une valeur** :
```python
plt.scatter(x[:, 0], x[:, 1], c=y)   # couleur = valeur de y (catégorie ou grandeur continue)
```
ou, sans passer par `c`, en sélectionnant les sous-ensembles manuellement (utile pour des styles différents par classe) :
```python
plt.plot(x[y==1, 0], x[y==1, 1], 'b+')    # classe 1 en croix bleues
plt.plot(x[y==-1, 0], x[y==-1, 1], 'r*')  # classe -1 en étoiles rouges
```

---

## I.3. Sous-graphiques (`subplot`)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.subplot(nlignes, ncolonnes, indice)` | `indice` commence à **1** (et pas 0 !) | Découpe la figure courante en grille et sélectionne la case active pour les tracés suivants. |
| `fig, ax = plt.subplots(n, m)` | `n`, `m` = grille | Alternative orientée objet : renvoie la figure et un tableau d'axes, pratique pour personnaliser chaque sous-graphique individuellement. |

```python
plt.figure()
plt.subplot(1, 2, 1)          # grille 1x2, case 1
plt.plot(x, y1, label='courbe 1')
plt.legend(loc=4)
plt.subplot(1, 2, 2)          # case 2
plt.plot(x, y2, label='courbe 2')
plt.show()
```

---

## I.4. Affichage de matrices (images)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.imshow(mat, cmap=, interpolation=)` | `cmap` (ex `'viridis'`, `'gray'`), `interpolation` (ex `'nearest'`) | Affiche une matrice comme une image : chaque case = un pixel coloré selon sa valeur. **L'outil de référence** pour explorer visuellement une grande matrice numpy. |
| `plt.colorbar()` | — | Affiche l'échelle de couleurs. **Quasi indispensable** dès qu'on utilise `imshow`, pour interpréter les couleurs. |
| `ax.set_xticks(indices)` / `ax.set_xticklabels(labels, rotation=, fontsize=)` | `indices`, `labels` | Personnalise les graduations et les étiquettes de l'axe des x (utile pour nommer les colonnes d'une matrice). |

```python
fig, ax = plt.subplots(1, 1)
plt.imshow(C, interpolation='nearest')
ax.set_xticks(np.arange(C.shape[1]))
ax.set_xticklabels(['col1', 'col2', ...], rotation=60, fontsize=8)
plt.colorbar()
```

---

## I.5. Histogrammes

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.bar(x, hauteurs, width=, edgecolor=)` | `x` = abscisses, `hauteurs` = valeurs, `width` = largeur des barres | Affiche des barres dont **on a déjà calculé** la hauteur (comptage manuel par exemple). |
| `plt.hist(donnees, bins=n)` | `donnees` = valeurs brutes, `bins` = nombre d'intervalles | Calcule **et** affiche l'histogramme directement à partir des données brutes (pas besoin de compter à la main). |
| `np.histogram(donnees, bins=)` | idem | Version numpy qui **calcule** l'histogramme sans l'afficher (renvoie comptages + bornes). |

⚠️ **`bar` vs `hist`** : `bar` est un simple outil d'affichage (on donne les abscisses et les hauteurs) ; `hist` fait le comptage à la place de l'utilisateur. Bien comprendre `bar` "à la main" une fois aide à comprendre ce que fait vraiment un histogramme, avant de passer à la version automatique `hist`.

```python
histo = np.zeros(21)
for v in notes:
    histo[v] += 1
plt.bar(np.arange(21), histo)     # affichage manuel

plt.hist(notes, bins=10)          # équivalent automatique, en jouant sur bins
```

---

## I.6. Remplissage entre courbes et intervalles de confiance

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.fill_between(x, y1, y2, alpha=, where=, fc=)` | `y1`,`y2` = bornes basse/haute, `alpha` = transparence, `where` = condition booléenne pour ne remplir qu'une partie, `fc` = couleur de remplissage | Colorie la zone entre deux courbes. Très utilisé pour représenter des **intervalles de confiance** ou des zones de dépassement de seuil. |
| `np.polyfit(x, y, deg=1)` | `deg` = degré du polynôme | Régression polynomiale simple (renvoie les coefficients), utile pour tracer une tendance avant de calculer un intervalle autour. |

```python
plt.plot(x, y_est, '-')
plt.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)  # bande d'incertitude
plt.fill_between(t, upper_bound, X, where=X > upper_bound, fc='red', alpha=0.4)  # ne colorie que le dépassement
```

---

## I.7. Boîtes à moustaches (boxplot)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `plt.boxplot(donnees)` | `donnees` = vecteur (ou liste de vecteurs) | Affiche médiane, quartiles et valeurs extrêmes ("moustaches") d'une distribution. |

---

## I.8. Affichage 3D et grilles (`meshgrid`)

Organisation générale pour tracer $z = f(x,y)$ :
1. Définir les plages sur `x` et `y` (`linspace`).
2. Construire une grille avec `np.meshgrid`.
3. Évaluer la fonction **sur la grille** (attention aux dimensions résultantes : ce sont des matrices, pas des vecteurs).
4. Afficher avec l'outil adapté (`contour`, `contourf`, `plot_surface`...).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `np.meshgrid(x, y)` | `x`, `y` = vecteurs 1D | Construit deux **matrices** `xgrid`, `ygrid` représentant toutes les combinaisons de coordonnées (maillage 2D). ⚠️ Résultat en matrices, pas en vecteurs — source classique d'erreurs de dimension. |
| `plt.contour(xgrid, ygrid, zgrid)` | — | Lignes de niveaux. |
| `plt.contourf(xgrid, ygrid, zgrid)` | — | Lignes de niveaux **remplies**. |
| `ax.plot_surface(xgrid, ygrid, zgrid, cmap=)` | nécessite `fig, ax = plt.subplots(subplot_kw={"projection": "3d"})` | Surface 3D. `ax.view_init(elev=, azim=)` règle l'angle de vue. |

```python
x = np.linspace(0, 10, 30)
y = np.linspace(-3, 3, 30)
xgrid, ygrid = np.meshgrid(x, y)
zgrid = 3*xgrid - 5*ygrid

plt.contourf(xgrid, ygrid, zgrid)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xgrid, ygrid, zgrid, cmap='coolwarm')
```

> ⚠️ Matplotlib est réputé peu adapté à une vraie 3D interactive (pas de z-buffer), mais reste très pratique pour des rendus statiques.

---

## I.9. Interactivité (usage avancé)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `fig.canvas.mpl_connect(evenement, callback)` | `evenement` (ex `'button_press_event'`), `callback` = fonction appelée | Associe une fonction à un événement sur la figure. Ne fonctionne pas dans les notebooks Jupyter classiques, mais dans un script python avec fenêtre graphique. |

---

# II. Seaborn

`seaborn` (`import seaborn as sns`) est construit sur matplotlib et propose des graphiques statistiques "clés en main", plus esthétiques et souvent plus rapides à écrire. La plupart des fonctions travaillent directement sur un **DataFrame pandas** (`data=df`) en désignant les colonnes par leur nom → voir la cheat sheet Pandas pour la construction des DataFrames.

## II.1. Style général

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sns.set_theme(style=...)` | `style` = `'darkgrid'`, `'whitegrid'`, `'white'`, `'ticks'`... | Configure le style global pour tous les graphiques suivants, y compris ceux en matplotlib pur. |
| `sns.set_palette(nom)` | `nom` = nom de palette (ex `'viridis'`, `'pastel'`) ou liste de couleurs | Change la palette de couleurs utilisée par défaut. |

## II.2. Distributions (1D / 2D)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sns.histplot(data, x=, hue=, kde=)` | `data` = DataFrame, `x` = colonne à tracer, `hue` = colonne de catégorie (couleur), `kde` = `True`/`False` (ajoute une estimation de densité) | Histogramme enrichi, équivalent de `plt.hist`. |
| `sns.kdeplot(data, x=, y=, fill=)` | `x` (et `y` optionnel pour la 2D), `fill` = `True`/`False` (remplissage sous la courbe) | Estimation de densité (courbe lissée). En 2D, affiche une carte de densité. |
| `sns.displot(data, x=, hue=, kind=, col=)` | `kind` = `'hist'`/`'kde'`/`'ecdf'`, `col` = colonne pour générer un sous-graphique par catégorie | Fonction générique de distribution, gère automatiquement le découpage en sous-graphiques. |
| `sns.ecdfplot(data, x=)` | `x` = colonne | Fonction de répartition empirique (cumulative). |

## II.3. Relations entre variables

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sns.scatterplot(data, x=, y=, hue=, size=)` | `hue` = couleur par catégorie, `size` = taille des points par valeur | Nuage de points, pratique pour visualiser 3-4 dimensions à la fois (position + couleur + taille). |
| `sns.lineplot(data, x=, y=, hue=)` | `hue` = une courbe par catégorie | Courbe, avec **intervalle de confiance ombré automatique** si plusieurs observations par valeur de `x` (pratique pour des courbes d'apprentissage moyennées sur plusieurs runs). |
| `sns.regplot(data, x=, y=)` | `x`, `y` = colonnes numériques | Nuage de points + droite de régression linéaire ajustée automatiquement. |
| `sns.jointplot(data, x=, y=, kind=)` | `kind` = `'scatter'`, `'kde'`, `'hex'`, `'reg'` | Combine un nuage de points central et les distributions marginales sur les côtés. |
| `sns.pairplot(df, hue=)` | `hue` = colonne de catégorie pour la couleur | Matrice de nuages de points **croisant toutes les paires de colonnes numériques** d'un DataFrame — excellent premier réflexe pour explorer un jeu de données avant un modèle de machine learning. |

## II.4. Variables catégorielles

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sns.boxplot(data, x=, y=, hue=)` | `x` = catégorie, `y` = valeur numérique | Boîtes à moustaches par catégorie. |
| `sns.violinplot(data, x=, y=, hue=)` | idem | Comme un boxplot mais affiche la forme de la distribution (densité miroir). |
| `sns.barplot(data, x=, y=, hue=)` | idem | Barres avec la moyenne (+ intervalle de confiance) par catégorie — pas un simple comptage. |
| `sns.countplot(data, x=)` | `x` = catégorie | Comptage du nombre d'observations par catégorie (l'équivalent catégoriel d'un histogramme). |
| `sns.stripplot(data, x=, y=)` / `sns.swarmplot(data, x=, y=)` | idem | Affiche chaque point individuel réparti par catégorie (`swarmplot` évite les recouvrements). |

## II.5. Matrices et évaluation de modèles (ML/DL)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `sns.heatmap(matrice, annot=, cmap=, fmt=)` | `annot` = `True`/`False` (écrit les valeurs dans chaque case), `cmap` (ex `'coolwarm'`, `'Blues'`), `fmt` = format d'affichage (ex `'d'` pour des entiers) | Affiche une matrice sous forme de carte de chaleur. **Usage très courant** : matrice de corrélation (`df.corr()`), ou **matrice de confusion** d'un modèle de classification (`sklearn.metrics.confusion_matrix`). |

```python
import seaborn as sns
sns.set_theme(style="whitegrid")

sns.pairplot(df, hue="classe")                     # exploration rapide d'un dataset
sns.heatmap(df.corr(), annot=True, cmap='coolwarm') # matrice de corrélation
sns.lineplot(data=df, x="epoch", y="loss", hue="run")  # courbe d'apprentissage avec IC
```

> 💡 Pour une matrice de confusion (classification), le réflexe courant est :
> `sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')`

---

## 📌 Récapitulatif des pièges classiques

1. **Oublier `plt.figure()`** avant un nouveau tracé → les courbes s'accumulent sur la figure précédente.
2. **`scatter` vs `plot`** : `scatter` = points isolés, `plot` = ligne reliant les points dans l'ordre donné.
3. **`bar` vs `hist`** : `bar` affiche des hauteurs déjà calculées, `hist` calcule et affiche en une seule fonction.
4. **`subplot` indexe à partir de 1**, pas de 0.
5. **`np.meshgrid` renvoie des matrices**, pas des vecteurs : bien vérifier les `.shape` avant de calculer `zgrid`.
6. **Oublier `plt.colorbar()`** après un `imshow` → impossible d'interpréter les couleurs.
7. Pour un rapport, privilégier `plt.savefig('fig.pdf')` (format **vectoriel**) plutôt qu'un PNG, pour un zoom sans perte.
8. **Seaborn attend en général un DataFrame** (`data=df`) avec des noms de colonnes en `x=`/`y=`/`hue=`, plutôt que des tableaux numpy bruts.
