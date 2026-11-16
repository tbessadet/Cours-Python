# FDV – TD Clustering

Sommaire :
- [TD 1 – Clustering hiérarchique (CAH)](#td-1--clustering-hiérarchique-cah)
  - [Exercice 1 : CAH, saut minimal / saut maximal](#exercice-1--classification-ascendante-hiérarchique)
  - [Exercice 2 (DM) : CAH avec l'écart de Ward](#exercice-2-dm--cah-avec-lécart-de-ward)
- [TD 2 – Méthode des k-moyennes](#td-2--méthode-des-k-moyennes)
  - [Exercice 1 : K-means](#exercice-1--k-means)
  - [Exercice 2 (DM) : dépendance à l'initialisation](#exercice-2--k-means-dm)

---

# TD 1 – Clustering hiérarchique (CAH)

## Exercice 1 : Classification Ascendante Hiérarchique

### Énoncé

Les données sont décrites par le tableau de distances ci-dessous :

|       | A | B | C | D |
|-------|---|---|---|---|
| **A** | 0 | 1 | 4 | 5 |
| **B** |   | 0 | 2 | 6 |
| **C** |   |   | 0 | 3 |
| **D** |   |   |   | 0 |

- Dessinez les dendrogrammes associés à une CAH :
  - en considérant dans un premier temps le **saut minimal** (ou plus proche voisin) comme mesure de distance entre clusters ;
  - en considérant ensuite le **saut maximal** (ou diamètre maximum) comme mesure de distance.

Les formules des distances à utiliser sont rappelées ci-dessous :

- plus proche voisin : $\min\big(d(i,j),\ i \in C_1,\ j \in C_2\big)$
- diamètre maximum : $\max\big(d(i,j),\ i \in C_1,\ j \in C_2\big)$

- Que peut-on en conclure ?

### Réponse

**Principe de la CAH** : au départ chaque point est un cluster. À chaque étape on fusionne les **deux clusters les plus proches**, puis on recalcule les distances entre le nouveau cluster et les autres avec la règle choisie (min ou max). On note la hauteur de chaque fusion = distance à laquelle elle se fait.

#### a) Saut minimal (single linkage)

**Étape 1** – plus petite distance : $d(A,B) = 1$ → on fusionne **{A,B}** à la hauteur **1**.

Nouvelles distances (on prend le **min**) :
- $d(\{A,B\}, C) = \min(4, 2) = 2$
- $d(\{A,B\}, D) = \min(5, 6) = 5$
- $d(C, D) = 3$

|           | {A,B} | C | D |
|-----------|-------|---|---|
| **{A,B}** | 0     | 2 | 5 |
| **C**     |       | 0 | 3 |
| **D**     |       |   | 0 |

**Étape 2** – plus petite distance : 2 → on fusionne **{A,B,C}** à la hauteur **2**.

- $d(\{A,B,C\}, D) = \min(5, 6, 3) = 3$

**Étape 3** – on fusionne **{A,B,C,D}** à la hauteur **3**.

Dendrogramme (saut minimal) :

```
            │
 3 ┤   ┌────┴───┐
   │   │        │
 2 ┤ ┌─┴───┐    │
   │ │     │    │
 1 ┤┌┴┐    │    │
   ││ │    │    │
 0 ┼A─B────C────D
```

#### b) Saut maximal (complete linkage)

**Étape 1** – plus petite distance : $d(A,B) = 1$ → on fusionne **{A,B}** à la hauteur **1**.

Nouvelles distances (on prend le **max**) :
- $d(\{A,B\}, C) = \max(4, 2) = 4$
- $d(\{A,B\}, D) = \max(5, 6) = 6$
- $d(C, D) = 3$

|           | {A,B} | C | D |
|-----------|-------|---|---|
| **{A,B}** | 0     | 4 | 6 |
| **C**     |       | 0 | 3 |
| **D**     |       |   | 0 |

**Étape 2** – plus petite distance : $d(C,D) = 3$ → on fusionne **{C,D}** à la hauteur **3**.

- $d(\{A,B\}, \{C,D\}) = \max(4, 5, 2, 6) = 6$

**Étape 3** – on fusionne **{A,B,C,D}** à la hauteur **6**.

Dendrogramme (saut maximal) :

```
                │
 6 ┤      ┌─────┴────┐
 5 ┤      │          │
 4 ┤      │          │
 3 ┤      │       ┌──┴─┐
 2 ┤      │       │    │
 1 ┤ ┌────┴─┐     │    │
 0 ┼─A──────B─────C────D
```

#### c) Conclusion

- **Les deux dendrogrammes sont différents** alors que les données sont les mêmes : le résultat d'une CAH dépend du **critère d'agrégation** choisi.
- Si on coupe pour obtenir 2 classes :
  - saut minimal → {A, B, C} et {D}
  - saut maximal → {A, B} et {C, D}
- Le **saut minimal** a tendance à « enchaîner » les points (effet de chaîne) : un cluster grossit en absorbant un point à la fois → clusters allongés, hauteurs de fusion faibles.
- Le **saut maximal** favorise des clusters **compacts** de diamètre faible, plus équilibrés, mais il est sensible aux points éloignés (outliers).

---

## Exercice 2 (DM) : CAH avec l'écart de Ward

### Énoncé

La succession des quatre dessins suivants correspond aux étapes successives d'une classification hiérarchique ascendante des cinq points $M_1(2,0)$, $M_2(0,1)$, $M_3(0,2)$, $M_4(3,4)$ et $M_5(5,4)$ progressivement regroupés en classes de deux ou trois points dont les centres de gravité sont notés $G_6$, $G_7$, $G_8$. On suppose que les cinq points initiaux sont tous affectés du poids 1. La distance choisie pour cette classification, qui apparaît dans les quatre matrices de distance, est l'**écart de Ward**.

*(Les dessins du sujet sont des nuages de points sur une grille 0–6 × 0–5. Ils sont décrits ci-dessous par la liste de leurs points.)*

**Dessin 1** : 5 points en (2,0), (0,1), (0,2), (3,4), (5,4).

|        | M1    | M2    | M3    | M4    | M5    |
|--------|-------|-------|-------|-------|-------|
| **M1** | 0.00  | 2.50  | 4.00  | 8.50  | 12.50 |
| **M2** | 2.50  | 0.00  | 0.50  | 9.00  | 17.00 |
| **M3** | 4.00  | 0.50  | 0.00  | 6.50  | 14.50 |
| **M4** | 8.50  | 9.00  | 6.50  | *?*   | *?*   |
| **M5** | 12.50 | *?*   | *?*   | 2.00  | 0.00  |

**Dessin 2** : 4 points en (2,0), (0,1.5), (3,4), (5,4).

|        | M1    | G6    | M4    | M5    |
|--------|-------|-------|-------|-------|
| **M1** | 0.00  | 4.17  | 8.50  | 12.50 |
| **G6** | 4.17  | 0.00  | *?*   | 20.83 |
| **M4** | 8.50  | *?*   | 0.00  | 2.00  |
| **M5** | 12.50 | 20.83 | 2.00  | 0.00  |

**Dessin 3** : grille vide (à compléter).

|        | M1    | G6    | G7    |
|--------|-------|-------|-------|
| **M1** | 0.00  | 4.17  | 13.33 |
| **G6** | 4.17  | 0.00  | 22.25 |
| **G7** | 13.33 | 22.25 | 0.00  |

**Dessin 4** : 2 points, environ (0.7, 1) et (4, 4).

|        | G8    | G7    |
|--------|-------|-------|
| **G8** | 0.00  | 24.13 |
| **G7** | 24.13 | 0.00  |

1. Compléter le troisième dessin en y plaçant les trois points devant y figurer et indiquer sur les quatre dessins le nom des points.
2. Précisez les coordonnées des points $G_6$, $G_7$, $G_8$.
3. Compléter les six distances manquantes dans les matrices de distances.
4. Calculer les coordonnées du centre de gravité $G_9$ des cinq points.
5. Tracer un dendrogramme résumant cette classification.

### Réponse

**Rappel – écart de Ward** entre deux classes $C_a$ (effectif $n_a$, centre $g_a$) et $C_b$ (effectif $n_b$, centre $g_b$) :

$$\Delta(C_a, C_b) = \frac{n_a\, n_b}{n_a + n_b}\ \lVert g_a - g_b \rVert^2$$

C'est l'**augmentation de l'inertie intra-classe** quand on fusionne les deux classes. Pour deux points seuls ($n_a = n_b = 1$) : $\Delta = \frac{1}{2} d^2$.
Exemple de vérification : $d^2(M_1, M_2) = (2-0)^2 + (0-1)^2 = 5$ → $\Delta = 2.5$ ✔.

À chaque étape, on fusionne la paire de **plus petit écart de Ward** :

| Étape | Plus petit écart | Fusion          | Nouveau centre |
|-------|------------------|-----------------|----------------|
| 1     | 0.50             | M2 + M3         | G6             |
| 2     | 2.00             | M4 + M5         | G7             |
| 3     | 4.17             | M1 + G6         | G8             |
| 4     | 24.13            | G8 + G7         | G9             |

#### 1. Nom des points sur les dessins

- **Dessin 1** : $M_1(2,0)$, $M_2(0,1)$, $M_3(0,2)$, $M_4(3,4)$, $M_5(5,4)$.
- **Dessin 2** : $M_1(2,0)$, $G_6(0\,;1.5)$, $M_4(3,4)$, $M_5(5,4)$.
- **Dessin 3** (à compléter) : $M_1(2,0)$, $G_6(0\,;1.5)$, $G_7(4,4)$.
- **Dessin 4** : $G_8(0.67\,;1)$, $G_7(4,4)$.

Dessin 3 complété :

```
 5 ┤
   │
 4 ┤                G7
   │
 3 ┤
   │
 2 ┤
   │G6                (G6 en y = 1,5)
 1 ┤
   │
 0 ┼────────M1──────────────────
    0   1   2   3   4   5   6
```

#### 2. Coordonnées de G6, G7, G8

Le centre de gravité d'une classe = moyenne des coordonnées de ses points (tous de poids 1).

- $G_6 = \text{centre}(M_2, M_3) = \left(\frac{0+0}{2}, \frac{1+2}{2}\right) = (0\,;\ 1.5)$
- $G_7 = \text{centre}(M_4, M_5) = \left(\frac{3+5}{2}, \frac{4+4}{2}\right) = (4\,;\ 4)$
- $G_8 = \text{centre}(M_1, M_2, M_3) = \left(\frac{2+0+0}{3}, \frac{0+1+2}{3}\right) = \left(\tfrac{2}{3}\,;\ 1\right) \approx (0.67\,;\ 1)$

#### 3. Les six distances manquantes

**Matrice 1** (4 cases vides) :
- $\Delta(M_4, M_4) = 0.00$ (diagonale)
- $\Delta(M_4, M_5) = \frac{1}{2}\big[(3-5)^2 + (4-4)^2\big] = \frac{4}{2} = 2.00$ (symétrique de la case $M_5$–$M_4$)
- $\Delta(M_5, M_2) = \frac{1}{2}\big[(5-0)^2 + (4-1)^2\big] = \frac{34}{2} = 17.00$
- $\Delta(M_5, M_3) = \frac{1}{2}\big[(5-0)^2 + (4-2)^2\big] = \frac{29}{2} = 14.50$

**Matrice 2** (2 cases vides, $G_6$–$M_4$ et $M_4$–$G_6$) : $n_{G_6} = 2$, $n_{M_4} = 1$

$$\Delta(G_6, M_4) = \frac{2 \times 1}{2+1}\big[(0-3)^2 + (1.5-4)^2\big] = \frac{2}{3} \times (9 + 6.25) = \frac{2}{3} \times 15.25 \approx 10.17$$

Matrices complétées :

|        | M1    | M2    | M3    | M4       | M5        |
|--------|-------|-------|-------|----------|-----------|
| **M1** | 0.00  | 2.50  | 4.00  | 8.50     | 12.50     |
| **M2** | 2.50  | 0.00  | 0.50  | 9.00     | 17.00     |
| **M3** | 4.00  | 0.50  | 0.00  | 6.50     | 14.50     |
| **M4** | 8.50  | 9.00  | 6.50  | **0.00** | **2.00**  |
| **M5** | 12.50 | **17.00** | **14.50** | 2.00 | 0.00 |

|        | M1    | G6        | M4        | M5    |
|--------|-------|-----------|-----------|-------|
| **M1** | 0.00  | 4.17      | 8.50      | 12.50 |
| **G6** | 4.17  | 0.00      | **10.17** | 20.83 |
| **M4** | 8.50  | **10.17** | 0.00      | 2.00  |
| **M5** | 12.50 | 20.83     | 2.00      | 0.00  |

*Vérification des valeurs données par le sujet* (même formule) :
- $\Delta(M_1, G_6) = \frac{2}{3}\big[2^2 + 1.5^2\big] = \frac{2}{3} \times 6.25 \approx 4.17$ ✔
- $\Delta(M_1, G_7) = \frac{2}{3}\big[2^2 + 4^2\big] = \frac{2}{3} \times 20 \approx 13.33$ ✔
- $\Delta(G_6, G_7) = \frac{2 \times 2}{4}\big[4^2 + 2.5^2\big] = 1 \times 22.25 = 22.25$ ✔
- $\Delta(G_8, G_7) = \frac{3 \times 2}{5}\big[(4 - \tfrac{2}{3})^2 + 3^2\big] = 1.2 \times (11.11 + 9) \approx 24.13$ ✔

#### 4. Centre de gravité G9 des cinq points

$$G_9 = \left(\frac{2+0+0+3+5}{5}\ ;\ \frac{0+1+2+4+4}{5}\right) = \left(\frac{10}{5}\ ;\ \frac{11}{5}\right) = (2\ ;\ 2.2)$$

Vérification avec les centres pondérés : $G_9 = \frac{3\,G_8 + 2\,G_7}{5} = \left(\frac{2 + 8}{5}\ ;\ \frac{3 + 8}{5}\right) = (2\ ;\ 2.2)$ ✔

#### 5. Dendrogramme

Hauteurs de fusion = écarts de Ward : 0.50, 2.00, 4.17, 24.13 (échelle non respectée, coupure ≈ sur l'axe).

```
                     │
24.13 ┤        ┌─────┴─────┐          G9
      ≈        │           │
 4.17 ┤    ┌───┴───┐       │          G8
      │    │       │       │
 2.00 ┤    │       │     ┌─┴─┐        G7
      │    │       │     │   │
 0.50 ┤  ┌─┴─┐     │     │   │        G6
      │  │   │     │     │   │
         M2  M3    M1    M4  M5
```

**Remarque** : la somme des hauteurs $0.5 + 2 + 4.17 + 24.13 = 30.8$ est égale à l'**inertie totale** du nuage autour de $G_9$ :
$4.84 + 5.44 + 4.04 + 4.24 + 12.24 = 30.8$. C'est une propriété de la méthode de Ward : chaque fusion ajoute exactement son écart à l'inertie intra-classe. Le grand saut entre 4.17 et 24.13 suggère de couper en **2 classes** : {M1, M2, M3} et {M4, M5}.

---

# TD 2 – Méthode des k-moyennes

## Exercice 1 : K-means

### Énoncé

On considère les points du graphique ci-dessous.
On considère comme centres de classes à l'initialisation les points **A1, A4 et A7**.

*(Coordonnées lues sur le graphique du sujet)*

| Point | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|-------|----|----|----|----|----|----|----|----|
| x     | 2  | 2  | 8  | 5  | 7  | 6  | 1  | 4  |
| y     | 10 | 5  | 4  | 8  | 5  | 4  | 2  | 9  |

```
 10 ┤     A1
  9 ┤           A8
  8 ┤              A4
  7 ┤
  6 ┤
  5 ┤     A2             A5
  4 ┤                 A6    A3
  3 ┤
  2 ┤  A7
  1 ┤
    └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬
       1  2  3  4  5  6  7  8  9  10
```

On considère la matrice de distances euclidiennes **au carré** associée :

|        | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|--------|----|----|----|----|----|----|----|----|
| **A1** | –  | 25 | 72 | 13 | 50 | 52 | 65 | 5  |
| **A2** | 25 | –  | 37 | 18 | 25 | 17 | 10 | 20 |
| **A3** | 72 | 37 | –  | 25 | 2  | 4  | 53 | 41 |
| **A4** | 13 | 18 | 25 | –  | 13 | 17 | 52 | 2  |
| **A5** | 50 | 25 | 2  | 13 | –  | 2  | 45 | 25 |
| **A6** | 52 | 17 | 4  | 17 | 2  | –  | 29 | 29 |
| **A7** | 65 | 10 | 53 | 52 | 45 | 29 | –  | 58 |
| **A8** | 5  | 20 | 41 | 2  | 25 | 29 | 58 | –  |

1. Déroulez une itération de l'algorithme de k-means pour ces données et cette initialisation et donnez :
   a. Les nouveaux clusters
   b. Les centres de chaque cluster
   c. Complétez la représentation graphique ci-dessus avec les clusters et les centres des clusters
2. Combien d'itérations supplémentaires sont-elles nécessaires pour converger ? Dessinez les résultats de chaque itération nécessaire.

### Réponse

**Rappel de l'algorithme** : répéter jusqu'à stabilité
1. **Affectation** : chaque point va dans le cluster dont le centre est le plus proche ;
2. **Mise à jour** : chaque centre devient la moyenne (centre de gravité) des points de son cluster.

On peut comparer les distances au carré directement (l'ordre est le même qu'avec les distances).

Légende des dessins : chiffre = numéro du cluster du point, `+` = centre du cluster (position arrondie à 0,5 près).

#### 1. Première itération

Centres initiaux : $c_1 = A1(2,10)$, $c_2 = A4(5,8)$, $c_3 = A7(1,2)$.
Les distances se lisent dans les colonnes A1, A4, A7 de la matrice :

| Point | d² à c1 (A1) | d² à c2 (A4) | d² à c3 (A7) | Cluster |
|-------|-----|-----|-----|---------|
| A1    | **0** | 13 | 65 | C1 |
| A2    | 25  | 18  | **10** | C3 |
| A3    | 72  | **25** | 53 | C2 |
| A4    | 13  | **0** | 52 | C2 |
| A5    | 50  | **13** | 45 | C2 |
| A6    | 52  | **17** | 29 | C2 |
| A7    | 65  | 52  | **0** | C3 |
| A8    | 5   | **2** | 58 | C2 |

**a. Nouveaux clusters** :
- C1 = {A1}
- C2 = {A3, A4, A5, A6, A8}
- C3 = {A2, A7}

**b. Centres** :
- $c_1 = (2\ ;\ 10)$
- $c_2 = \left(\frac{8+5+7+6+4}{5}\ ;\ \frac{4+8+5+4+9}{5}\right) = (6\ ;\ 6)$
- $c_3 = \left(\frac{2+1}{2}\ ;\ \frac{5+2}{2}\right) = (1.5\ ;\ 3.5)$

**c. Représentation** (après l'itération 1) :

```
10.0 | .  1+   .   .   .   .   .   .
     |
 9.0 | .   .   .   2   .   .   .   .
     |
 8.0 | .   .   .   .   2   .   .   .
     |
 7.0 | .   .   .   .   .   .   .   .
     |
 6.0 | .   .   .   .   .   +   .   .
     |
 5.0 | .   3   .   .   .   .   2   .
     |
 4.0 | .   .   .   .   .   2   .   2
     |   +
 3.0 | .   .   .   .   .   .   .   .
     |
 2.0 | 3   .   .   .   .   .   .   .
     +------------------------------
       1   2   3   4   5   6   7   8
```

#### 2. Itérations supplémentaires

Après l'itération 1, les centres ne sont plus des points du nuage : on calcule les d² avec les coordonnées.

**Itération 2** – centres $c_1(2\ ;10)$, $c_2(6\ ;6)$, $c_3(1.5\ ;3.5)$

| Point | d² à c1 | d² à c2 | d² à c3 | Cluster |
|-------|------|------|------|---------|
| A1    | **0**    | 32   | 42.5 | C1 |
| A2    | 25   | 17   | **2.5**  | C3 |
| A3    | 72   | **8**    | 42.5 | C2 |
| A4    | 13   | **5**    | 32.5 | C2 |
| A5    | 50   | **2**    | 32.5 | C2 |
| A6    | 52   | **4**    | 20.5 | C2 |
| A7    | 65   | 41   | **2.5**  | C3 |
| A8    | **5**    | 13   | 36.5 | **C1** (change !) |

Clusters : C1 = {A1, A8}, C2 = {A3, A4, A5, A6}, C3 = {A2, A7}
Centres : $c_1 = (3\ ;\ 9.5)$, $c_2 = \left(\frac{26}{4}\ ;\ \frac{21}{4}\right) = (6.5\ ;\ 5.25)$, $c_3 = (1.5\ ;\ 3.5)$

```
10.0 | .   1   .   .   .   .   .   .
     |         +
 9.0 | .   .   .   1   .   .   .   .
     |
 8.0 | .   .   .   .   2   .   .   .
     |
 7.0 | .   .   .   .   .   .   .   .
     |
 6.0 | .   .   .   .   .   .   .   .
     |
 5.0 | .   3   .   .   .   . + 2   .
     |
 4.0 | .   .   .   .   .   2   .   2
     |   +
 3.0 | .   .   .   .   .   .   .   .
     |
 2.0 | 3   .   .   .   .   .   .   .
     +------------------------------
       1   2   3   4   5   6   7   8
```

**Itération 3** – centres $c_1(3\ ;9.5)$, $c_2(6.5\ ;5.25)$, $c_3(1.5\ ;3.5)$

| Point | d² à c1 | d² à c2 | d² à c3 | Cluster |
|-------|-------|-------|------|---------|
| A1    | **1.25**  | 42.81 | 42.5 | C1 |
| A2    | 21.25 | 20.31 | **2.5**  | C3 |
| A3    | 55.25 | **3.81**  | 42.5 | C2 |
| A4    | **6.25**  | 9.81  | 32.5 | **C1** (change !) |
| A5    | 36.25 | **0.31**  | 32.5 | C2 |
| A6    | 39.25 | **1.81**  | 20.5 | C2 |
| A7    | 60.25 | 40.81 | **2.5**  | C3 |
| A8    | **1.25**  | 20.31 | 36.5 | C1 |

Clusters : C1 = {A1, A4, A8}, C2 = {A3, A5, A6}, C3 = {A2, A7}
Centres : $c_1 = \left(\frac{11}{3}\ ;\ 9\right) \approx (3.67\ ;\ 9)$, $c_2 = \left(7\ ;\ \frac{13}{3}\right) \approx (7\ ;\ 4.33)$, $c_3 = (1.5\ ;\ 3.5)$

```
10.0 | .   1   .   .   .   .   .   .
     |
 9.0 | .   .   . + 1   .   .   .   .
     |
 8.0 | .   .   .   .   1   .   .   .
     |
 7.0 | .   .   .   .   .   .   .   .
     |
 6.0 | .   .   .   .   .   .   .   .
     |
 5.0 | .   3   .   .   .   .   2   .
     |                         +
 4.0 | .   .   .   .   .   2   .   2
     |   +
 3.0 | .   .   .   .   .   .   .   .
     |
 2.0 | 3   .   .   .   .   .   .   .
     +------------------------------
       1   2   3   4   5   6   7   8
```

**Itération 4** – centres $c_1(3.67\ ;9)$, $c_2(7\ ;4.33)$, $c_3(1.5\ ;3.5)$

| Point | d² à c1 | d² à c2 | d² à c3 | Cluster |
|-------|-------|-------|------|---------|
| A1    | **3.78**  | 57.11 | 42.5 | C1 |
| A2    | 18.78 | 25.44 | **2.5**  | C3 |
| A3    | 43.78 | **1.11**  | 42.5 | C2 |
| A4    | **2.78**  | 17.44 | 32.5 | C1 |
| A5    | 27.11 | **0.44**  | 32.5 | C2 |
| A6    | 30.44 | **1.11**  | 20.5 | C2 |
| A7    | 56.11 | 41.44 | **2.5**  | C3 |
| A8    | **0.11**  | 30.78 | 36.5 | C1 |

Aucun point ne change de cluster → les centres ne bougent plus → **convergence**.

**Réponse** : après la 1ʳᵉ itération, il faut **2 itérations supplémentaires** qui modifient les clusters (itérations 2 et 3), puis une 3ᵉ (itération 4) qui sert seulement à **constater** que plus rien ne change.

Clustering final :
- C1 = {A1, A4, A8}, centre (3.67 ; 9)
- C2 = {A3, A5, A6}, centre (7 ; 4.33)
- C3 = {A2, A7}, centre (1.5 ; 3.5)

---

## Exercice 2 : K-means (DM)

### Énoncé

Proposez un ensemble de points tels que, lorsque l'on applique le k-means en 3 classes, les résultats dépendent de l'initialisation.

Vous montrerez que votre exemple est correct en détaillant les étapes et le clustering final pour des initialisations bien choisies.

### Réponse

**Idée** : k-means ne trouve qu'un **minimum local** de l'inertie intra-classe. Si deux centres initiaux tombent dans le même « groupe naturel », l'algorithme peut rester bloqué avec un groupe coupé en deux et deux autres groupes fusionnés.

**Données** : 6 points sur une droite (1 dimension), formant 3 paires bien séparées :

| Point | P1 | P2 | P3 | P4 | P5 | P6 |
|-------|----|----|----|----|----|----|
| x     | 0  | 1  | 5  | 6  | 10 | 11 |

```
P1 P2          P3 P4          P5 P6
 0  1  2  3  4  5  6  7  8  9 10 11
```

On affecte chaque point au centre le plus proche (distance $|x - c|$).

#### Initialisation 1 : centres = P1, P3, P5 (c = 0, 5, 10)

**Itération 1**

| Point | \|x−0\| | \|x−5\| | \|x−10\| | Cluster |
|-------|-----|-----|------|---------|
| P1 (0)  | **0** | 5 | 10 | C1 |
| P2 (1)  | **1** | 4 | 9  | C1 |
| P3 (5)  | 5 | **0** | 5  | C2 |
| P4 (6)  | 6 | **1** | 4  | C2 |
| P5 (10) | 10 | 5 | **0** | C3 |
| P6 (11) | 11 | 6 | **1** | C3 |

Nouveaux centres : $c_1 = 0.5$, $c_2 = 5.5$, $c_3 = 10.5$.

**Itération 2** : chaque point est à 0.5 de son centre et à au moins 4.5 des autres → aucun changement → **convergence**.

Résultat : **{P1, P2}, {P3, P4}, {P5, P6}**
Inertie intra-classe : $6 \times 0.5^2 = 1.5$ (c'est la bonne solution).

#### Initialisation 2 : centres = P1, P2, P3 (c = 0, 1, 5)

**Itération 1**

| Point | \|x−0\| | \|x−1\| | \|x−5\| | Cluster |
|-------|-----|-----|-----|---------|
| P1 (0)  | **0** | 1 | 5 | C1 |
| P2 (1)  | 1 | **0** | 4 | C2 |
| P3 (5)  | 5 | 4 | **0** | C3 |
| P4 (6)  | 6 | 5 | **1** | C3 |
| P5 (10) | 10 | 9 | **5** | C3 |
| P6 (11) | 11 | 10 | **6** | C3 |

Nouveaux centres : $c_1 = 0$, $c_2 = 1$, $c_3 = \frac{5+6+10+11}{4} = 8$.

**Itération 2**

| Point | \|x−0\| | \|x−1\| | \|x−8\| | Cluster |
|-------|-----|-----|-----|---------|
| P1 (0)  | **0** | 1 | 8 | C1 |
| P2 (1)  | 1 | **0** | 7 | C2 |
| P3 (5)  | 5 | 4 | **3** | C3 |
| P4 (6)  | 6 | 5 | **2** | C3 |
| P5 (10) | 10 | 9 | **2** | C3 |
| P6 (11) | 11 | 10 | **3** | C3 |

Aucun changement → **convergence**.

Résultat : **{P1}, {P2}, {P3, P4, P5, P6}**
Inertie intra-classe : $0 + 0 + (3^2 + 2^2 + 2^2 + 3^2) = 26$.

#### Conclusion

Avec les **mêmes données** et le **même k = 3**, on obtient deux partitions différentes selon l'initialisation :

| Initialisation | Clusters finaux | Inertie intra |
|----------------|-----------------|---------------|
| 0, 5, 10       | {0,1}, {5,6}, {10,11} | 1.5 |
| 0, 1, 5        | {0}, {1}, {5,6,10,11} | 26  |

La 2ᵉ solution est un **minimum local** : l'algorithme s'arrête car plus aucun point ne change de cluster, mais la partition est mauvaise. En pratique, on lance donc k-means **plusieurs fois** avec des initialisations différentes et on garde la solution de plus faible inertie (paramètre `n_init` de `sklearn.cluster.KMeans`), ou on utilise une initialisation plus intelligente comme **k-means++**.
