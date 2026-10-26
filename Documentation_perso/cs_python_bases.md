# 🐍 Cheat Sheet Python — Les bases

> Basée sur les notebooks *Tuto python*, *Tuto python - boucles & clauses*, *Tuto python avancé* et *Python objet*.

## Sommaire
1. [Variables, types de base et affichage](#1-variables-types-de-base-et-affichage)
2. [Chaînes de caractères (`str`)](#2-chaînes-de-caractères-str)
3. [Listes](#3-listes)
4. [Dictionnaires](#4-dictionnaires)
5. [Conditions et opérateurs logiques](#5-conditions-et-opérateurs-logiques)
6. [Boucles](#6-boucles)
7. [Compréhensions de listes](#7-compréhensions-de-listes)
8. [Fonctions](#8-fonctions)
9. [Types de base vs objets : copie ou référence ?](#9-types-de-base-vs-objets--copie-ou-référence-)
10. [Imports et modules](#10-imports-et-modules)
11. [Programmation objet : classes et instances](#11-programmation-objet--classes-et-instances)
12. [Méthodes spéciales (dunder methods)](#12-méthodes-spéciales-dunder-methods)
13. [Héritage](#13-héritage)
14. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Variables, types de base et affichage

En python, une variable n'a pas de type déclaré à l'avance : le type est **inféré automatiquement** à partir de la valeur affectée, et peut changer à chaque nouvelle affectation.

| Fonction / usage | Explication |
|---|---|
| `a = 1` | Affectation. Pas de déclaration de type au préalable. |
| `type(a)` | Renvoie le type de la variable (`int`, `float`, `str`, `bool`, `list`, `dict`...). |
| `print(a)` | Affichage simple. |
| `print("a = {}, mess = {}".format(a, message))` | Affichage **formaté** : insère des variables dans un texte via `{}`. |
| `f"a = {a}, mess = {message}"` | **f-string** : équivalent plus moderne et lisible de `.format()`, à privilégier. La variable est directement insérée entre accolades dans la chaîne. |

```python
a = 1
print(type(a))        # <class 'int'>
a = 45                 # la valeur précédente est perdue
b = 18.5
b = b + a              # 63.5, b devient un float

nom, age = "Sophie", 24
print(f"{nom} a {age} ans")   # f-string, syntaxe recommandée
```

⚠️ **Piège des notebooks** : l'état des variables dépend de **l'ordre d'exécution des cellules**, pas de leur ordre visuel dans le fichier. Exécuter une cellule du bas puis remonter exécuter une cellule du haut peut donner des résultats surprenants — toujours vérifier en cas de doute (ou tout ré-exécuter depuis le début : "Restart & Run All").

### Fonctions mathématiques et aléatoire (module `math` et `random`)

⚠️ Ces fonctions sont utiles pour du python "pur", mais dès qu'on travaille avec des tableaux/matrices, on utilisera plutôt les équivalents **numpy** (`np.sin`, `np.random.rand`...), qui fonctionnent aussi sur des tableaux entiers et pas seulement sur un nombre à la fois.

| Fonction | Explication |
|---|---|
| `math.pi`, `math.sqrt(x)`, `math.sin(x)`, `math.cos(x)`, `math.acos(x)` | Constantes et fonctions mathématiques classiques (angles en radians). |
| `random.random()` | Flottant aléatoire uniforme entre 0 et 1. |
| `random.randint(a, b)` | Entier aléatoire entre `a` et `b`, **bornes incluses des deux côtés** (différent de `range` !). |

---

## 2. Chaînes de caractères (`str`)

Une chaîne se comporte comme un **tableau de caractères** : indexable, avec de nombreuses méthodes dédiées.

| Fonction / syntaxe | Explication |
|---|---|
| `s[i]` | Caractère à l'indice `i` (indices commencent à **0**). |
| `s.find(sous_chaine)` | Renvoie la **position** de la première occurrence, ou `-1` si absente. Sensible à la casse. |
| `s.split(sep)` | Découpe la chaîne en liste de mots (par défaut sur les espaces). |
| `s.replace(a, b)` | Remplace toutes les occurrences de `a` par `b`. |
| `s.lower()` / `s.upper()` | Passage en minuscules / majuscules. |
| `s.strip()` | Supprime les espaces (ou caractères indiqués) en début/fin de chaîne. |

```python
s = "Bonjour et bienvenue"
print(s[2])              # 'n'
print(s.find('bienvenue'))  # position du mot
for mot in s.split():    # découpage en mots + boucle
    print(mot)
```

---

## 3. Listes

Structure de données **la plus utilisée** en python. ⚠️ Une liste est un **objet** (voir §9) : elle peut être modifiée "en place" et est **partagée** entre variables lors d'une simple affectation.

| Fonction / syntaxe | Paramètres essentiels | Explication |
|---|---|---|
| `[]` ou `list()` | — | Liste vide. |
| `liste.append(x)` | `x` = élément à ajouter | Ajoute un élément **à la fin**. |
| `liste + [a, b]` | — | Concatène deux listes (crée une **nouvelle** liste). |
| `len(liste)` | — | Nombre d'éléments. |
| `liste[i] = x` | — | Modifie l'élément d'indice `i`. |
| `liste.pop(i)` | `i` = indice (déf. dernier élément) | Supprime **et renvoie** l'élément à l'indice `i`. |
| `liste.remove(valeur)` | `valeur` | Supprime la **première occurrence** de cette valeur (par valeur, pas par indice). |
| `liste.sort()` | — | Trie la liste **en place** (modifie `liste`, ne renvoie rien). |
| `sorted(liste)` | — | Renvoie une **nouvelle** liste triée, sans modifier l'originale. |
| `liste.count(valeur)` | `valeur` | Nombre d'occurrences d'une valeur. |
| `sum(liste)` | — | Somme des éléments (fonction native python, pas une méthode de liste). |
| `set(liste)` | — | Convertit en **ensemble** (valeurs uniques, non ordonné). Pratique pour dédupliquer. |
| `valeur in liste` | — | Test d'appartenance (renvoie `True`/`False`). |
| `liste.copy()` | — | Crée une **vraie copie indépendante** de la liste (voir §9 pour comprendre pourquoi c'est important). |

```python
li = [12, 15, 3, 15, 9]
li.sort()                    # tri en place -> [3, 9, 12, 15, 15]
print(li.count(15))          # 2
print(set(li))               # {3, 9, 12, 15} - valeurs uniques
if 12 in li:
    print("trouvé !")
```

---

## 4. Dictionnaires

Structure **clé → valeur**, centrale en python (l'équivalent d'une table de hash). Peu utilisée pour manipuler de grosses données numériques (on préfère numpy/pandas), mais très utile pour stocker des informations hétérogènes ou faire des correspondances (traductions, comptages par catégorie, configuration...).

| Fonction / syntaxe | Explication |
|---|---|
| `d = {'cle1': valeur1, 'cle2': valeur2}` | Construction. |
| `d['cle']` | Lecture de la valeur associée à une clé (erreur si la clé n'existe pas). |
| `d.get('cle', defaut)` | Lecture "sûre" : renvoie `defaut` si la clé est absente, au lieu de planter. |
| `d['nouvelle_cle'] = valeur` | Ajout (ou modification si la clé existe déjà) — même syntaxe pour les deux. |
| `d.keys()` / `d.values()` / `d.items()` | Listes des clés / des valeurs / des paires (clé, valeur). |
| `cle in d` | Test d'appartenance **sur les clés**. |
| `{k: f(k) for k in liste}` | Compréhension de dictionnaire (construit un dico en une ligne à partir d'une liste). |

```python
trad = {"le": "the", "chat": "cat", "table": "table"}
print(trad['chat'])                    # 'cat'
trad['sur'] = 'on'                     # ajout d'une clé

phrase = "le chat est sur la table"
for mot in phrase.split():
    print(trad.get(mot, mot))          # traduit le mot, ou l'affiche tel quel si absent
```

---

## 5. Conditions et opérateurs logiques

Python **n'utilise pas d'accolades** : les blocs sont délimités par l'**indentation** (généralement 4 espaces ou une tabulation, à ne jamais mélanger).

| Syntaxe | Explication |
|---|---|
| `if cond: ... elif cond2: ... else: ...` | `elif`/`else` optionnels. Dès qu'une condition est vraie, les suivantes ne sont **pas évaluées** (choix exclusif). |
| `and`, `or`, `not` | Opérateurs logiques de base. |
| Deux `if` successifs (sans `elif`) | Les deux conditions sont évaluées **indépendamment** — ce n'est **pas** la même chose qu'un `if`/`elif` (voir piège ci-dessous). |

```python
i = 2
if i < 3:
    print("i inférieur à 3")
elif i < 10:
    print("i inférieur à 10")   # PAS affiché car le if précédent était déjà vrai

if i < 3:
    print("i inférieur à 3")
if i < 10:
    print("i inférieur à 10")   # AFFICHÉ cette fois : deux tests indépendants
```

⚠️ **Piège d'indentation** : une ligne indentée est **dans** le bloc du `if` (exécutée seulement si la condition est vraie) ; une ligne revenue au même niveau que le `if` est **après** le bloc (toujours exécutée). Toujours vérifier visuellement l'alignement du code.

---

## 6. Boucles

| Syntaxe | Explication |
|---|---|
| `for i in range(n):` | Parcourt les **indices** `0` à `n-1`. Pratique pour accéder à `liste[i]`. |
| `for element in liste:` | Parcourt directement les **valeurs** de la liste. Plus lisible quand on n'a pas besoin de l'indice. |
| `for i, element in enumerate(liste):` | Parcourt **à la fois** l'indice et la valeur. |
| `for a, b in zip(liste1, liste2):` | Parcourt **en parallèle** deux listes de même taille. |
| `while condition:` | Répète tant que la condition est vraie. ⚠️ Il faut que la condition finisse par devenir fausse, sinon **boucle infinie** (typiquement : oubli d'incrémenter le compteur). |

```python
for i in range(10):        # 0, 1, ..., 9
    print(i)

li = [12, 43, 90]
for val in li:              # parcours direct des valeurs
    print(val)

for i, val in enumerate(li):
    print(i, val)

i = 0
while i < 10:
    i += 1                  # indispensable pour sortir de la boucle !
```

---

## 7. Compréhensions de listes

Syntaxe compacte pour construire une liste à partir d'une boucle (et éventuellement d'une condition), en une seule ligne. Plus rapide qu'une boucle classique en python, et très utilisée une fois qu'on est à l'aise avec.

⚠️ Peut devenir illisible si on empile trop de conditions/boucles — à réserver aux cas simples ou une fois le concept bien maîtrisé.

```python
a = [n for n in range(10)]                       # [0, 1, ..., 9]
pairs = [n for n in range(20) if n % 2 == 0]     # avec filtre
m = [[n + m*10 for n in range(5)] for m in range(5)]  # boucle imbriquée -> liste de listes
```

---

## 8. Fonctions

| Élément | Explication |
|---|---|
| `def ma_fonction(a, b=10):` | Déclaration. `b=10` = argument avec **valeur par défaut** — ⚠️ les arguments avec valeur par défaut doivent toujours être placés **après** les arguments obligatoires. |
| `"""docstring"""` | Chaîne de documentation juste après la déclaration : décrit ce que fait la fonction et ses arguments. Bonne pratique systématique. |
| `return valeur` | Renvoie un résultat. Sans `return`, la fonction renvoie `None`. |
| `ma_fonction(5)` / `ma_fonction(b=5, a=2)` | Appel positionnel, ou **nommé** (l'ordre n'importe plus si on nomme les arguments). |

```python
def calcul_angle(u, v):
    """
    Calcule l'angle (en radians) entre deux vecteurs 2D u=[ux,uy] et v=[vx,vy].
    """
    produit_scalaire = u[0]*v[0] + u[1]*v[1]
    norme_u = math.sqrt(u[0]**2 + u[1]**2)
    norme_v = math.sqrt(v[0]**2 + v[1]**2)
    return math.acos(produit_scalaire / (norme_u * norme_v))

def affichage(nIter=10):     # valeur par défaut
    for i in range(nIter + 1):
        print(i)

affichage(5)   # utilise 5
affichage()    # utilise la valeur par défaut, 10
```

⚠️ **Piège des variables globales** : une fonction ne devrait **jamais** dépendre d'une variable définie en dehors d'elle (autre que ses arguments). Sinon, le résultat de la fonction peut changer entre deux appels identiques, ce qui rend le code très difficile à déboguer.
```python
glob = 2
def mafonction(a):
    return a * glob     # dépend d'une variable extérieure = À ÉVITER

print(mafonction(4))    # 8
glob = 3
print(mafonction(4))    # 12 !! même appel, résultat différent
```

---

## 9. Types de base vs objets : copie ou référence ?

C'est l'un des concepts les plus déroutants pour un débutant, mais **essentiel** à comprendre.

- **Types de base** (`int`, `float`, `str`, `bool`) : à chaque affectation, la **valeur est copiée**. Modifier une variable n'affecte jamais une autre variable, même si elles avaient la même valeur au départ.
- **Objets** (`list`, `dict`, et plus généralement les instances de classes) : une affectation ne copie **pas** l'objet, elle crée une **deuxième référence vers le même objet**. Modifier l'objet via une variable le modifie donc **aussi** vu depuis l'autre variable.

```python
# type de base : aucun piège
a = 2
b = a
a = 18
print(a, b)          # 18 2  -> b n'a pas changé

# objet (liste) : piège classique
li_a = [2, 4, 6]
li_b = li_a           # li_b et li_a pointent vers LA MÊME liste
li_b.append(8)
print(li_a)           # [2, 4, 6, 8]  -> li_a a aussi changé !

# pour éviter ça : faire une vraie copie
li_c = li_a.copy()
li_c.append(42)
print(li_a)           # inchangé
```

| Test | Explication |
|---|---|
| `a is b` | Teste si `a` et `b` désignent **le même objet en mémoire** (égalité référentielle), pas juste des valeurs égales. |
| `a == b` | Teste l'**égalité de valeur** (ce qu'on veut généralement comparer). |

**Conséquence dans les fonctions** : une variable de type de base passée en argument ne peut pas être modifiée par la fonction (elle est copiée à l'entrée) ; un objet passé en argument, lui, peut être modifié "en place" par la fonction, et le changement persiste après l'appel.

```python
def modif_base(a):
    a = 2
    return a

i = 5
modif_base(i)
print(i)              # 5, inchangé : i est un int

def modif_objet(a):
    a.append(6)

li = [5]
modif_objet(li)
print(li)              # [5, 6], modifié ! li est une liste
```

---

## 10. Imports et modules

| Syntaxe | Explication |
|---|---|
| `import module` | Importe **tout** le module ; chaque fonction s'appelle avec le préfixe `module.fonction()`. |
| `from module import fonction` | Importe une fonction précise ; appel direct `fonction()`, sans préfixe. |
| `from module import *` | Importe **tout** sans préfixe (pratique mais risque de conflits de noms — à utiliser avec modération). |
| `from mon_fichier import *` | Fonctionne aussi avec un fichier `.py` local (ex. `mesfonctions.py`), placé dans le même dossier que le notebook/script : permet de séparer du code réutilisable dans un fichier dédié plutôt que de tout mettre dans le notebook. |

```python
import math
print(math.sqrt(2))

from math import sqrt
print(sqrt(2))          # pas besoin du préfixe math.
```

---

## 11. Programmation objet : classes et instances

Une **classe** est un plan de construction générique (comme une fonction) ; une **instance** est un objet concret créé à partir de ce plan, avec ses propres valeurs.

| Élément | Explication |
|---|---|
| `class NomClasse:` | Déclare une nouvelle classe. |
| `def __init__(self, params):` | **Constructeur** : appelé automatiquement à la création d'une instance, pour initialiser ses attributs. |
| `self` | Fait référence à **l'instance elle-même**, à l'intérieur de la classe. Premier argument de **toutes** les méthodes, mais jamais fourni explicitement à l'appel. |
| `self.attribut = valeur` | Déclare/stocke un **attribut** (donnée propre à l'instance). ⚠️ Sans `self.`, la variable reste locale à la méthode et disparaît à la fin de son exécution — elle n'est **pas** accessible depuis l'extérieur. |
| `instance.methode()` | Invoque une méthode sur une instance. |
| `instance.attribut` | Accède à un attribut de l'instance. |

```python
class Voiture:
    def __init__(self, couleur, marque):
        self.couleur = couleur
        self.marque = marque
        self.vitesse = 0

    def accelerer(self):
        self.vitesse += 10
        print(f"La {self.marque} roule à {self.vitesse} km/h.")

    def freiner(self):
        self.vitesse = max(0, self.vitesse - 10)

v1 = Voiture("rouge", "Ferrari")   # instance 1
v2 = Voiture("bleue", "Renault")   # instance 2, indépendante de v1

v1.accelerer()   # affecte seulement v1
print(v1.couleur, v2.couleur)   # chaque instance garde ses propres valeurs
```

⚠️ **Pièges classiques** :
- Oublier `self.` dans le constructeur → l'attribut n'existe pas en dehors de `__init__`, erreur à l'utilisation.
- Confondre le **nom de l'argument** du constructeur et le **nom de l'attribut** stocké (rien n'oblige à ce qu'ils soient identiques, mais c'est plus clair si c'est le cas).
- Faire dépendre une classe d'une **variable globale** dans son constructeur → deux instances créées avec le même appel peuvent se retrouver avec des valeurs différentes selon le moment de la création (même défaut que pour les fonctions, voir §8).
- Arguments par défaut du constructeur : même règle que pour les fonctions, ils doivent être placés **en dernier** (ex. `def __init__(self, nom, age=0):`).

---

## 12. Méthodes spéciales (dunder methods)

Ce sont des méthodes au nom **encadré de doubles underscores**, invoquées **automatiquement** par python dans certaines situations (affichage, comparaison, opérateurs...), sans qu'on les appelle explicitement par leur nom.

| Méthode | Invoquée quand... | Explication |
|---|---|---|
| `__init__(self, ...)` | `NomClasse(...)` | Le constructeur (déjà vu ci-dessus). |
| `__str__(self)` | `print(instance)` ou `str(instance)` | Doit renvoyer une **chaîne de caractères** décrivant l'objet. Sans elle, `print` affiche une représentation technique peu lisible (adresse mémoire). |
| `__eq__(self, other)` | `instance1 == instance2` | Définit ce que signifie "être égal" pour deux instances (par défaut, sans cette méthode, `==` compare les **adresses mémoire**, comme `is`). |
| `__add__(self, other)` | `instance1 + instance2` | Définit le comportement de l'opérateur `+` entre deux instances. |

```python
class Personne:
    def __init__(self, nom, age=0):
        self.nom = nom
        self.age = age

    def __str__(self):
        return f"nom: {self.nom}, {self.age} ans"

    def __eq__(self, other):
        return self.nom == other.nom and self.age == other.age

    def __add__(self, other):
        return Personne(self.nom + other.nom, self.age + other.age)

p1 = Personne("Vincent", 35)
p2 = Personne("Vincent", 35)
print(p1)          # utilise __str__ : "nom: Vincent, 35 ans"
print(p1 == p2)    # True grâce à __eq__ (sinon False : deux objets différents en mémoire)
```

---

## 13. Héritage

Permet de créer une classe (**fille**) qui réutilise les attributs et méthodes d'une classe existante (**mère**), en n'ajoutant que ce qui lui est spécifique.

| Élément | Explication |
|---|---|
| `class Fille(Mere):` | Déclare que `Fille` hérite de `Mere` : elle possède automatiquement toutes ses méthodes. |
| `super().__init__(...)` | Appelle le constructeur de la classe mère, pour initialiser les attributs communs, avant d'ajouter les attributs spécifiques à la classe fille. |
| `isinstance(objet, Classe)` | Teste si `objet` est une instance de `Classe` **ou d'une de ses classes filles**. |
| `type(objet) == Classe` | Teste l'**égalité stricte de type** (contrairement à `isinstance`, une instance de la classe fille ne "matche" pas ici). |

```python
class Vehicule:
    def __init__(self, couleur):
        self.couleur = couleur
        self.vitesse = 0

    def accelerer(self):
        self.vitesse += 10

class Voiture(Vehicule):
    def __init__(self, marque, couleur):
        super().__init__(couleur)   # initialise couleur et vitesse via la classe mère
        self.marque = marque        # attribut spécifique à Voiture

v1 = Voiture("Ferrari", "rouge")
v1.accelerer()                      # méthode héritée, pas besoin de la redéfinir

print(isinstance(v1, Vehicule))     # True  : une Voiture EST UN Vehicule
v0 = Vehicule("bleu")
print(isinstance(v0, Voiture))      # False : un Vehicule n'est pas forcément une Voiture
```

👉 **Intérêt principal** : factoriser le code commun (ex. gérer la vitesse) dans la classe mère, et ne coder que les spécificités (ex. la marque) dans chaque classe fille — utile dès qu'on a plusieurs variantes d'un même concept (`Voiture`, `Camion`, `Moto`... toutes des `Vehicule`).

---

## 📌 Récapitulatif des pièges classiques

1. **Listes et dictionnaires sont des objets** : une simple affectation (`li_b = li_a`) ne copie pas la liste, elle crée une deuxième référence vers la même liste. Utiliser `.copy()` pour une vraie copie indépendante.
2. **`a is b` ≠ `a == b`** : le premier teste l'identité en mémoire, le second l'égalité de valeur.
3. **Deux `if` successifs ≠ `if`/`elif`** : dans le premier cas, les deux conditions sont testées indépendamment ; dans le second, une seule branche s'exécute.
4. **Indentation = portée du bloc** en python (pas d'accolades) : une erreur d'alignement change complètement le comportement du programme.
5. **Boucle `while` sans mise à jour du compteur** → boucle infinie.
6. **Fonctions dépendant de variables globales** → résultat non reproductible d'un appel à l'autre. Toujours passer les valeurs nécessaires en argument.
7. **Arguments par défaut en fin de liste uniquement** (`def f(a, b=0):`, jamais `def f(a=0, b):`).
8. **Oublier `self.`** dans une classe → l'attribut ne persiste pas au-delà de la méthode où il est créé.
9. **`__eq__` non défini** → `==` compare l'identité mémoire des objets, pas leurs attributs (deux instances "identiques en apparence" seront jugées différentes).
10. **`isinstance` vs `type() ==`** : préférer `isinstance` dès qu'on utilise de l'héritage, car `type() ==` ignore les relations de filiation entre classes.
