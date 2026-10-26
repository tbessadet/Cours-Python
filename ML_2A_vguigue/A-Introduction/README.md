# ML-2A -- Introduction et premier algorithme *à la main*

## Tutoriels: informations sur l'environnement


Ce tutoriel est basé sur des notebook jupyter, il requiert:
* une distribution scientifique de python, la plus répendue (et surtout très facile à installer et prendre en main) est anaconda [lien](https://www.anaconda.com)
  * les alternatives pour ceux qui sont à l'aise en infomratique et veulent un système plus léger & plus ouvert : `Miniconda`, `Python + pip + venv` (from scratch)
* [mac/linux] pour travailler sur les canevas proposés, il suffit ensuite de taper:
  jupyter-notebook <fichier.ipynb> 
* [windows] naviguer dans anaconda pour lancer jupyter-notebook puis charger le fichier depuis l'interface
* [option avancée] éventuellement un éditeur avancé capable de gérer les notebooks: VS Code [lien](https://code.visualstudio.com)
VS Code est très bien fait: à l'ouverture des fichiers, en fonction des extensions, il propose de télécharger des plugins pour gérer les spécificités desdits fichiers.


**L'introduction vise à mettre tout le monde à niveau sur:**
- python
- numpy, matplotlib (et un peu de pandas)

L'introduction est particulièrement dense... Mais la plupart des informations transmises sont optionnelles. L'enjeu est de survoler, de retenir l'endroit où se trouvent les informations pour les récupérer ultérieurement si besoin.

## Python

Rappels généraux sur python, depuis la syntaxe élémentaire jusqu'aux modélisations objets. Les tutoriels sont à la fois très rapides et très généraux: ça peut donner une impression *d'insurmontable* pour qui n'est pas très à l'aise avec le langage python... Mais **ATTENTION** on va en réalité travailler majoritairement en `numpy`, un environnement riche et assez différent du python de base.

$\Rightarrow$ il faut donc trouver un équilibre: maitriser un peu sans perdre trop de temps, l'essentiel étant de se concentrer sur numpy.

## Numpy

Le coeur de ce tutoriel: il faut impérativement maitriser cette bibliothèque qui forme quasiement un langage à part entière, situé juste entre `Matlab` et `R`.
Toute la librairie `scikit-learn`, dédiée au machine-learning, est basée sur `numpy`: c'est donc ici que se trouve le verrou à ouvrir pour accéder à l'ensemble des outils.


## Naïve Bayes: premier modèle à la main

Une sorte de tutoriel orienté données. L'objectif est de consolider la pratique à travers un cas d'usage ludique: classer des images de chiffres manuscrits provenant d'enveloppes postales (des codes postaux décomposés).

- Choisir une modélisation statistique
- Calculer les paramètres des distributions au sens du max de vraisemblance
- Prendre une décision et mesurer les performances

<img src="/A-Introduction/3-Naive-Bayes/notebooks/ressources/usps.png" width=200px>


## Gradient

Dernier exercice consacré à numpy: nous allons apprendre un régresseur par descente de gradient afin de réviser:
1. numpy
2. la problématique de la régression
3. l'algorithme de la descente de gradient [qui est à la base des réseaux de neurones!]

De nouveau, nous consolidons les bases de numpy/matplotlib... Tout en poursuivant le panorama du ML vers les problèmes de régression et de gradient. Ce dernier point est critique: le gradient est un outil central dans tous les réseaux de neurones et il est important d'avoir une représentation mentale de l'impact des différents réglages possibles.

<img src="/A-Introduction/4-Gradient/notebooks/fig/animation.gif" width=600px>