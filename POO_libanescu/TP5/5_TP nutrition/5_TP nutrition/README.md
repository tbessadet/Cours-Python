# TP INCA/Ciqual — Python, données et jointures

## Objectifs
- manipuler plusieurs tables CSV avec pandas ;
- comprendre des clés primaires et étrangères ;
- effectuer des jointures (`merge`) ;
- écrire des fonctions Python ;
- calculer des apports nutritionnels à partir d'une consommation ;
- raisonner sur la qualité et les limites d'un jeu de données.

## Contexte
Les données proposées sont SYNTHÉTIQUES. Elles sont inspirées de la logique de données de consommation alimentaire (INCA) et de composition nutritionnelle (Ciqual), mais ne constituent pas une copie des données officielles.

## Installation
Python 3.x et pandas.

    pip install pandas pytest

## Lancement
Depuis le dossier du TP :

    python inca_ciqual.py

Tests :

    pytest -q

## Règle importante
Les valeurs nutritionnelles sont exprimées pour 100 g. Pour une quantité q :

    apport = teneur * q / 100

