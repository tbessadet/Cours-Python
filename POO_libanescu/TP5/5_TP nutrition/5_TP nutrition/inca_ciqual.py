"""
TP INCA/Ciqual - Squelette étudiant
Ne modifiez pas les noms des fonctions demandées.
"""
from pathlib import Path

import pandas as pd

DOSSIER = Path(__file__).resolve().parent / "donnees"

individus = pd.read_csv(DOSSIER / "individus.csv")
groupes = pd.read_csv(DOSSIER / "groupes.csv")
aliments = pd.read_csv(DOSSIER / "aliments.csv")
constituants = pd.read_csv(DOSSIER / "constituants.csv")
composition = pd.read_csv(DOSSIER / "composition.csv")
consommations = pd.read_csv(DOSSIER / "consommations.csv")


def rechercher_aliment(texte):
    """Retourne les aliments dont le libellé contient texte."""
    # À compléter
    pass


def composition_aliment(code_aliment):
    """Retourne la composition d'un aliment, avec nom et unité du constituant."""
    # À compléter
    pass


def apport_aliment(code_aliment, quantite_g):
    """Calcule les apports pour une quantité donnée en grammes."""
    # À compléter
    pass


def apports_individu(id_individu):
    """Calcule les apports nutritionnels totaux d'un individu."""
    # À compléter
    pass


def apports_par_repas(id_individu):
    """Calcule les apports nutritionnels par repas pour un individu."""
    # À compléter
    pass


if __name__ == "__main__":
    print(rechercher_aliment("pain"))
