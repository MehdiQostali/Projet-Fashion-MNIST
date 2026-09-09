"""Paquet local ``iads`` utilisé par le notebook du projet.

Ce fichier expose les quatre modules du dossier :

- ``Classifiers`` : création des modèles supervisés ;
- ``utils`` : préparation des données et visualisations ;
- ``evaluation`` : mesures et tableaux d'évaluation ;
- ``Clustering`` : fonctions pour l'apprentissage non-supervisé.
"""

from . import Classifiers, Clustering, evaluation, utils

# Liste des modules exportés lorsque l'utilisateur écrit ``from iads import *``.
__all__ = ["Classifiers", "Clustering", "evaluation", "utils"]
