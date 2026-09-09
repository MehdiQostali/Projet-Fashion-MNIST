"""Fabriques de modèles utilisées dans le notebook Fashion-MNIST.

Ce fichier regroupe les fonctions qui construisent les classifieurs testés
dans le projet. Chaque fonction renvoie un modèle scikit-learn prêt à être
entraîné avec ``fit`` puis utilisé avec ``predict``.
"""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def baseline_classifier():
    """Crée un classifieur de référence très simple.

    Le modèle prédit toujours la classe la plus fréquente observée pendant
    l'entraînement. Il sert de point de comparaison minimal : un vrai modèle
    doit obtenir une meilleure accuracy que cette baseline.

    Exemple
    -------
    Entrée : ``baseline_classifier()``.
    Calcul : configure ``strategy="most_frequent"``.
    Sortie : ``DummyClassifier`` ; après ``fit``, prédit toujours la classe
    majoritaire, par exemple 0 si 0 est la classe la plus fréquente.
    """
    return DummyClassifier(strategy="most_frequent")


def perceptron_classifier(random_state=42):
    """Crée un perceptron avec standardisation préalable des variables.

    Paramètre
    ---------
    random_state : int
        Graine aléatoire utilisée pour rendre l'entraînement reproductible.

    Retour
    ------
    sklearn.pipeline.Pipeline
        Pipeline qui standardise les pixels puis entraîne un perceptron.

    Exemple
    -------
    Entrée : ``random_state=42``.
    Calcul : ``StandardScaler`` puis ``Perceptron(max_iter=1000)``.
    Sortie : ``Pipeline`` ; entrée future ``X.shape=(n, 784)``, sortie future
    ``y_pred.shape=(n,)``.
    """
    # Le perceptron est sensible à l'échelle des variables : on standardise
    # donc les pixels avant l'apprentissage.
    return make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=1000, tol=1e-3, random_state=random_state),
    )


def sgd_logistic_classifier(random_state=42):
    """Crée une régression logistique entraînée par descente de gradient.

    Paramètre
    ---------
    random_state : int
        Graine aléatoire utilisée pour rendre les résultats reproductibles.

    Retour
    ------
    sklearn.pipeline.Pipeline
        Pipeline qui standardise les données puis entraîne un classifieur
        linéaire avec une perte logistique.

    Exemple
    -------
    Entrée : ``random_state=42``.
    Calcul : ``StandardScaler`` puis ``SGDClassifier(loss="log_loss")``.
    Sortie : ``Pipeline`` de régression logistique ; après ``fit``, prédit une
    classe parmi ``0..9``.
    """
    # SGDClassifier avec loss="log_loss" correspond à une régression
    # logistique optimisée par descente de gradient stochastique.
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
            n_jobs=-1,
        ),
    )


def knn_classifier(n_neighbors=5):
    """Crée un classifieur des k plus proches voisins.

    Paramètre
    ---------
    n_neighbors : int
        Nombre de voisins utilisés pour décider de la classe prédite.

    Retour
    ------
    sklearn.neighbors.KNeighborsClassifier
        Modèle kNN pondéré par la distance : les voisins les plus proches ont
        davantage d'influence que les voisins plus éloignés.

    Exemple
    -------
    Entrée : ``n_neighbors=5``.
    Calcul : cherche les 5 voisins les plus proches, pondérés par distance.
    Sortie : ``KNeighborsClassifier`` ; pour 1 image test, renvoie 1 label.
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)


def decision_tree_classifier(random_state=42, max_depth=18):
    """Crée un arbre de décision.

    Paramètres
    ----------
    random_state : int
        Graine aléatoire utilisée pour rendre l'arbre reproductible.
    max_depth : int ou None
        Profondeur maximale de l'arbre. Une valeur limitée réduit le risque de
        sur-apprentissage.

    Retour
    ------
    sklearn.tree.DecisionTreeClassifier
        Arbre de décision entraînable avec ``fit``.

    Exemple
    -------
    Entrée : ``random_state=42`` et ``max_depth=18``.
    Calcul : construit un arbre limité à 18 niveaux.
    Sortie : ``DecisionTreeClassifier(max_depth=18)`` ; après ``fit``, prédit
    un label par image.
    """
    return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)


def random_forest_classifier(random_state=42, n_estimators=120, max_depth=None):
    """Crée une forêt aléatoire.

    Paramètres
    ----------
    random_state : int
        Graine aléatoire utilisée pour rendre la forêt reproductible.
    n_estimators : int
        Nombre d'arbres dans la forêt.
    max_depth : int ou None
        Profondeur maximale des arbres. ``None`` laisse les arbres se développer
        jusqu'aux critères d'arrêt internes.

    Retour
    ------
    sklearn.ensemble.RandomForestClassifier
        Ensemble d'arbres de décision entraînés sur des sous-échantillons.

    Exemple
    -------
    Entrée : ``n_estimators=120``, ``max_depth=None`` et ``random_state=42``.
    Calcul : configure 120 arbres, vote majoritaire entre arbres.
    Sortie : ``RandomForestClassifier(n_estimators=120)`` ; prédiction finale
    = classe la plus votée.
    """
    # n_jobs=-1 demande à scikit-learn d'utiliser tous les cœurs disponibles.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


__all__ = [
    "baseline_classifier",
    "perceptron_classifier",
    "sgd_logistic_classifier",
    "knn_classifier",
    "decision_tree_classifier",
    "random_forest_classifier",
]
