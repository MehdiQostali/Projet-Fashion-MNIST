"""Fonctions utilitaires pour le projet Fashion-MNIST.

Ce fichier contient les fonctions de préparation des données et de
visualisation utilisées dans le notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def split_features_labels(dataframe, label_col="label", normalize=True):
    """Sépare les pixels et les labels d'un DataFrame Fashion-MNIST.

    Paramètres
    ----------
    dataframe : pandas.DataFrame
        Table contenant une colonne de labels et 784 colonnes de pixels.
    label_col : str
        Nom de la colonne contenant les classes.
    normalize : bool
        Si ``True``, les pixels sont divisés par 255 pour passer de l'intervalle
        ``[0, 255]`` à l'intervalle ``[0, 1]``.

    Retour
    ------
    tuple[numpy.ndarray, numpy.ndarray]
        ``X`` contient les pixels, ``y`` contient les labels.

    Exemple
    -------
    Entrée : DataFrame ``shape=(60000, 785)`` avec ``label`` + 784 pixels.
    Calcul : extrait ``y`` et divise les pixels par 255.
    Sortie : ``X.shape=(60000, 784)``, ``y.shape=(60000,)``.
    """
    y = dataframe[label_col].to_numpy(dtype=np.int64)
    X = dataframe.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    if normalize:
        X = X / 255.0
    return X, y


def stratified_sample(X, y, n_samples, random_state=42):
    """Prélève un échantillon stratifié dans un jeu de données.

    Paramètres
    ----------
    X : array-like
        Variables explicatives.
    y : array-like
        Labels associés.
    n_samples : int
        Nombre d'exemples à conserver.
    random_state : int
        Graine aléatoire utilisée pour rendre le tirage reproductible.

    Retour
    ------
    tuple
        Sous-ensemble ``(X_sample, y_sample)`` qui conserve les proportions de
        classes du jeu initial.

    Exemple
    -------
    Entrée : ``X.shape=(60000, 784)``, ``y.shape=(60000,)``,
    ``n_samples=5000``.
    Calcul : tirage stratifié.
    Sortie : ``X_sample.shape=(5000, 784)``, ``y_sample.shape=(5000,)``.
    """
    if n_samples >= len(y):
        return X, y
    # train_test_split avec stratify=y conserve la répartition des classes.
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=n_samples,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample


def binary_subset(X, y, class_a, class_b):
    """Construit un sous-problème de classification binaire.

    Paramètres
    ----------
    X : array-like
        Données complètes.
    y : array-like
        Labels complets.
    class_a, class_b : int
        Deux classes à conserver.

    Retour
    ------
    tuple[numpy.ndarray, numpy.ndarray]
        Données filtrées et labels binaires. La classe ``class_a`` devient 0 et
        la classe ``class_b`` devient 1.

    Exemple
    -------
    Entrée : ``class_a=0``, ``class_b=6``, ``X.shape=(60000, 784)``.
    Calcul : garde les labels 0 et 6, puis encode 0 -> 0 et 6 -> 1.
    Sortie : ``X_bin.shape=(12000, 784)``, ``y_bin`` contient seulement 0 et 1.
    """
    mask = np.isin(y, [class_a, class_b])
    X_bin = X[mask]
    y_bin = (y[mask] == class_b).astype(np.int64)
    return X_bin, y_bin


def class_distribution(y):
    """Calcule l'effectif de chaque classe.

    Paramètre
    ---------
    y : array-like
        Labels des exemples.

    Retour
    ------
    pandas.DataFrame
        Tableau contenant le label numérique, le nom lisible de la classe et
        son effectif.

    Exemple
    -------
    Entrée : ``y=[0, 0, 1, 6]``.
    Calcul : compte les labels.
    Sortie : DataFrame, ex. classe 0 -> effectif 2, classe 1 -> effectif 1.
    """
    counts = pd.Series(y).value_counts().sort_index()
    return pd.DataFrame(
        {
            "classe": counts.index,
            "nom": [CLASS_NAMES.get(int(label), str(label)) for label in counts.index],
            "effectif": counts.values,
        }
    )


def plot_images_grid(X, y=None, indices=None, n_cols=8, title=None):
    """Affiche une grille d'images Fashion-MNIST.

    Paramètres
    ----------
    X : array-like
        Images aplaties sous forme de vecteurs de 784 pixels.
    y : array-like ou None
        Labels des images. Si fourni, le nom de la classe est affiché au-dessus
        de chaque vignette.
    indices : iterable[int] ou None
        Indices des images à afficher. Si ``None``, les premières images sont
        utilisées.
    n_cols : int
        Nombre de colonnes de la grille.
    title : str ou None
        Titre global de la figure.

    Retour
    ------
    matplotlib.figure.Figure
        Figure Matplotlib créée.

    Exemple
    -------
    Entrée : ``X.shape=(10000, 784)``, ``indices=[0, 10, 25]``, ``n_cols=3``.
    Calcul : chaque vecteur 784 devient une image ``28 x 28``.
    Sortie : une figure Matplotlib avec 3 vignettes.
    """
    if indices is None:
        indices = np.arange(min(len(X), n_cols * 2))
    n_rows = int(np.ceil(len(indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.8 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, indices):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if y is not None:
            label = int(y[idx])
            ax.set_title(CLASS_NAMES.get(label, str(label)), fontsize=9)
    for ax in axes[len(indices) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix, title="Matrice de confusion"):
    """Affiche une matrice de confusion sous forme de carte de chaleur.

    Paramètres
    ----------
    matrix : pandas.DataFrame
        Matrice de confusion, avec les vraies classes en lignes et les classes
        prédites en colonnes.
    title : str
        Titre du graphique.

    Retour
    ------
    matplotlib.figure.Figure
        Figure Matplotlib contenant la matrice colorée.

    Exemple
    -------
    Entrée : matrice ``[[900, 100], [193, 807]]``.
    Calcul : affichage avec ``cmap="Blues"``.
    Sortie : figure ; bleu clair = petite valeur, bleu foncé = grande valeur.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    # La palette Blues affiche les petites valeurs en bleu clair et les grandes
    # valeurs en bleu foncé.
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe réelle")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


__all__ = [
    "CLASS_NAMES",
    "split_features_labels",
    "stratified_sample",
    "binary_subset",
    "class_distribution",
    "plot_images_grid",
    "plot_confusion_matrix",
]
