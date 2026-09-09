"""Fonctions de clustering utilisées dans le notebook du projet.

Ce fichier regroupe les fonctions d'évaluation non-supervisée : k-means,
composition des clusters et clustering hiérarchique.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def evaluate_kmeans_range(X, y_true, k_values, random_state=42):
    """Évalue k-means pour plusieurs valeurs de k.

    Paramètres
    ----------
    X : array-like
        Données utilisées par k-means, généralement après réduction de dimension.
    y_true : array-like
        Vraies classes. Elles ne servent pas à entraîner k-means, seulement à
        évaluer les clusters avec ARI et NMI.
    k_values : iterable[int]
        Valeurs de k à tester.
    random_state : int
        Graine aléatoire utilisée pour rendre l'initialisation reproductible.

    Retour
    ------
    pandas.DataFrame
        Tableau contenant l'inertie, le score silhouette, l'ARI et la NMI pour
        chaque valeur de k.

    Exemple
    -------
    Entrée : ``X.shape=(2500, 50)``, ``y_true.shape=(2500,)``,
    ``k_values=[5, 10, 15, 20]``.
    Calcul : un k-means par k, puis inertie, silhouette, ARI, NMI.
    Sortie : DataFrame, ex. pour ``k=10`` : ``ari=0.385``, ``nmi=0.536``.
    """
    rows = []
    for k in k_values:
        # n_init=10 lance plusieurs initialisations et garde la meilleure.
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertie": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "ari": adjusted_rand_score(y_true, labels),
                "nmi": normalized_mutual_info_score(y_true, labels),
            }
        )
    return pd.DataFrame(rows)


def cluster_label_table(cluster_labels, true_labels):
    """Décrit la composition réelle de chaque cluster.

    Paramètres
    ----------
    cluster_labels : array-like
        Numéro de cluster attribué à chaque exemple.
    true_labels : array-like
        Vraie classe de chaque exemple.

    Retour
    ------
    pandas.DataFrame
        Tableau croisé indiquant combien d'exemples de chaque vraie classe se
        trouvent dans chaque cluster, plus la classe majoritaire et la pureté.

    Exemple
    -------
    Entrée : ``cluster_labels=[0, 0, 1]`` et ``true_labels=[8, 8, 3]``.
    Calcul : comptage par cluster puis classe majoritaire.
    Sortie : cluster 0 -> classe majoritaire 8, ``pureté=1.0`` ; cluster 1 ->
    classe majoritaire 3, ``pureté=1.0``.
    """
    # Le tableau croisé permet de comparer les clusters non-supervisés aux
    # vraies classes, uniquement pour l'analyse a posteriori.
    table = pd.crosstab(
        pd.Series(cluster_labels, name="cluster"),
        pd.Series(true_labels, name="classe_réelle"),
    )
    # Classe la plus représentée dans chaque cluster.
    dominant = table.idxmax(axis=1).rename("classe_majoritaire")
    # Pureté = proportion de la classe majoritaire dans le cluster.
    purity = (table.max(axis=1) / table.sum(axis=1)).rename("pureté")
    return pd.concat([table, dominant, purity], axis=1)


def agglomerative_labels(X, n_clusters=10, linkage="ward"):
    """Calcule les labels d'un clustering hiérarchique agglomératif.

    Paramètres
    ----------
    X : array-like
        Données à regrouper.
    n_clusters : int
        Nombre de groupes souhaités après découpe de la hiérarchie.
    linkage : str
        Critère de fusion des groupes. La valeur ``"ward"`` cherche à minimiser
        la variance intra-cluster.

    Retour
    ------
    numpy.ndarray
        Label de cluster attribué à chaque exemple.

    Exemple
    -------
    Entrée : ``X.shape=(500, 20)``, ``n_clusters=10``, ``linkage="ward"``.
    Calcul : fusions hiérarchiques puis découpe en 10 groupes.
    Sortie : ``labels.shape=(500,)`` avec des labels de clusters.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X)


__all__ = ["evaluate_kmeans_range", "cluster_label_table", "agglomerative_labels"]
