"""Fonctions d'évaluation pour les expériences supervisées.

Ce fichier contient les outils utilisés pour comparer les modèles supervisés :
validation croisée, entraînement final, matrice de confusion et rapport de
classification.
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate


def cross_validate_classifiers(classifiers, X, y, cv=3, random_state=42):
    """Lance une validation croisée stratifiée pour plusieurs classifieurs.

    Paramètres
    ----------
    classifiers : dict[str, estimator]
        Dictionnaire associant un nom lisible à un modèle scikit-learn.
    X : array-like
        Variables explicatives, ici les pixels des images.
    y : array-like
        Labels associés aux images.
    cv : int
        Nombre de folds de validation croisée.
    random_state : int
        Graine aléatoire utilisée pour mélanger les folds de façon
        reproductible.

    Retour
    ------
    pandas.DataFrame
        Tableau trié par accuracy moyenne décroissante, avec le temps total et
        les scores obtenus fold par fold.

    Exemple
    -------
    Entrée : ``classifiers={"kNN": knn}``, ``X.shape=(5000, 784)``,
    ``y.shape=(5000,)``, ``cv=3``.
    Calcul : 3 folds stratifiés, puis accuracy par fold.
    Sortie : DataFrame, ex. ``accuracy_moyenne=0.834`` et
    ``scores_folds=[0.8338, 0.8338, 0.8355]``.
    """
    # StratifiedKFold conserve la proportion des classes dans chaque fold.
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, model in classifiers.items():
        # On mesure le temps global de la validation croisée pour comparer
        # grossièrement le coût des modèles.
        started = time.perf_counter()
        scores = cross_validate(
            model,
            X,
            y,
            cv=splitter,
            scoring="accuracy",
            n_jobs=None,
            return_train_score=False,
        )
        rows.append(
            {
                "modèle": name,
                "accuracy_moyenne": scores["test_score"].mean(),
                "accuracy_écart_type": scores["test_score"].std(),
                "temps_total_s": time.perf_counter() - started,
                "scores_folds": [round(float(v), 4) for v in scores["test_score"]],
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_moyenne", ascending=False)


def fit_and_score(model, X_train, y_train, X_test, y_test):
    """Entraîne un modèle et l'évalue sur un jeu de test.

    Paramètres
    ----------
    model : estimator
        Modèle scikit-learn disposant des méthodes ``fit`` et ``predict``.
    X_train, y_train
        Données et labels d'entraînement.
    X_test, y_test
        Données et labels de test.

    Retour
    ------
    dict
        Dictionnaire contenant le modèle entraîné, les prédictions, l'accuracy,
        le temps d'entraînement et le temps de prédiction.

    Exemple
    -------
    Entrée : ``X_train.shape=(20000, 784)``, ``X_test.shape=(10000, 784)``.
    Calcul : ``fit`` puis ``predict`` puis ``accuracy_score``.
    Sortie : ``{"accuracy": 0.8700, "predictions": y_pred, ...}``.
    """
    # Premier chronomètre : durée de l'apprentissage.
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    # Deuxième chronomètre : durée de la prédiction sur le jeu de test.
    started = time.perf_counter()
    predictions = model.predict(X_test)
    pred_time = time.perf_counter() - started
    return {
        "model": model,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "fit_time_s": fit_time,
        "predict_time_s": pred_time,
    }


def confusion_dataframe(y_true, y_pred, labels, names=None):
    """Construit une matrice de confusion sous forme de DataFrame.

    Paramètres
    ----------
    y_true : array-like
        Vraies classes.
    y_pred : array-like
        Classes prédites par le modèle.
    labels : list
        Ordre des labels à utiliser dans la matrice.
    names : list ou None
        Noms lisibles des classes. Si ``None``, les labels numériques sont
        utilisés comme noms de lignes et de colonnes.

    Retour
    ------
    pandas.DataFrame
        Matrice dont les lignes sont les vraies classes et les colonnes les
        classes prédites.

    Exemple
    -------
    Entrée : ``y_true=[0, 0, 1]``, ``y_pred=[0, 1, 1]``,
    ``labels=[0, 1]`` et ``names=["T-shirt", "Shirt"]``.
    Calcul : compte les couples ``(vrai, prédit)``.
    Sortie : DataFrame ``[[1, 1], [0, 1]]``.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    index = names if names is not None else labels
    return pd.DataFrame(matrix, index=index, columns=index)


def classification_report_dataframe(y_true, y_pred, target_names=None):
    """Transforme le rapport de classification scikit-learn en DataFrame.

    Paramètres
    ----------
    y_true : array-like
        Vraies classes.
    y_pred : array-like
        Classes prédites.
    target_names : list ou None
        Noms lisibles des classes.

    Retour
    ------
    pandas.DataFrame
        Tableau contenant precision, recall, f1-score et support pour chaque
        classe, ainsi que les moyennes globales.

    Exemple
    -------
    Entrée : ``y_true.shape=(10000,)``, ``y_pred.shape=(10000,)``.
    Calcul : precision, recall, f1-score et support par classe.
    Sortie : DataFrame, ex. ligne ``Shirt`` avec ``recall=0.566``.
    """
    # zero_division=0 évite les avertissements si une classe n'est jamais
    # prédite par un modèle.
    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    ).T


__all__ = [
    "cross_validate_classifiers",
    "fit_and_score",
    "confusion_dataframe",
    "classification_report_dataframe",
]
