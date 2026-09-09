# Projet Fashion-MNIST

Projet d'analyse supervisée et non supervisée sur le jeu de données Fashion-MNIST.

Le notebook principal compare plusieurs modèles de classification, affiche des matrices de confusion, puis explore le clustering avec k-means et un clustering hiérarchique.

## Contenu

- `Projet/projet-Qostali1_Ziane2.ipynb` : notebook principal exécuté.
- `Projet/projet.html` : export HTML du notebook.
- `iads/` : fonctions Python utilisées par le notebook.
- `poster_Mehdi_Qostali_&_Hakim_Ziane.pdf` : poster du projet.
- `soutenance_sdd.pdf` : support de soutenance.

## Données

Les fichiers CSV Fashion-MNIST ne sont pas inclus dans le dépôt, car `fashion-mnist_train.csv` dépasse la limite GitHub de 100 Mo.

Place les fichiers suivants dans le dossier `data/` avant d'exécuter le notebook :

- `data/fashion-mnist_train.csv`
- `data/fashion-mnist_test.csv`

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Exécution

```powershell
jupyter lab
```

Ouvre ensuite `Projet/projet-Qostali1_Ziane2.ipynb` et exécute les cellules.

Pour lancer le notebook en ligne de commande :

```powershell
jupyter nbconvert --to notebook --execute --inplace Projet\projet-Qostali1_Ziane2.ipynb --ExecutePreprocessor.timeout=1800
```

## Résultats principaux

- Accuracy test binaire : `0.8535`
- Accuracy test multi-classe : `0.8700`
- Erreurs multi-classe : `1300 / 10000`
- ARI k-means avec `k=10` : `0.3847`
- NMI k-means avec `k=10` : `0.5355`
- ARI clustering hiérarchique : `0.4305`
- NMI clustering hiérarchique : `0.6109`
