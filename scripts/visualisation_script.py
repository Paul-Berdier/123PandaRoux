import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fonction pour générer une matrice de corrélation
def visualisation_correlation_matrix(data, output_image):
    """
    Génère une matrice de corrélation à partir des colonnes numériques d'un DataFrame et sauvegarde l'image.

    Parameters:
        data (str): Chemin du fichier CSV contenant les données.
        output_image (str): Chemin pour sauvegarder l'image de la matrice de corrélation.
    """
    print("Chargement des données...")
    data = pd.read_csv(data)

    # Afficher les colonnes disponibles pour debug
    print(f"Colonnes du DataFrame : {data.columns}")

    # Filtrer uniquement les colonnes numériques
    data_numeric = data.select_dtypes(include=[np.number])

    # Afficher les colonnes retenues pour la corrélation
    print(f"Colonnes utilisées pour la corrélation : {data_numeric.columns}")

    # Création de la figure et calcul de la matrice de corrélation
    plt.figure(figsize=(12, 10))
    correlation_matrix = data_numeric.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Matrice de corrélation des variables')

    # Sauvegarde de l'image
    plt.savefig(output_image)
    print(f"Matrice de corrélation sauvegardée sous : {output_image}")
    plt.show()


def plot_learning_curve(estimator, X_train, y_train, title, output_file):
    """
    Trace la courbe d'apprentissage pour un modèle donné.

    Parameters:
        estimator: Le modèle à évaluer.
        X_train: Caractéristiques des données d'entraînement.
        y_train: Labels des données d'entraînement.
        title: Titre pour le graphique.
        output_file: Chemin pour sauvegarder l'image de la courbe d'apprentissage.
    """
    # Définir les tailles de jeu d'entraînement pour la courbe
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []

    print("Calcul des scores pour différentes tailles d'entraînement...")
    for train_size in train_sizes:
        # Convertir la taille en entier pour sélectionner les données
        size = int(train_size * len(X_train))
        X_partial = X_train[:size]
        y_partial = y_train[:size]

        # Entraîner le modèle sur une partie des données
        estimator.fit(X_partial, y_partial)
        train_scores.append(accuracy_score(y_partial, estimator.predict(X_partial)))
        test_scores.append(accuracy_score(y_train, estimator.predict(X_train)))

    # Tracer les courbes d'apprentissage
    plt.figure()
    plt.plot(train_sizes, train_scores, label="Training Accuracy")
    plt.plot(train_sizes, test_scores, label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Sauvegarde de l'image
    plt.savefig(output_file)
    print(f"Courbe d'apprentissage sauvegardée sous : {output_file}")
    plt.close()
