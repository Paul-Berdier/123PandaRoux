from operator import index

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def column_mapping(input_file, output_file, mapping, column_name):
    input_data = pd.read_csv(input_file)

    input_data[column_name] = input_data[column_name].map(mapping)

    input_data.to_csv(output_file, index=False)

# Fonction pour normaliser les données
def normalize_data(input_file, output_file):
    # Charger les données
    data = pd.read_csv(input_file)

    # Séparer les features et la colonne cible 'Class'
    features = data.drop(columns=['Class'])
    target = data['Class']

    # Appliquer la normalisation uniquement sur les features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Réintégrer la colonne 'Class'
    scaled_data = pd.concat([scaled_features, target], axis=1)
    scaled_data.to_csv(output_file, index=False)


# Fonction pour calculer des statistiques essentielles
def compute_statistics(input_file, output_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    stats = data.describe().T
    stats['median'] = data.median()
    stats['IQR'] = stats['75%'] - stats['25%']
    stats.to_csv(output_file)

def correlation_matrix(input_file, output_file, threshold=0.1):
    # Recharger le fichier dataset réduit
    data = pd.read_csv(input_file)

    # Calculer la matrice de corrélation avec la variable cible 'Class'
    correlation_matrix = data.corr()
    correlations_with_class = correlation_matrix['Class'].sort_values(ascending=False)

    # Identifier les variables avec une corrélation absolue > 0.1 (seuil)
    important_features = correlations_with_class[abs(correlations_with_class) > threshold].index.tolist()

    # Réduire la dimensionnalité en conservant uniquement les colonnes pertinentes
    reduced_data = data[important_features]

    # Sauvegarder le dataset réduit
    reduced_data.to_csv(output_file, index=False)

    # Afficher les résultats
    print(important_features, reduced_data.shape)

def isolate_random_row(data_file, output_data_file, isolated_row_file, target_column='Class'):
    """
    Isoler une ligne aléatoire du dataset pour une prédiction finale.
    La ligne isolée ne sera pas utilisée pour l'entraînement.

    Parameters:
        data_file (str): Chemin vers le fichier contenant les données originales.
        output_data_file (str): Chemin pour sauvegarder le dataset sans la ligne isolée.
        isolated_row_file (str): Chemin pour sauvegarder la ligne isolée.
        target_column (str): Nom de la colonne cible (par défaut 'Class').

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(data_file)

    # Vérifier que le dataset n'est pas vide
    if data.empty:
        raise ValueError("Le dataset est vide. Veuillez vérifier le fichier source.")

    print("Isolation d'une ligne aléatoire...")
    isolated_row = data.sample(n=1, random_state=42)

    # Supprimer la ligne isolée du dataset
    data = data.drop(isolated_row.index)

    print(f"Ligne isolée : \n{isolated_row}")

    # Sauvegarder la ligne isolée et le dataset restant
    print("Sauvegarde de la ligne isolée et du dataset modifié...")
    isolated_row.to_csv(isolated_row_file, index=False)
    data.to_csv(output_data_file, index=False)

    print(f"Dataset sans la ligne isolée sauvegardé sous : {output_data_file}")
    print(f"Ligne isolée sauvegardée sous : {isolated_row_file}")


