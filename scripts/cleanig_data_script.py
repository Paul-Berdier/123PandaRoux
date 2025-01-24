from operator import index
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def debug_ligne(input_file, output_file):
    """
    Fonction pour nettoyer les lignes d'un fichier CSV :
    - Supprime les guillemets (simples et doubles).
    - Remplace les virgules dans les listes par des points-virgules.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
    """
    print("Lecture du fichier d'entrée...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    print("Traitement des lignes pour supprimer les guillemets et ajuster les séparateurs...")
    processed_data = []
    for line in raw_data:
        if '[' in line and ']' in line:
            line = line.replace(', ', '; ')
        line = line.replace('"', '').replace("'", '')
        processed_data.append(line)

    print("Écriture des données nettoyées dans le fichier de sortie...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.writelines(processed_data)

    print(f"Nettoyage terminé. Les données ont été sauvegardées dans : {output_file}")


def column_mapping(input_file, output_file, mapping, column_name):
    """
    Applique un mapping à une colonne spécifique d'un fichier CSV.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
        mapping (dict): Dictionnaire de correspondance pour la colonne.
        column_name (str): Nom de la colonne à mapper.
    """
    print(f"Lecture des données depuis {input_file}...")
    input_data = pd.read_csv(input_file)

    print(f"Application du mapping sur la colonne '{column_name}'...")
    input_data[column_name] = input_data[column_name].map(mapping)

    print(f"Sauvegarde des données avec mapping dans {output_file}...")
    input_data.to_csv(output_file, index=False)


def drop_column(input_file, output_file, important_features):
    """
    Réduit un fichier CSV en conservant uniquement des colonnes spécifiques.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
        important_features (list): Liste des colonnes à conserver.
    """
    print(f"Lecture des données depuis {input_file}...")
    input_data = pd.read_csv(input_file)

    print("Filtrage des colonnes importantes...")
    reduced_data = input_data[important_features]

    print(f"Sauvegarde des données réduites dans {output_file}...")
    reduced_data.to_csv(output_file, index=False)

def normalize_humidity(input_file, output_file):
    """
    Normalise la colonne 'humidite' en divisant les valeurs par 100 pour les ramener à une échelle de 0 à 1.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
    """
    print(f"Chargement des données depuis {input_file}...")
    data = pd.read_csv(input_file)

    if 'humidite' in data.columns:
        print("Normalisation de la colonne 'humidite' (division par 100)...")
        data['humidite'] = data['humidite'] / 100
    else:
        print("La colonne 'humidite' est introuvable dans les données.")
        return

    print(f"Sauvegarde des données normalisées dans {output_file}...")
    data.to_csv(output_file, index=False)
    print("Normalisation terminée avec succès.")

def normalize_data(input_file, output_file, Class='catastrophe'):
    """
    Normalise les données numériques d'un fichier CSV (exclut les colonnes non numériques).

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
        Class (str): Nom de la colonne cible à exclure de la normalisation.
    """
    print(f"Chargement des données depuis {input_file}...")
    data = pd.read_csv(input_file)

    print(f"Identification des colonnes numériques à normaliser (excluant '{Class}' et 'date')...")
    # Séparer les colonnes numériques à normaliser
    features = data.drop(columns=[Class, 'date'])  # Exclure la colonne cible et la colonne 'date'
    target = data[Class]

    # Appliquer la normalisation uniquement sur les colonnes numériques
    print("Application de la normalisation sur les colonnes numériques...")
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    print("Réintégration de la colonne cible et sauvegarde des données normalisées...")
    scaled_data = pd.concat([scaled_features, target, data['date']], axis=1)  # Réintégrer 'date' et 'Class'
    scaled_data.to_csv(output_file, index=False)

    print(f"Données normalisées sauvegardées dans {output_file}.")

def compute_statistics(input_file, output_file):
    """
    Calcule les statistiques descriptives essentielles d'un fichier CSV en excluant les colonnes non numériques.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
    """
    print(f"Chargement des données depuis {input_file}...")
    data = pd.read_csv(input_file)

    print("Exclusion des colonnes non numériques...")
    numeric_data = data.select_dtypes(include=[np.number])  # Sélectionner uniquement les colonnes numériques

    print("Calcul des statistiques descriptives pour les colonnes numériques...")
    stats = numeric_data.describe().T
    stats['median'] = numeric_data.median()
    stats['IQR'] = stats['75%'] - stats['25%']

    print(f"Sauvegarde des statistiques dans {output_file}...")
    stats.to_csv(output_file)

    print("Statistiques descriptives calculées et sauvegardées avec succès.")

def correlation_matrix(input_file, output_file, Class='catastrophe', threshold=0.05):
    """
    Calcule la matrice de corrélation et identifie les colonnes pertinentes.

    Parameters:
        input_file (str): Chemin du fichier d'entrée.
        output_file (str): Chemin du fichier de sortie.
        Class (str): Colonne cible pour la corrélation.
        threshold (float): Seuil pour sélectionner les colonnes importantes.
    """
    print(f"Chargement des données depuis {input_file}...")
    data = pd.read_csv(input_file)

    # Sauvegarder la colonne 'date' pour la réintégrer plus tard
    if 'date' in data.columns:
        date_column = data[['date']]
    else:
        date_column = None

    # Filtrer uniquement les colonnes numériques
    data_numeric = data.select_dtypes(include=[np.number])

    # Vérification des colonnes utilisées
    print(f"Colonnes numériques utilisées pour la corrélation : {data_numeric.columns}")

    print(f"Calcul de la matrice de corrélation avec '{Class}'...")
    correlation_matrix = data_numeric.corr()
    correlations_with_class = correlation_matrix[Class].sort_values(ascending=False)

    print("Sélection des colonnes ayant une corrélation significative...")
    important_features = correlations_with_class[abs(correlations_with_class) > threshold].index.tolist()

    # Conserver uniquement les colonnes importantes
    reduced_data = data[important_features]

    # Réintégrer la colonne 'date' si elle existe
    if date_column is not None:
        reduced_data = pd.concat([date_column, reduced_data], axis=1)

    print(f"Sauvegarde des données réduites dans {output_file}...")
    reduced_data.to_csv(output_file, index=False)

    print(f"Colonnes importantes identifiées (avec 'date') : {reduced_data.columns.tolist()}")


def isolate_random_row(data_file, output_data_file, isolated_row_file, target_column='Class'):
    """
    Isoler une ligne aléatoire pour prédiction et sauvegarder le reste.

    Parameters:
        data_file (str): Chemin du fichier contenant les données originales.
        output_data_file (str): Chemin pour sauvegarder le dataset sans la ligne isolée.
        isolated_row_file (str): Chemin pour sauvegarder la ligne isolée.
        target_column (str): Nom de la colonne cible.
    """
    print(f"Chargement des données depuis {data_file}...")
    data = pd.read_csv(data_file)

    if data.empty:
        raise ValueError("Le dataset est vide. Veuillez vérifier le fichier source.")

    print("Isolation d'une ligne aléatoire...")
    isolated_row = data.sample(n=1, random_state=42)

    print("Suppression de la ligne isolée du dataset...")
    data = data.drop(isolated_row.index)

    print(f"Ligne isolée : \n{isolated_row}")

    print("Sauvegarde des résultats...")
    isolated_row.to_csv(isolated_row_file, index=False)
    data.to_csv(output_data_file, index=False)

    print(f"Dataset sans la ligne isolée sauvegardé sous : {output_data_file}")
    print(f"Ligne isolée sauvegardée sous : {isolated_row_file}")
