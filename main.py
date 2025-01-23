import os
from random import random

import pandas as pd

from scripts.ML_model_training_script import train_ml_model
from scripts.cleanig_data_script import debug_ligne, column_mapping, normalize_data, compute_statistics, \
    correlation_matrix, isolate_random_row, drop_column, normalize_humidity
from scripts.visualisation_script import visualisation_V1_V2, visualisation_correlation_matrix
from scripts.ML_model_training_script import train_ml_model
# from scripts.DL_model_training_script import train_deep_learning_model  # Nouvelle fonction pour le DL
# from scripts.prediction_script import perform_prediction  # Nouvelle fonction pour la prédiction
from colorama import Fore, Style

def display_message(message):
    """
    Affiche un message en vert avec un style particulier pour attirer l'attention.

    Parameters:
        message (str): Le message à afficher.
    """
    print(Fore.GREEN + message.upper() + Style.RESET_ALL)

# Définir les chemins des fichiers
catastrophes_naturelles_data = 'data/catastrophes_naturelles.csv'
clean_catastrophes_naturelles_data = 'data/clean_catastrophes_naturelles.csv'
statistics_data = 'data/statistics_data.csv'
random_row = 'data/random_row.csv'
reformed_catastrophes_naturelles_data = 'data/reformed_catastrophes_naturelles_data.csv'
output_roc_curve = 'docs/output_roc_curve.png'
output_learning_curve = 'docs/learning_curve.png'
ml_model_file = 'models/ml_model2.joblib'

# Définir le mapping des colonnes
mapping_cata = {
    "aucun": 0,
    "[seisme]": 1,
    "[innondation]": 2,
    "[innondation; seisme]": 3
}
mapping_zone = {
    "Zone 1": 1,
    "Zone 2": 3,
    "Zone 4": 2,
    "Zone 3": 4,
    "Zone 5": 5
}

# Définir les colonnes importantes à conserver
important_features = [
    'date',
    'quartier',
    'humidite',
    'sismicite',
    'catastrophe'
]

# Options pour le choix de l'utilisateur
def main():
    """
    Programme principal pour gérer le traitement des données et l'entraînement des modèles.
    """
    display_message("Bienvenue dans le programme de traitement des données et d'entraînement de modèle")
    options = {
        "1": "Réduire les données",
        '2': 'Séparer une ligne du dataset',
        '3': "Entrainement d'un modèle ML"
    }

    # Afficher les options disponibles
    for key, value in options.items():
        print(f"{key}. {value}")

    # Demander à l'utilisateur de choisir une option ou de tout exécuter
    choice = input(Fore.CYAN + "\nEntrez les numéros des étapes à exécuter (séparés par des virgules) ou appuyez sur Entrée pour tout exécuter : " + Style.RESET_ALL)

    if not choice.strip():
        choice = list(map(int, options.keys()))  # Exécuter toutes les étapes
    else:
        choice = list(map(int, choice.split(',')))

    # Étape 1 : Réduction des données
    if 1 in choice:
        display_message("\nÉtape 1 : Nettoyage et réduction des données...")
        debug_ligne(catastrophes_naturelles_data, clean_catastrophes_naturelles_data)
        column_mapping(clean_catastrophes_naturelles_data, clean_catastrophes_naturelles_data, mapping_cata, "catastrophe")
        column_mapping(clean_catastrophes_naturelles_data, clean_catastrophes_naturelles_data, mapping_zone, "quartier")
        drop_column(clean_catastrophes_naturelles_data, clean_catastrophes_naturelles_data, important_features)
        normalize_humidity(clean_catastrophes_naturelles_data, clean_catastrophes_naturelles_data)
        compute_statistics(clean_catastrophes_naturelles_data, statistics_data)
        display_message("Réduction et nettoyage des données terminés avec succès !")

    # Étape 2 : Separation d'une ligne
    if 2 in choice:
        display_message("\nÉtape 2 : Separation d'une ligne aléatoire")
        isolate_random_row(clean_catastrophes_naturelles_data, reformed_catastrophes_naturelles_data, random_row)
        display_message(f"Ligne isolée sauvegardée sous : {random_row}")

    # Étape 3 : Entrainement d'un modèle ML
    if 3 in choice:
        display_message("\nÉtape 3 : Entrainement d'un modèle ML")
        train_ml_model(reformed_catastrophes_naturelles_data, output_roc_curve, output_learning_curve, output_model_file=ml_model_file)
        display_message(f"Modèle DL sauvegardé sous : {ml_model_file}")

if __name__ == '__main__':
    main()
