import pandas as pd
import xgboost as xgb
import joblib
from io import StringIO  # Importer StringIO depuis io

# Fonction pour pré-traiter les données avant la prédiction
def preprocess_data(df):
    # Vérifier si df est un DataFrame
    if isinstance(df, pd.DataFrame):
        # Conversion de la colonne 'date' en un format numérique (nombre de jours depuis la date min)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = (df['date'] - df['date'].min()).dt.days  # Convertir en nombre de jours depuis la date min

        # Convertir toutes les colonnes en numériques, forcer les erreurs en NaN
        df = df.apply(pd.to_numeric, errors='coerce')  # Cela convertit toutes les colonnes en numérique, les autres en NaN

        # Supprimer les lignes avec des valeurs manquantes après la conversion
        df = df.dropna()

        return df
    else:
        raise ValueError("L'entrée doit être un DataFrame.")

# Fonction de prédiction avec le modèle ML
def perform_prediction(random_row, random_row_iot, ml_model_file, ml_model_file_iot):
    # Assurer que random_row est un DataFrame
    if isinstance(random_row, str):
        # Si random_row est une chaîne de caractères, la convertir en DataFrame
        try:
            random_row = pd.read_csv(StringIO(random_row))  # Utiliser StringIO pour lire la chaîne comme un CSV
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture de random_row : {e}")

    if isinstance(random_row_iot, str):
        # Si random_row_iot est une chaîne de caractères, la convertir en DataFrame
        try:
            random_row_iot = pd.read_csv(StringIO(random_row_iot))  # Utiliser StringIO pour lire la chaîne comme un CSV
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture de random_row_iot : {e}")

    # Assurer que random_row est un DataFrame à ce point
    if not isinstance(random_row, pd.DataFrame) or not isinstance(random_row_iot, pd.DataFrame):
        raise ValueError("random_row et random_row_iot doivent être des DataFrame.")

    # Charger le modèle de base avec joblib (si enregistré avec joblib)
    base_ml_model = joblib.load(ml_model_file)

    # Pré-traiter la ligne isolée (random_row)
    random_row = preprocess_data(random_row)

    # Pré-traiter la ligne isolée (random_row_iot)
    random_row_iot = preprocess_data(random_row_iot)

    # Prédiction avec le modèle de base
    base_ml_prediction = base_ml_model.predict(random_row)[0]
    print(f"Prédiction avec le modèle ML de base : {base_ml_prediction}")

    # Charger et pré-traiter les données IoT
    iot_model = joblib.load(ml_model_file_iot)

    # Prédiction avec le modèle IoT
    iot_ml_prediction = iot_model.predict(random_row_iot)[0]
    print(f"Prédiction avec le modèle IoT : {iot_ml_prediction}")

    return base_ml_prediction, iot_ml_prediction
