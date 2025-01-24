import optuna
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scripts.visualisation_script import plot_learning_curve
import xgboost as xgb
import joblib


def train_ml_model(prepared_data_file, output_roc_curve, output_learning_curve, output_matrice_conf,
                   target_column='catastrophe', output_model_file='best_model_ML.joblib'):
    """
    Train a Machine Learning model with XGBoost and evaluate its performance.

    Parameters:
        prepared_data_file (str): Path to the prepared dataset.
        output_roc_curve (str): Path to save the ROC curve plot.
        output_learning_curve (str): Path to save the learning curve plot.
        output_matrice_conf (str): Path to save the confusion matrix plot.
        target_column (str): Name of the target column.
        output_model_file (str): Path to save the trained model.

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(prepared_data_file)

    # Séparation des caractéristiques et de la cible
    X = data.drop(columns=[target_column, 'date'])
    y = data[target_column]

    # Division des données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fonction d'optimisation avec Optuna pour XGBoost
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    # Optimisation avec Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Meilleur essai: score {},\nparamètres {}".format(study.best_trial.value, study.best_trial.params))

    best_params = study.best_trial.params
    best_xgb_model = xgb.XGBClassifier(**best_params, random_state=42)

    # Entraînement du modèle avec les meilleurs paramètres
    print("Entraînement du modèle XGBoost avec les meilleurs paramètres...")
    best_xgb_model.fit(X_train, y_train)

    # Prédictions et évaluation
    print("Évaluation des performances...")
    y_pred = best_xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(cm)

    disp = ConfusionMatrixDisplay(cm, display_labels=best_xgb_model.classes_)
    disp.plot()
    plt.title('Matrice de confusion')
    plt.savefig(output_matrice_conf)
    plt.show()

    # Sauvegarde du modèle
    joblib.dump(best_xgb_model, output_model_file)
    print(f"Modèle sauvegardé sous : {output_model_file}")

    # Calcul des courbes ROC pour chaque classe
    print("Calcul des courbes ROC pour chaque classe...")
    y_test_bin = label_binarize(y_test, classes=best_xgb_model.classes_)
    y_pred_proba = best_xgb_model.predict_proba(X_test)

    plt.figure()
    for i, class_label in enumerate(best_xgb_model.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f"Classe {class_label} (AUC={auc_score:.2f})")

    # Moyenne micro
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    auc_micro = roc_auc_score(y_test_bin, y_pred_proba, average='micro')
    plt.plot(fpr_micro, tpr_micro, linestyle='--', label=f"Micro-average (AUC={auc_micro:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.grid()
    plt.savefig(output_roc_curve)
    plt.close()

    # Courbe d'apprentissage
    print("Génération des courbes d'apprentissage...")
    plot_learning_curve(best_xgb_model, X_train, y_train, "Courbe d'apprentissage", output_learning_curve)
