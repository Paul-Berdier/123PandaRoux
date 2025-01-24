import optuna
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

data = pd.read_csv('data/catastrophes_naturelles_transformes.csv')


data = pd.get_dummies(data, columns=['quartier'], drop_first=True)


X = data.drop(['catastrophe','date'], axis=1)
y = data['catastrophe']

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


# study = optuna.create_study(direction='maximize')
#
#
# study.optimize(objective, n_trials=100)
#
#
# print("Best trial: score {},\nparams {}".format(study.best_trial.value, study.best_trial.params))
#
#
# best_params = study.best_trial.params
#
#
# best_xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
#
#
# best_xgb_model.fit(X_train, y_train)
#
#
# y_pred = best_xgb_model.predict(X_test)
model = xgb.XGBClassifier(n_estimators = 179, learning_rate= 0.19938380477723183, max_depth= 10, subsample= 0.6974169604519268, colsample_bytree= 0.6169828804734236, gamma= 0.050321171355875105, min_child_weight= 1, reg_alpha= 0.4500293330158086, reg_lambda= 0.638264258149184,random_state= 42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
# print("Predicted vs Actual:")
# for pred, actual in zip(y_pred, y_test):
#     print(f"Predicted: {pred}, Actual: {actual}")
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
disp.plot()
plt.title('Confusion Matrix')
plt.show()