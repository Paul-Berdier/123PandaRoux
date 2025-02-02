# Hackathon "La Garonne déborde !"

## Contexte du projet

En 2180, la mégapole qu'est devenue Toulouse est sujette à de violentes catastrophes naturelles qui détruisent la ville. Heureusement, un groupe d'experts en Data Science a formé une équipe pour protéger la ville des inondations, tremblements de terre, hackers et autres cataclysmes. Chaque membre de l'équipe a mis en place une solution IT pour couvrir toutes ces menaces.

## Équipe

Notre équipe est composée de 7 étudiants en Master Data Science sous le nom de **Equipe Panda Roux**. Chaque membre a contribué à une partie spécifique du projet, avec un focus particulier sur la Data Science et la mise en place de modèles de prédiction.

## Objectifs du projet

- **Nettoyer et normaliser les données fournies.**
- **Créer un modèle pour prédire les catastrophes climatiques par jour et par zone.**
- **Évaluer les performances des modèles.**
- **Intégrer les données dans une visualisation graphique.**

## Résumé des actions

### Étapes réalisées :
1. **Réduction et nettoyage des données** : Nettoyage des données de catastrophes naturelles, normalisation des valeurs et élimination des colonnes inutiles.
2. **Création de modèles prédictifs** : Un modèle basé sur des données IoT et un modèle standard avec le dataset de base.
3. **Prédiction avec les modèles** : Utilisation des modèles pour effectuer des prédictions sur des données isolées.
4. **Visualisation des résultats** : Utilisation de graphiques pour visualiser les performances des modèles (courbes d'apprentissage, matrices de corrélation, etc.).

### Modèles créés :
1. **Modèle IoT** : Prédiction des catastrophes naturelles en utilisant des données issues de capteurs IoT.
2. **Modèle Standard** : Utilisation d'un dataset de base sans capteurs IoT.

## Visualisations Générées

1. **Matrice de corrélation avant et après nettoyage des données**  
   ![Matrice de corrélation avant nettoyage](docs/visu_corr_before.png)  
   ![Matrice de corrélation après nettoyage](docs/visu_corr_after.png)

2. **Courbes d'apprentissage des modèles**  
   - Courbe d'apprentissage du modèle standard  
     ![Courbe d'apprentissage standard](docs/learning_curve.png)
   - Courbe d'apprentissage du modèle IoT  
     ![Courbe d'apprentissage IoT](docs/learning_curve_iot.png)

3. **Matrice de confusion des modèles**  
   - Modèle standard  
     ![Matrice de confusion standard](docs/output_matrice_conf.png)
   - Modèle IoT  
     ![Matrice de confusion IoT](docs/output_matrice_conf_iot.png)

4. **Courbes ROC des modèles**  
   - Modèle standard  
     ![Courbe ROC standard](docs/output_roc_curve.png)
   - Modèle IoT  
     ![Courbe ROC IoT](docs/output_roc_curve_iot.png)

5. **Prédiction du jour**  
   ![Carte des prédictions](docs/prediction_map.png)

6. **Historique des catastrophes**  
   ![Historique des catastrophes](docs/historique_catastrophes.png)

## Colonnes utilisées pour le modèle IoT

Pour le modèle IoT, les données suivantes ont été sélectionnées et conservées :

- **date**
- **quartier**
- **humidite**
- **sismicite**
- **catastrophe**

Ces colonnes ont été choisies pour leur pertinence dans la prédiction des catastrophes naturelles, permettant d'intégrer à la fois des facteurs temporels, géographiques et environnementaux.

---

## Technologies et outils utilisés

- **Python** : Langage principal pour le traitement des données, la création des modèles et la visualisation.
- **Pandas** : Gestion et manipulation des données tabulaires.
- **XGBoost** : Algorithme de machine learning utilisé pour la création des modèles prédictifs.
- **Joblib** : Sauvegarde et chargement des modèles ML.
- **Matplotlib** : Création de visualisations graphiques.
- **Power BI** : Création de tableaux de bord interactifs pour visualiser les prédictions et l’historique des catastrophes.
- **Git** : Collaboration et gestion des versions du projet.

## Organisation des fichiers

Le projet est structuré en plusieurs dossiers pour faciliter l'organisation et la gestion des différentes étapes :

1. **Création des dossiers nécessaires** :
   - Créez les dossiers suivants à la racine du projet :
     - `data/` : Ce dossier contiendra les fichiers de données en entrée et sortie, comme les datasets nettoyés, les lignes isolées, et les statistiques.
     - `models/` : Ce dossier contiendra les modèles de machine learning entraînés.
     - `docs/` : Ce dossier contiendra les visualisations générées, telles que les courbes ROC, les matrices de confusion et les courbes d'apprentissage.

2. **Placement des datasets** :
   - Placez le dataset `catastrophes_naturelles.csv` dans le dossier `data/` pour le traitement des données.
   - Le projet générera les autres fichiers et visualisations dans leurs dossiers respectifs.

## Fichiers principaux

- **main.py** : Programme principal pour l’exécution des étapes du projet.
- **script_hackathon.ipynb** : Jupyter Notebook contenant les étapes complètes du projet pour les utilisateurs préférant cette interface.
- **scripts/** : Contient les différents modules Python pour le nettoyage des données, l’entraînement des modèles et les prédictions.

## Instructions pour exécuter le projet

### Option 1 : Utilisation du code Python

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Organisation des dossiers** :
   - Placez vos données dans le dossier `data`.
   - Créez les dossiers `docs` et `models` si ce n’est pas déjà fait.

3. **Exécution du script principal** :
   ```bash
   python main.py
   ```

4. **Instructions dans le terminal** :
   Suivez les instructions dans le terminal pour choisir les étapes à exécuter.

### Option 2 : Utilisation du Notebook Jupyter

1. **Lancez le fichier** `script_hackathon.ipynb` dans Jupyter Notebook.
2. **Exécutez chaque cellule** dans l’ordre pour reproduire les étapes du projet.

### Option 3 : Visualisation avec Power BI

1. **Ouvrez le fichier Power BI** présentant les graphiques interactifs.
2. **Connectez-le aux données** du dossier `data` pour visualiser les éléments interactifs (historique et prédictions).

