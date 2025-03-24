# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 09:53:16 2025

@author: Christian
"""

from itertools import combinations
import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import svm
from scripts.svm.data_loading import load_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
from imblearn.under_sampling import NearMiss, ClusterCentroids

from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import (undersampling_random, undersampling_random_timelimited,
                                               undersampling_nearmiss, undersampling_cnn,
                                               undersampling_enn, undersampling_clustercentroids,
                                               undersampling_tomeklinks)
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                                    plot_roc_curve, permutation_ranking, evaluate_svm_with_feature_selection)
from scripts.svm.utils import (save_outputfile, get_adjacent_values, PermutationImportanceWrapper,
                               remove_correlated_features, remove_low_variance, select_k_best)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# --- PATHS ---

# Filepath and plot folder paths
common_path = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

# filepath = common_path / 'mod1_newfeatures_NEW.csv'
filepath = common_path / 'mod1_newfeatures_NEW.csv'
results_path = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

# --- DATA IMPORT ---

# Load and clean data
mod1 = load_data(filepath)
print(mod1.dtypes)  # For initial data type inspection

# List of candidate features
candidate_features = [
    'TaG_delta_5d',
    'TminG_delta_3d',
    'HS_delta_5d',
    'WetSnow_Temperature',
    'New_MF_Crust',
    'Precip_3d',
    'Precip_2d',
    'TempGrad_HS',
    'Tsnow_delta_3d',
    'TmaxG_delta_3d',
    'HSnum',
    'TempAmplitude_2d',
    'WetSnow_CS',
    'TaG',
    'Tsnow_delta_2d',
    'DayOfSeason',
    'Precip_5d',
    'TH10_tanh',
    'TempAmplitude_1d',
    'TaG_delta_2d',
    'HS_delta_1d',
    'HS_delta_3d',
    'TaG_delta_3d'
]

feature_plus = candidate_features + ['AvalDay']
mod1_clean = mod1[feature_plus]
mod1_clean = mod1_clean.dropna()

X = mod1_clean[candidate_features]
y = mod1_clean['AvalDay']


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Define maximum feature combination size
max_combination_size = 2  # Adjust based on computational resources

# Store results
results = []

# Function to optimize C for a given feature combination


def objective(trial, X_train, y_train):
    C = trial.suggest_loguniform("C", 0.0001, 1000)
    gamma = trial.suggest_loguniform("gamma", 0.00001, 100)
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    score = cross_val_score(model, X_train, y_train,
                            cv=5, scoring="f1_macro").mean()
    return score


# Iterate over feature combinations
for r in range(2, max_combination_size + 1):
    for feature_comb in combinations(candidate_features, r):
        feature_list = list(feature_comb)
        print(f"Evaluating features: {feature_list}")

        # Prepare feature subset
        X_subset = X_train[feature_list]  # Assuming X_train is your dataset
        y_subset = y_train  # Target variable

        # Use Optuna to find best C for this feature set
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(
            trial, X_subset, y_subset), n_trials=30)

        # Store the results
        results.append({
            'features': feature_list,
            'best_C': study.best_params["C"],
            'best_score': study.best_value
        })

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Sort by lowest C
df_sorted = df_results.sort_values(by='best_C', ascending=True)

# Display top results
print(df_sorted.head())

# Save results
df_sorted.to_csv("feature_combinations_C_analysis.csv", index=False)


####
SHAP = ['TaG_delta_5d',
        'TminG_delta_3d',
        'HS_delta_5d',
        'WetSnow_Temperature',
        'New_MF_Crust',
        'Precip_3d',
        'Precip_2d',
        'TempGrad_HS',
        'Tsnow_delta_3d',
        'TmaxG_delta_3d',
        'HSnum',
        'TempAmplitude_2d',
        'WetSnow_CS',
        'TaG',
        'Tsnow_delta_2d',
        'DayOfSeason',
        'Precip_5d',
        'TH10_tanh',
        'TempAmplitude_1d',
        'TaG_delta_2d',
        'HS_delta_1d',
        'HS_delta_3d',
        'TaG_delta_3d']

# best_features = list(set(BestFeatures_FW_20 + BestFeatures_BW_27))

# Data preparation
feature_plus = SHAP + ['AvalDay']
mod1_clean = mod1[feature_plus].dropna()
X = mod1_clean[SHAP]
y = mod1_clean['AvalDay']

X_resampled, y_resampled = undersampling_clustercentroids(X, y)

# param_grid = {
#     'C': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
#           0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
#           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#           1, 2, 3, 4, 5, 6, 7, 8, 9,
#           10, 20, 30, 40, 50, 60, 70, 80, 90,
#           100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
#     'gamma': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
#               0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
#               0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
#               0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
#               1, 2, 3, 4, 5, 6, 7, 8, 9,
#               10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# }

param_grid = {
    'C': [0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
          0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
          750, 1000],
    'gamma': [100, 75, 50, 30, 20, 15, 10, 7.5, 5, 3, 2, 1.5, 1,
              0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008,
              0.007, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0007, 0.0005, 0.0003, 0.0002,
              0.00015, 0.0001]
}

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42)

scaler = StandardScaler()
# scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Applicazione di LDA per Feature Extraction**
# Ridurre a 1 dimensione, ad esempio
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(
    X_test_scaled, columns=X_test.columns, index=X_test.index)

res_tuning = tune_train_evaluate_svm(
    X_train, y_train, X_test, y_test, param_grid,
    resampling_method=f'ClusterCentroids')

classifier = train_evaluate_final_svm(
    X_train, y_train, X_test, y_test, res_tuning['best_params'])
