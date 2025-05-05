# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:55:18 2025

@author: Christian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
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


if __name__ == '__main__':
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures_NEW.csv'
    # filepath = common_path / 'mod1_certified.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # -------------------------------------------------------
    # STABILITY VARYING C AND GAMMA
    # -------------------------------------------------------
    SHAP_16 = ['TaG_delta_5d',
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
               'DayOfSeason']
    res_shap16 = evaluate_svm_with_feature_selection(mod1, SHAP_16)

    best_C = res_shap16[2]['best_params']['C']
    best_gamma = res_shap16[2]['best_params']['gamma']

    # Data preparation
    feature_plus = SHAP_16 + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[SHAP_16]
    y = mod1_clean['AvalDay']

    X_resampled, y_resampled = undersampling_clustercentroids(X, y)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    common_indices = X_train.index.intersection(X_test.index)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index)

    # # Generate range of C and gamma values around the best parameters
    # C_values = np.linspace(500, 1000, num=501)
    C_values = np.arange(100, 1501, 5)  # Includes 1000
    gamma_values = np.linspace(0.004, 0.012, num=41)

    performance_results = []

    for C in C_values:
        for gamma in gamma_values:
            params = {'C': C, 'gamma': gamma}
            # params = {'C': C, 'gamma': 'auto'}
            print(f'--- Testing C={C}, gamma={gamma} ---')

            classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
                X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False
            )

            performance_results.append({
                'C': C,
                'gamma': gamma,
                'accuracy': evaluation_metrics_SVM['accuracy'],
                'precision': evaluation_metrics_SVM['precision'],
                'recall': evaluation_metrics_SVM['recall'],
                'f1': evaluation_metrics_SVM['f1'],
                'MCC': evaluation_metrics_SVM['MCC']
            })

    df_performance = pd.DataFrame(performance_results)

    # Group by C and compute statistics for MCC
    df_grouped_mcc = df_performance.groupby("C")["MCC"].agg([
        ("mean", "mean"),
        ("10th", lambda x: np.percentile(x, 10)),
        ("25th", lambda x: np.percentile(x, 25)),
        ("50th", lambda x: np.percentile(x, 50)),  # Median
        ("75th", lambda x: np.percentile(x, 75)),
        ("90th", lambda x: np.percentile(x, 90)),
        ("min", "min"),
        ("max", "max")
    ]).reset_index()

    # Group by C and compute statistics for F1
    df_grouped_f1 = df_performance.groupby("C")["f1"].agg([
        ("mean", "mean"),
        ("10th", lambda x: np.percentile(x, 10)),
        ("25th", lambda x: np.percentile(x, 25)),
        ("50th", lambda x: np.percentile(x, 50)),  # Median
        ("75th", lambda x: np.percentile(x, 75)),
        ("90th", lambda x: np.percentile(x, 90)),
        ("min", "min"),
        ("max", "max")
    ]).reset_index()

    # Create the plot
    plt.figure(figsize=(12, 8))

    # --- MCC Plot ---
    plt.plot(df_grouped_mcc["C"], df_grouped_mcc["50th"], color='#1565C0',
             linestyle='-', linewidth=2, label="MCC Median (50%)")
    plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["min"], df_grouped_mcc["max"],
                     color='#B3E5FC', alpha=0.5, label="MCC min-max")
    plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["10th"], df_grouped_mcc["90th"],
                     color='#81D4FA', alpha=0.75, label="MCC 10th-90th Percentile")
    plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["25th"], df_grouped_mcc["75th"],
                     color='#4FC3F7', alpha=1, label="MCC 25th-75th Percentile")

    # --- F1 Plot ---
    plt.plot(df_grouped_f1["C"], df_grouped_f1["50th"], color='#C62828',
             linestyle='-', linewidth=2, label="F1-score Median (50%)")
    plt.fill_between(df_grouped_f1["C"], df_grouped_f1["min"], df_grouped_f1["max"],
                     color='#FFCDD2', alpha=0.5, label="F1 min-max")
    plt.fill_between(df_grouped_f1["C"], df_grouped_f1["10th"], df_grouped_f1["90th"],
                     color='#EF9A9A', alpha=0.75, label="F1 10th-90th Percentile")
    plt.fill_between(df_grouped_f1["C"], df_grouped_f1["25th"], df_grouped_f1["75th"],
                     color='#E57373', alpha=1, label="F1 25th-75th Percentile")

    # Plot labels and settings
    plt.xlabel("C value")
    plt.ylabel("Score")
    plt.title(
        'Performance Metrics (MCC & F1) Across C Values — Percentiles over Gamma ∈ [0.006, 0.01]', fontsize=18)
    plt.legend(fontsize=10)
    # plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle='dotted')
    plt.tight_layout()
    plt.show()

    # ------ interpola spline

    from scipy.interpolate import UnivariateSpline

    df = pd.DataFrame({
        'C': df_grouped_f1['C'],
        'F1_median': df_grouped_f1['50th'],
        'MCC_median': df_grouped_mcc['50th']
    })

    # Smussa su scala log(C)
    x = np.log(df['C'].values)
    y_f1 = df['F1_median'].values
    y_mcc = df['MCC_median'].values

    # Interpolazione spline
    spline_f1 = UnivariateSpline(x, y_f1, s=0.05)
    spline_mcc = UnivariateSpline(x, y_mcc, s=0.05)

    # Punti interpolati
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_f1_dense = spline_f1(x_dense)
    y_mcc_dense = spline_mcc(x_dense)

    # Derivata numerica smussata
    dy_f1 = np.gradient(y_f1_dense, x_dense)
    dy_mcc = np.gradient(y_mcc_dense, x_dense)

    # Parametri di soglia e stabilità
    eps = 0.02  # soglia di variazione minima
    n = 1        # punti consecutivi per stabilità

    def detect_plateau(x_vals, deriv_vals, eps, n):
        abs_der = np.abs(deriv_vals)
        for i in range(len(abs_der) - n):
            if np.all(abs_der[i:i+n] < eps):
                return np.exp(x_vals[i])  # ritorna in scala C
        return np.exp(x_vals[-1])

    def detect_plateau_relative(x_vals, y_vals, pct=0.95):
        y_max = np.max(y_vals)
        for i, y in enumerate(y_vals):
            if y >= pct * y_max:
                return np.exp(x_vals[i])
        return np.exp(x_vals[-1])

    C_f1_plateau = detect_plateau(x_dense, dy_f1, eps, n)
    C_mcc_plateau = detect_plateau(x_dense, dy_mcc, eps, n)

    C_f1_plateau_rel = detect_plateau_relative(x_dense, y_f1_dense)
    C_mcc_plateau_rel = detect_plateau_relative(x_dense, y_mcc_dense)

    y_avg = (y_f1_dense + y_mcc_dense) / 2
    dy_avg = np.gradient(y_avg, x_dense)

    C_avg_plateau = detect_plateau(x_dense, dy_avg, eps, n)
    C_avg_plateau_rel = detect_plateau_relative(x_dense, y_avg)

    # Calcolo dei punti originali medi
    y_avg_original = (y_f1 + y_mcc) / 2

    # Plot
    plt.figure(figsize=(12, 6))

    # F1
    plt.plot(df['C'], y_f1, 'o', alpha=0.3, color='red', label='F1 (original)')
    plt.plot(np.exp(x_dense), y_f1_dense, '-', color='red', label='F1 spline')
    plt.axvline(C_f1_plateau_rel, linestyle='--', color='red',
                label=f'F1 plateau @ C ≈ {int(C_f1_plateau_rel)}')

    # MCC
    plt.plot(df['C'], y_mcc, 'o', alpha=0.3,
             color='blue', label='MCC (original)')
    plt.plot(np.exp(x_dense), y_mcc_dense, '-',
             color='blue', label='MCC spline')
    plt.axvline(C_mcc_plateau_rel, linestyle='--', color='blue',
                label=f'MCC plateau @ C ≈ {int(C_mcc_plateau_rel)}')

    # Media (punti originali)
    plt.plot(df['C'], y_avg_original, 's', alpha=0.5,
             color='green', label='Avg (F1+MCC original)')
    # Media spline
    plt.plot(np.exp(x_dense), y_avg, '-', color='green', label='Avg spline')
    plt.axvline(C_avg_plateau_rel, linestyle='--', color='green',
                label=f'Avg plateau @ C ≈ {int(C_avg_plateau_rel)}')

    # Finalizzazione
    plt.xscale('linear')
    plt.xlabel("C value")
    plt.ylabel("Score")
    plt.title("Plateau Detection via Smoothed Numerical Derivative")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Output dei valori
    print(f"📌 F1 plateau a C ≈ {C_f1_plateau_rel:.2f}")
    print(f"📌 MCC plateau a C ≈ {C_mcc_plateau_rel:.2f}")
    print(f"📌 Plateau medio (F1+MCC) @ C ≈ {C_avg_plateau_rel:.2f}")

    C_optimal = C_avg_plateau_rel
    # Ricerca su gamma
    # più ampio rispetto a [0.006, 0.01]
    gamma_values_expanded = np.linspace(0.01, 0.1, num=201)
    performance_gamma = []

    for gamma in gamma_values_expanded:
        print(f'--- Testing gamma={gamma:.5f}, fixed C={C_optimal:.2f} ---')

        params = {'C': C_optimal, 'gamma': gamma}
        classifier_SVM, evaluation_metrics = train_evaluate_final_svm(
            X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False
        )

        performance_gamma.append({
            'gamma': gamma,
            'accuracy': evaluation_metrics['accuracy'],
            'precision': evaluation_metrics['precision'],
            'recall': evaluation_metrics['recall'],
            'f1': evaluation_metrics['f1'],
            'MCC': evaluation_metrics['MCC']
        })

    df_gamma = pd.DataFrame(performance_gamma)

    # --- MCC Plot ---
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(df_gamma["gamma"], df_gamma["MCC"], color='#1565C0',
             linestyle='-', linewidth=2, label=f'MCC @ C {C_optimal:.2f}')
    plt.plot(df_gamma["gamma"], df_gamma["f1"], color='#C62828',
             linestyle='-', linewidth=2, label=f'F1-score @ C {C_optimal:.2f}')

    # Plot labels and settings
    plt.xlabel("gamma value")
    plt.ylabel("Score")
    plt.title(
        'Performance Metrics (MCC & F1) Across gamma Values — with C optimal = {C_optimal:.2f}', fontsize=18)
    plt.legend(fontsize=10)
    # plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle='dotted')
    plt.tight_layout()
    plt.show()

    # Interpolazione per MCC
    x = np.log(df_gamma['gamma'].values)
    y = df_gamma['f1'].values

    spline = UnivariateSpline(x, y, s=0.05)
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = spline(x_dense)
    dy = np.gradient(y_dense, x_dense)

    # Rilevamento plateau
    gamma_plateau = detect_plateau(x_dense, dy, eps=0.02, n=5)
    gamma_plateau_rel = detect_plateau_relative(x_dense, y_dense, pct=0.99)

    print(f"📌 Gamma plateau a gamma ≈ {gamma_plateau_rel:.5f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df_gamma['gamma'], df_gamma['f1'],
             'o', alpha=0.3, label='F1 (original)')
    plt.plot(np.exp(x_dense), y_dense, '-', label='f1 spline')
    plt.axvline(gamma_plateau_rel, linestyle='--', color='red',
                label=f'Plateau @ gamma ≈ {gamma_plateau_rel:.4f}')
    plt.xlabel("Gamma value")
    plt.ylabel("F1 Score")
    plt.title("Gamma Plateau Detection")
    plt.grid(True)
    plt.legend()
    plt.show()

    gamma_optimal = gamma_plateau_rel

feature_list = ['TaG_delta_5d',
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
                'DayOfSeason']

# Add target variable to the feature list
feature_with_target = feature_list + ['AvalDay']

# Data preprocessing: filter relevant features and drop missing values
clean_data = mod1[feature_with_target].dropna()

# Extract features and target variable
X = clean_data[feature_list]
y = clean_data['AvalDay']

features_to_remove = remove_correlated_features(X, y)

X = X.drop(columns=features_to_remove)

# initial_param_grid = {
#     'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
#     'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
# }
# initial_param_grid = {
#     'C': [
#         0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003,
#         0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
#         0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
#         750, 1000, 1500, 2000, 3000, 5000, 7500, 10000
#     ],
#     'gamma': [
#         1000, 750, 500, 300, 200, 150, 100, 75, 50, 30, 20, 15, 10, 7.5, 5, 3, 2, 1.5, 1,
#         0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008,
#         0.007, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0007, 0.0005, 0.0003, 0.0002,
#         0.00015, 0.0001, 0.00008, 0.00007, 0.00005, 0.00003, 0.00002, 0.000015, 0.00001
#     ]
# }
initial_param_grid = {
    'C': [
        0.001, 0.0015, 0.002, 0.003,
        0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
        0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 50, 75, 100, 125, 150, 200,
        250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500
    ],
    'gamma': [
        100, 75, 50, 30, 20, 15, 10, 7.5, 5, 3, 2, 1.5, 1,
        0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008,
        0.007, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0007, 0.0005, 0.0003, 0.0002,
        0.00015, 0.0001, 0.000075
    ]
}

# X_resampled, y_resampled = undersampling_nearmiss(
#     X, y, version=3, n_neighbors=10)
X_resampled, y_resampled = undersampling_clustercentroids(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(
    X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(
    X_test), columns=X_test.columns, index=X_test.index)

# result = tune_train_evaluate_svm(
#     X_train, y_train, X_test, y_test, initial_param_grid,
#     resampling_method='Cluster Centroids')

# Step 6: Train the final model with the best hyperparameters and evaluate it
classifier, evaluation_metrics = train_evaluate_final_svm(
    X_train, y_train, X_test, y_test, {'C': C_optimal, 'gamma': gamma_optimal})

# # --- Estrazione best parameters
# best_C = res_shap16[2]['best_params']['C']
# best_gamma = res_shap16[2]['best_params']['gamma']

# # --- Finestra locale attorno ai best parameters
# # delta_C = 1000
# # delta_gamma = 0.002

# # C_values = np.arange(best_C - delta_C, best_C + delta_C + 1, step=100)
# # gamma_values = np.linspace(best_gamma - delta_gamma, best_gamma + delta_gamma, num=21)
# C_values = np.arange(100, 1501, 5)  # Includes 1000
# gamma_values = np.linspace(0.004, 0.012, num=41)

# # --- Preparazione dei dati
# feature_plus = SHAP_16 + ['AvalDay']
# mod1_clean = mod1[feature_plus].dropna()
# X = mod1_clean[SHAP_16]
# y = mod1_clean['AvalDay']

# X_resampled, y_resampled = undersampling_clustercentroids(X, y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.25, random_state=42
# )

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# X_train_scaled = pd.DataFrame(
#     X_train_scaled, columns=X_train.columns, index=X_train.index)
# X_test_scaled = pd.DataFrame(
#     X_test_scaled, columns=X_test.columns, index=X_test.index)

# # --- Valutazione delle prestazioni nella regione locale
# performance_results_local = []

# for C in C_values:
#     for gamma in gamma_values:
#         params = {'C': C, 'gamma': gamma}
#         print(f'Testing C={C}, gamma={gamma:.5f}')
#         model, metrics = train_evaluate_final_svm(
#             X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False
#         )

#         performance_results_local.append({
#             'C': C,
#             'gamma': gamma,
#             'accuracy': metrics['accuracy'],
#             'precision': metrics['precision'],
#             'recall': metrics['recall'],
#             'f1': metrics['f1'],
#             'MCC': metrics['MCC']
#         })

# df_local = pd.DataFrame(performance_results_local)

# # --- Analisi di stabilità: descrizione statistica
# print("\n🔎 Stabilità F1:")
# print(df_local["f1"].describe())
# print("\n🔎 Stabilità MCC:")
# print(df_local["MCC"].describe())

# # --- Heatmap F1-score
# pivot_f1 = df_local.pivot(index='gamma', columns='C', values='f1')
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_f1, cmap="YlGnBu", annot=False)
# plt.title("F1-score Heatmap attorno ai best parameters", fontsize=14)
# plt.xlabel("C")
# plt.ylabel("Gamma")
# plt.tight_layout()
# plt.show()

# # --- Heatmap MCC
# pivot_mcc = df_local.pivot(index='gamma', columns='C', values='MCC')
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_mcc, cmap="RdPu", annot=False)
# plt.title("MCC Heatmap attorno ai best parameters", fontsize=14)
# plt.xlabel("C")
# plt.ylabel("Gamma")
# plt.tight_layout()
# plt.show()

# # -------------------------------------------------------
# # STABILITY USING BOOTSTRAP
# # -------------------------------------------------------
# SHAP_16 = ['TaG_delta_5d',
#            'TminG_delta_3d',
#            'HS_delta_5d',
#            'WetSnow_Temperature',
#            'New_MF_Crust',
#            'Precip_3d',
#            'Precip_2d',
#            'TempGrad_HS',
#            'Tsnow_delta_3d',
#            'TmaxG_delta_3d',
#            'HSnum',
#            'TempAmplitude_2d',
#            'WetSnow_CS',
#            'TaG',
#            'Tsnow_delta_2d',
#            'DayOfSeason']
# res_shap16 = evaluate_svm_with_feature_selection(mod1, SHAP_16)

# # --- Estrazione best parameters

# from sklearn.utils import resample

# best_C = res_shap16[2]['best_params']['C']
# best_gamma = res_shap16[2]['best_params']['gamma']

# # Parametri migliori trovati precedentemente
# best_params = {'C': best_C, 'gamma': best_gamma}

# # Numero di bootstrap iterations
# n_iterations = 100
# bootstrap_results = []

# for i in range(n_iterations):
#     print(f'Bootstrap iteration {i+1}/{n_iterations}')

#     # Resampling con rimpiazzo
#     X_resampled_bs, y_resampled_bs = resample(
#         X_resampled, y_resampled, replace=True, random_state=i)

#     X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(
#         X_resampled_bs, y_resampled_bs, test_size=0.25, random_state=42
#     )

#     scaler = MinMaxScaler()
#     X_train_scaled_bs = scaler.fit_transform(X_train_bs)
#     X_test_scaled_bs = scaler.transform(X_test_bs)

#     X_train_scaled_bs = pd.DataFrame(
#         X_train_scaled_bs, columns=X.columns, index=X_train_bs.index)
#     X_test_scaled_bs = pd.DataFrame(
#         X_test_scaled_bs, columns=X.columns, index=X_test_bs.index)

#     model_bs, metrics_bs = train_evaluate_final_svm(
#         X_train_scaled_bs, y_train_bs, X_test_scaled_bs, y_test_bs, best_params, display_plot=False
#     )

#     bootstrap_results.append(metrics_bs)

# # --- Risultati in DataFrame
# df_bootstrap = pd.DataFrame(bootstrap_results)

# # --- Statistiche descrittive
# print("\n📊 Bootstrap Stability F1:")
# print(df_bootstrap["f1"].describe())
# print("\n📊 Bootstrap Stability MCC:")
# print(df_bootstrap["MCC"].describe())

# # --- Boxplot per visualizzare la variabilità
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.boxplot(y=df_bootstrap["f1"])
# plt.title("F1-score (Bootstrap)")

# plt.subplot(1, 2, 2)
# sns.boxplot(y=df_bootstrap["MCC"])
# plt.title("MCC (Bootstrap)")

# plt.tight_layout()
# plt.show()

# # Preparazione dei dati per il boxplot comparativo
# comparison_df = pd.DataFrame({
#     'F1-score': pd.concat([df_local["f1"], df_bootstrap["f1"]], ignore_index=True),
#     'Metodo': ['Griglia Parametri'] * len(df_local) + ['Bootstrap'] * len(df_bootstrap)
# })

# # Boxplot comparativo
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Metodo', y='F1-score', data=comparison_df,
#             palette=["#1f77b4", "#ff7f0e"])
# plt.title("Confronto Stabilità F1-score: Griglia vs Bootstrap", fontsize=14)
# plt.ylabel("F1-score")
# plt.xlabel("")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
# ------------------------------------------------------
