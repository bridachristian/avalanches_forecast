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
    # SHAP STABILITY VARYING C AND GAMMA
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
    C_values = np.arange(100, 1001, 1)  # Includes 1000
    gamma_values = np.linspace(0.006, 0.01, num=41)

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
    n = 5         # punti consecutivi per stabilità

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

    # Plot risultati
    plt.figure(figsize=(12, 6))
    plt.plot(df['C'], y_f1, 'o', alpha=0.3, label='F1 (original)')
    plt.plot(np.exp(x_dense), y_f1_dense, '-', label='F1 spline')
    plt.axvline(C_f1_plateau_rel, linestyle='--', color='red',
                label=f'F1 plateau @ C ≈ {int(C_f1_plateau_rel)}')

    plt.plot(df['C'], y_mcc, 'o', alpha=0.3, label='MCC (original)')
    plt.plot(np.exp(x_dense), y_mcc_dense, '-', label='MCC spline')
    plt.axvline(C_mcc_plateau_rel, linestyle='--', color='blue',
                label=f'MCC plateau @ C ≈ {int(C_mcc_plateau_rel)}')

    plt.xscale('linear')
    plt.xlabel("C value")
    plt.ylabel("Score")
    plt.title("Plateau Detection via Smoothed Numerical Derivative")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"📌 F1 plateau a C ≈ {C_f1_plateau_rel:.2f}")
    print(f"📌 MCC plateau a C ≈ {C_mcc_plateau_rel:.2f}")

    C_optimal = C_mcc_plateau_rel
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

    print(f"📌 Gamma plateau a gamma ≈ {gamma_plateau:.5f}")

    plt.figure(figsize=(10, 5))
    plt.plot(df_gamma['gamma'], df_gamma['f1'],
             'o', alpha=0.3, label='F1 (original)')
    plt.plot(np.exp(x_dense), y_dense, '-', label='f1 spline')
    plt.axvline(gamma_plateau, linestyle='--', color='red',
                label=f'Plateau @ gamma ≈ {gamma_plateau:.4f}')
    plt.xlabel("Gamma value")
    plt.ylabel("F1 Score")
    plt.title("Gamma Plateau Detection")
    plt.grid(True)
    plt.legend()
    plt.show()

    def gamma_search_and_plateau(C_value, X_train_scaled, y_train, X_test_scaled, y_test, label_prefix=''):
        gamma_values = np.linspace(0.01, 0.1, num=201)
        performance_gamma = []

        print(f"\n🔍 Ricerca gamma per C = {C_value:.2f} ({label_prefix})")

        for gamma in gamma_values:
            print(f'--- Testing gamma={gamma:.5f} ---')
            params = {'C': C_value, 'gamma': gamma}

            clf, metrics = train_evaluate_final_svm(
                X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False)

            performance_gamma.append({
                'gamma': gamma,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'MCC': metrics['MCC']
            })

        df_gamma = pd.DataFrame(performance_gamma)

        # Interpolazione + derivata per F1
        x = np.log(df_gamma['gamma'].values)
        y = df_gamma['f1'].values

        spline = UnivariateSpline(x, y, s=0.05)
        x_dense = np.linspace(x.min(), x.max(), 1000)
        y_dense = spline(x_dense)
        dy = np.gradient(y_dense, x_dense)

        gamma_plateau = detect_plateau(x_dense, dy, eps=0.02, n=5)

        print(f"📌 Gamma plateau ({label_prefix}) ≈ {gamma_plateau:.5f}")

        return df_gamma, gamma_plateau

    def calculate_plateau_curve(x_vals, y_vals, metric_name, label_prefix):
        # Spline su scala logaritmica
        x_log = np.log(x_vals)
        spline = UnivariateSpline(x_log, y_vals, s=0.05)
        x_dense = np.linspace(x_log.min(), x_log.max(), 1000)
        y_dense = spline(x_dense)
        dy = np.gradient(y_dense, x_dense)

        gamma_plateau = detect_plateau(x_dense, dy, eps=0.02, n=5)
        print(
            f"📌 Plateau {metric_name} ({label_prefix}) ≈ gamma {np.exp(gamma_plateau):.5f}")
        # restituiamo anche le curve per eventuali plottaggi
        return np.exp(gamma_plateau), x_dense, y_dense

    # --- Esegui ricerca gamma con C ottimale da MCC ---
    df_gamma_mcc, gamma_plateau_mcc = gamma_search_and_plateau(
        C_mcc_plateau_rel, X_train_scaled, y_train, X_test_scaled, y_test, label_prefix='MCC')

    # --- Esegui ricerca gamma con C ottimale da F1 ---
    df_gamma_f1, gamma_plateau_f1 = gamma_search_and_plateau(
        C_f1_plateau_rel, X_train_scaled, y_train, X_test_scaled, y_test, label_prefix='F1')

    # --- Calcolo dei plateau ---
    gamma_f1_plateau_mcc, x_f1_mcc, y_f1_mcc = calculate_plateau_curve(
        df_gamma_mcc['gamma'].values, df_gamma_mcc['f1'].values, "F1", "C_MCC")
    gamma_mcc_plateau_mcc, x_mcc_mcc, y_mcc_mcc = calculate_plateau_curve(
        df_gamma_mcc['gamma'].values, df_gamma_mcc['MCC'].values, "MCC", "C_MCC")
    gamma_f1_plateau_f1, x_f1_f1, y_f1_f1 = calculate_plateau_curve(
        df_gamma_f1['gamma'].values, df_gamma_f1['f1'].values, "F1", "C_F1")
    gamma_mcc_plateau_f1, x_mcc_f1, y_mcc_f1 = calculate_plateau_curve(
        df_gamma_f1['gamma'].values, df_gamma_f1['MCC'].values, "MCC", "C_F1")

    # --- Plot aggiornato ---
    plt.figure(figsize=(12, 8))

    # Curve
    plt.plot(df_gamma_mcc["gamma"], df_gamma_mcc["f1"],
             color='red', label=f'F1 @ C_MCC={C_mcc_plateau_rel:.2f}')
    plt.plot(df_gamma_mcc["gamma"], df_gamma_mcc["MCC"],
             color='blue', label=f'MCC @ C_MCC={C_mcc_plateau_rel:.2f}')
    plt.plot(df_gamma_f1["gamma"], df_gamma_f1["f1"], '--',
             color='orange', label=f'F1 @ C_F1={C_f1_plateau_rel:.2f}')
    plt.plot(df_gamma_f1["gamma"], df_gamma_f1["MCC"], '--',
             color='cyan', label=f'MCC @ C_F1={C_f1_plateau_rel:.2f}')

    # Plateau lines
    plt.axvline(gamma_f1_plateau_mcc, color='red', linestyle=':',
                label=f'γ plateau F1 (C_MCC) ≈ {gamma_f1_plateau_mcc:.4f}')
    plt.axvline(gamma_mcc_plateau_mcc, color='blue', linestyle=':',
                label=f'γ plateau MCC (C_MCC) ≈ {gamma_mcc_plateau_mcc:.4f}')
    plt.axvline(gamma_f1_plateau_f1, color='orange', linestyle=':',
                label=f'γ plateau F1 (C_F1) ≈ {gamma_f1_plateau_f1:.4f}')
    plt.axvline(gamma_mcc_plateau_f1, color='cyan', linestyle=':',
                label=f'γ plateau MCC (C_F1) ≈ {gamma_mcc_plateau_f1:.4f}')

    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Confronto dei plateau (MCC & F1) su gamma per C ottimali da MCC e F1")
    plt.grid(True, linestyle='dotted')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # --- Confronto grafico ---
    plt.figure(figsize=(12, 8))
    plt.plot(df_gamma_mcc["gamma"], df_gamma_mcc["f1"],
             color='red', label=f'F1 @ C_MCC={C_mcc_plateau_rel:.2f}')
    plt.plot(df_gamma_mcc["gamma"], df_gamma_mcc["MCC"],
             color='blue', label=f'MCC @ C_MCC={C_mcc_plateau_rel:.2f}')
    plt.plot(df_gamma_f1["gamma"], df_gamma_f1["f1"], '--',
             color='orange', label=f'F1 @ C_F1={C_f1_plateau_rel:.2f}')
    plt.plot(df_gamma_f1["gamma"], df_gamma_f1["MCC"], '--',
             color='cyan', label=f'MCC @ C_F1={C_f1_plateau_rel:.2f}')

    plt.axvline(gamma_plateau_mcc, color='blue', linestyle=':',
                label=f'Plateau γ (MCC) ≈ {gamma_plateau_mcc:.4f}')
    plt.axvline(gamma_plateau_f1, color='orange', linestyle=':',
                label=f'Plateau γ (F1) ≈ {gamma_plateau_f1:.4f}')

    plt.xlabel("Gamma")
    plt.ylabel("Score")
    plt.title("Confronto delle performance (MCC/F1) su gamma per due C ottimali")
    plt.grid(True, linestyle='dotted')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calcola la variazione percentuale del valore medio (F1)

    # params1 = {'C': 53.7, 'gamma': 0.0058}  # best params for full SHAP
    # params2 = {'C': 4.07, 'gamma': 0.037}  # best params for full SHAP
    # params3 = {'C': 9.95, 'gamma': 0.106}  # best params for full SHAP
    params3 = {'C': 486, 'gamma': 0.08}  # best params for full SHAP
    params3 = {'C': 266.33, 'gamma': 0.0594}  # best params for full SHAP
    # params = {'C': 50, 'gamma': 0.01}

    # print(f"Training SVM with C = {C} and gamma = {params['gamma']}")

    # Train and evaluate the model with the current C value
    classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, params3, display_plot=True
    )

    # -------------------------------------------------------
    # SHAP STABILITY VARYING C AND GAMMA
    # -------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from scipy.interpolate import UnivariateSpline

    # Presumed to be pre-defined
    # - mod1
    # - evaluate_svm_with_feature_selection
    # - undersampling_clustercentroids
    # - train_evaluate_final_svm

    SHAP_16 = [
        'TaG_delta_5d', 'TminG_delta_3d', 'HS_delta_5d', 'WetSnow_Temperature',
        'New_MF_Crust', 'Precip_3d', 'Precip_2d', 'TempGrad_HS', 'Tsnow_delta_3d',
        'TmaxG_delta_3d', 'HSnum', 'TempAmplitude_2d', 'WetSnow_CS',
        'TaG', 'Tsnow_delta_2d', 'DayOfSeason']

    res_shap16 = evaluate_svm_with_feature_selection(mod1, SHAP_16)
    best_C = res_shap16[2]['best_params']['C']
    best_gamma = res_shap16[2]['best_params']['gamma']

    feature_plus = SHAP_16 + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[SHAP_16]
    y = mod1_clean['AvalDay']
    X_resampled, y_resampled = undersampling_clustercentroids(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(
        X_test), columns=X_test.columns, index=X_test.index)

    C_values = np.arange(100, 1001, 1)
    gamma_values = np.linspace(0.006, 0.01, num=41)
    performance_results = []

    for C in C_values:
        for gamma in gamma_values:
            params = {'C': C, 'gamma': gamma}
            clf, metrics = train_evaluate_final_svm(
                X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False)
            performance_results.append({'C': C, 'gamma': gamma, **metrics})

    df_perf = pd.DataFrame(performance_results)

    def summarize_metric(df, metric):
        return df.groupby("C")[metric].agg([
            ("mean", "mean"),
            ("10th", lambda x: np.percentile(x, 10)),
            ("25th", lambda x: np.percentile(x, 25)),
            ("50th", lambda x: np.percentile(x, 50)),
            ("75th", lambda x: np.percentile(x, 75)),
            ("90th", lambda x: np.percentile(x, 90)),
            ("min", "min"),
            ("max", "max")
        ]).reset_index()

    df_mcc = summarize_metric(df_perf, "MCC")
    df_f1 = summarize_metric(df_perf, "f1")

    plt.figure(figsize=(12, 8))
    plt.plot(df_mcc["C"], df_mcc["50th"], label="MCC Median", color="blue")
    plt.plot(df_f1["C"], df_f1["50th"], label="F1 Median", color="red")
    plt.fill_between(df_mcc["C"], df_mcc["25th"],
                     df_mcc["75th"], alpha=0.3, color="blue")
    plt.fill_between(df_f1["C"], df_f1["25th"],
                     df_f1["75th"], alpha=0.3, color="red")
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.title("Performance Metrics across C values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    def detect_plateau(x_vals, y_vals, eps=0.02, n=5):
        x_log = np.log(x_vals)
        spline = UnivariateSpline(x_log, y_vals, s=0.05)
        x_dense = np.linspace(x_log.min(), x_log.max(), 1000)
        y_dense = spline(x_dense)
        dy = np.gradient(y_dense, x_dense)
        for i in range(len(dy) - n):
            if np.all(np.abs(dy[i:i+n]) < eps):
                return np.exp(x_dense[i])
        return np.exp(x_dense[-1])

    C_f1_plateau = detect_plateau(df_f1["C"], df_f1["50th"])
    C_mcc_plateau = detect_plateau(df_mcc["C"], df_mcc["50th"])

    print(f"📌 F1 plateau at C ≈ {C_f1_plateau:.2f}")
    print(f"📌 MCC plateau at C ≈ {C_mcc_plateau:.2f}")

    # Gamma search at C = C_mcc_plateau
    gamma_values_ext = np.linspace(0.01, 0.1, 201)
    performance_gamma = []
    for gamma in gamma_values_ext:
        params = {'C': C_mcc_plateau, 'gamma': gamma}
        clf, metrics = train_evaluate_final_svm(
            X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False)
        performance_gamma.append({'gamma': gamma, **metrics})

    df_gamma = pd.DataFrame(performance_gamma)

    def detect_gamma_plateau(df, metric):
        x_log = np.log(df['gamma'].values)
        y = df[metric].values
        spline = UnivariateSpline(x_log, y, s=0.05)
        x_dense = np.linspace(x_log.min(), x_log.max(), 1000)
        y_dense = spline(x_dense)
        dy = np.gradient(y_dense, x_dense)
        for i in range(len(dy) - 5):
            if np.all(np.abs(dy[i:i+5]) < 0.02):
                return np.exp(x_dense[i])
        return np.exp(x_dense[-1])

    gamma_plateau = detect_gamma_plateau(df_gamma, 'f1')
    print(f"📌 Gamma plateau at gamma ≈ {gamma_plateau:.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_gamma['gamma'], df_gamma['f1'], 'o', alpha=0.3, label='F1')
    plt.axvline(gamma_plateau, linestyle='--', color='red',
                label=f'Plateau @ gamma ≈ {gamma_plateau:.4f}')
    plt.title("Gamma Plateau Detection")
    plt.xlabel("Gamma")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
