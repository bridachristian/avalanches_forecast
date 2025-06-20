# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:55:18 2025

@author: Christian
"""

import matplotlib.ticker as mticker
import time
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
from scipy.interpolate import UnivariateSpline

# --- Plateau Detection Utilities ---

# def detect_plateau(x_vals, deriv_vals, eps=0.02, n=1):
#     """Detect where the absolute gradient stays below `eps` for `n` steps."""
#     for i in range(len(deriv_vals) - n):
#         if np.all(np.abs(deriv_vals[i:i+n]) < eps):
#             return np.exp(x_vals[i])
#     return np.exp(x_vals[-1])


# def detect_plateau_relative(x_vals, y_vals, threshold=0.95):
#     """Detect where the score reaches 95% of its maximum."""
#     y_max = np.max(y_vals)
#     for i, y in enumerate(y_vals):
#         if y >= threshold * y_max:
#             return np.exp(x_vals[i])
#     return np.exp(x_vals[-1])


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
    SHAP = ['HSnum', 'TH01G', 'PR', 'DayOfSeason', 'TmaxG_delta_5d',
            'HS_delta_5d', 'TH03G', 'HS_delta_1d', 'TmaxG_delta_3d',
            'Precip_3d', 'TempGrad_HS', 'HS_delta_2d', 'TmaxG_delta_2d',
            'TminG_delta_5d', 'TminG_delta_3d', 'Tsnow_delta_3d',
            'TaG_delta_5d', 'Tsnow_delta_1d',
            'TmaxG_delta_1d', 'Precip_2d']  # 20 features

    res_shap = evaluate_svm_with_feature_selection(mod1, SHAP)

    # -------------------------------------------------------
    # OPTIMIZATION OF PARAMERTER C
    # -------------------------------------------------------

    # --- Load Best Hyperparameters from Previous Optimization ---
    best_C = res_shap[2]['best_params']['C']
    best_gamma = res_shap[2]['best_params']['gamma']

    # --- Feature Preparation ---
    feature_plus = SHAP + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()

    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']

    # 3. RandomUnderSampler sul training set
    X_resampled, y_resampled = undersampling_random(X, y)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # best_params = {'C': 100, 'gamma': 10}
    best_params = {'C': 2, 'gamma': 0.5}

    train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test,
        best_params, display_plot=True)
    # -------------------------------------------------------
    # STABILITY USING BOOTSTRAP
    # -------------------------------------------------------

    # --- Estrazione best parameters

    from sklearn.utils import resample

    best_C = res_shap[2]['best_params']['C']
    best_gamma = res_shap[2]['best_params']['gamma']

    print(f'Best paramter from GridSearch: C = {best_C}, gamma = {best_gamma}')

    # C_values = np.linspace(0.1, 10, 51)     # 50 valori
    C_values = np.arange(0.2, 20.1, 0.2)      # 100 valori
    # gamma_values = np.linspace(0.06, 0.10, 21)  # 20 valori
    gamma_values = np.arange(0.05, 2.01, 0.05)  # 40 valori
    # [0.04, 0.06, 0.08, 0.10, 0.12]

    n_iterations = 50
    results_summary = []
    start_time = time.time()
    print(
        f"\n⏱️ Inizio del processo: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    for C in C_values:
        for gamma in gamma_values:
            print(f"\nTesting C={C}, gamma={gamma}")
            bootstrap_results = []

            best_params = {'C': C, 'gamma': gamma}

            for i in range(n_iterations):
                print(
                    f'\n C = {C}, gamma = {gamma}, Bootstrap {i}/{n_iterations}')
                X_resampled_bs, y_resampled_bs = resample(
                    X_resampled, y_resampled, replace=True, random_state=i)

                X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(
                    X_resampled_bs, y_resampled_bs, test_size=0.25, random_state=42
                )

                scaler = MinMaxScaler()
                X_train_scaled_bs = scaler.fit_transform(X_train_bs)
                X_test_scaled_bs = scaler.transform(X_test_bs)

                X_train_scaled_bs = pd.DataFrame(
                    X_train_scaled_bs, columns=X.columns, index=X_train_bs.index)
                X_test_scaled_bs = pd.DataFrame(
                    X_test_scaled_bs, columns=X.columns, index=X_test_bs.index)

                model_bs, metrics_bs = train_evaluate_final_svm(
                    X_train_scaled_bs, y_train_bs, X_test_scaled_bs, y_test_bs,
                    best_params, display_plot=False
                )

                bootstrap_results.append(metrics_bs)

            df_bootstrap = pd.DataFrame(bootstrap_results)
            f1_std = df_bootstrap["f1"].std()
            mcc_std = df_bootstrap["MCC"].std()

            results_summary.append({
                "C": C,
                "gamma": gamma,
                "f1_std": f1_std,
                "mcc_std": mcc_std,
                "f1_mean": df_bootstrap["f1"].mean(),
                "mcc_mean": df_bootstrap["MCC"].mean()
            })

    end_time = time.time()
    print(
        f"\n⏱️ Fine del processo: {time.strftime('%H:%M:%S', time.localtime(end_time))}")

    duration_seconds = end_time - start_time

    # Convert to minutes
    duration_minutes = duration_seconds / 60

    # Convert to hours
    duration_hours = duration_seconds / 3600

    print(f"Duration in minutes: {duration_minutes:.2f} min")
    print(f"Duration in hours: {duration_hours:.2f} hr")

    # Confronto tra combinazioni
    df_summary = pd.DataFrame(results_summary)
    print("\n📊 Risultati sintetici (minima deviazione standard):")
    print(df_summary.sort_values(by=["f1_std", "mcc_std"]).head())

    summarise = df_summary.describe()

    # # SCATTER PLOT OF MEAN AND STANDARD DEVIATION OF F1
    # plt.figure(figsize=(10, 6))
    # scatter = sns.scatterplot(
    #     data=df_summary,
    #     x="C",
    #     y="gamma",
    #     size="f1_std",
    #     hue="f1_mean",
    #     palette="viridis",
    #     sizes=(5, 200),
    #     legend="brief"
    # )

    # plt.title("F1 Mean (color) and F1 Std (size) using Bootstrap")
    # plt.xlabel("C")
    # plt.ylabel("Gamma")

    # # Migliora la leggibilità della legenda
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.show()

    # # SCATTER PLOT OF MEAN AND STANDARD DEVIATION OF MCC
    # plt.figure(figsize=(10, 6))
    # scatter = sns.scatterplot(
    #     data=df_summary,
    #     x="C",
    #     y="gamma",
    #     size="mcc_std",
    #     hue="mcc_mean",
    #     palette="YlOrRd",
    #     sizes=(5, 200),
    #     legend="brief"
    # )

    # plt.title("MCC Mean (color) and F1 Std (size) using Bootstrap")
    # plt.xlabel("C")
    # plt.ylabel("Gamma")

    # # Migliora la leggibilità della legenda
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.show()

    # # HEATMAP OF STD DEV OF F1 AND MCC
    # fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

    # # Heatmap F1 Mean
    # sns.heatmap(
    #     df_summary.pivot(index="C", columns="gamma", values="f1_std"),
    #     ax=axes[0],
    #     cmap="YlGnBu",
    #     annot=False,
    #     fmt=".3f"
    # )
    # axes[0].set_title("F1 Std Dev (Bootstrap)", fontsize=16)
    # axes[0].set_xlabel("Gamma")
    # axes[0].set_ylabel("C")
    # axes[0].set_xticklabels(
    #     [f"{float(label.get_text()):.3f}" for label in axes[0].get_xticklabels()])
    # axes[0].invert_yaxis()

    # # Heatmap F1 Std
    # sns.heatmap(
    #     df_summary.pivot(index="C", columns="gamma", values='mcc_std'),
    #     ax=axes[1],
    #     cmap="YlOrRd",
    #     annot=False,
    #     fmt=".3f"
    # )
    # axes[1].set_title("MCC Std Dev (Bootstrap)", fontsize=16)
    # axes[1].set_xlabel("Gamma")
    # axes[1].set_ylabel("")
    # axes[1].set_xticklabels(
    #     [f"{float(label.get_text()):.3f}" for label in axes[1].get_xticklabels()])
    # axes[1].invert_yaxis()

    # plt.tight_layout()
    # plt.show()

    # --- DEVIAZIONE STANDARD ---

    # CONTOUR PLOT STANDARD DEVIATION
    C_vals = sorted(df_summary['C'].unique())
    gamma_vals = sorted(df_summary['gamma'].unique())
    C_grid, gamma_grid = np.meshgrid(C_vals, gamma_vals, indexing='ij')

    # Reshape Z values for contour plots
    f1_std_grid = df_summary.pivot(
        index="C", columns="gamma", values="f1_std").values
    mcc_std_grid = df_summary.pivot(
        index="C", columns="gamma", values="mcc_std").values

from matplotlib.ticker import MultipleLocator, AutoMinorLocator, ScalarFormatter
from matplotlib.ticker import FixedLocator, LogLocator, LogFormatter

# Convert to thesis-friendly dimensions (cm → inches)
# Thesis figure size (cm → inches)
fig_width = 20 / 2.54  # ~7.9 inches
fig_height = 10 / 2.54  # ~3.9 inches

fig, axes = plt.subplots(1, 2, figsize=(
    fig_width, fig_height), sharex=True, sharey=True)

# Colormaps: professional & readable
f1_cmap = plt.cm.viridis      # Neutral, perceptually uniform
mcc_cmap = plt.cm.plasma      # Slightly warmer, still professional

# ============================
# F1 Std Dev Plot
# ============================
cf1 = axes[0].contourf(gamma_grid, C_grid, f1_std_grid,
                       levels=30, cmap=f1_cmap, vmin=0.034, vmax=0.082)
axes[0].contour(gamma_grid, C_grid, f1_std_grid,
                levels=10, colors='black', linewidths=0.1, alpha=0.5)

# # Highlight unstable regions (top 1%)
# threshold_f1 = np.percentile(f1_std_grid, 99)
# axes[0].contourf(gamma_grid, C_grid, f1_std_grid, levels=[threshold_f1, f1_std_grid.max()],
#                  colors='none', hatches=['///'], linewidths=0.5)

# Best point
axes[0].plot(best_gamma, best_C,  'o', color='red', markersize=8,
             markeredgecolor='black', label='Best (C, γ)')
axes[0].legend(fontsize=9, loc='upper right')

# Axis labels and grid
axes[0].set_title("F1 Std Dev (Bootstrap)", fontsize=12, weight='bold')
axes[0].set_xlabel("Gamma (log scale)", fontsize=11)
axes[0].set_ylabel("C (log scale)", fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.5)

# Log scale
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# Colorbar for F1
# cbar1 = fig.colorbar(cf1, ax=axes[0])
cbar1 = fig.colorbar(cf1, ax=axes[0], pad=0.02, shrink=0.95)

cbar1.ax.set_ylabel('Std Dev', fontsize=10)

cbar1.ax.tick_params(labelsize=9)
cbar1.ax.text(1.05, -0.06, 'Stable', ha='left', va='center',
              fontsize=9, transform=cbar1.ax.transAxes)
cbar1.ax.text(1.05, 1.05, 'Unstable', ha='left', va='center',
              fontsize=9, transform=cbar1.ax.transAxes)

# ============================
# MCC Std Dev Plot
# ============================
cf2 = axes[1].contourf(gamma_grid, C_grid, mcc_std_grid,
                       levels=30, cmap=mcc_cmap, vmin=0.074, vmax=0.120)

axes[1].contour(gamma_grid, C_grid, mcc_std_grid,
                levels=10, colors='black', linewidths=0.1, alpha=0.5)

# # Highlight unstable regions (top 1%)
# threshold_mcc = np.percentile(mcc_std_grid, 99)
# axes[1].contourf(gamma_grid, C_grid, mcc_std_grid, levels=[threshold_mcc, mcc_std_grid.max()],
#                  colors='none', hatches=['///'], linewidths=0.5)

# Best point
axes[1].plot(best_gamma, best_C,  'o', color='red', markersize=8,
             markeredgecolor='black', label='Best (C, γ)')
axes[1].legend(fontsize=9, loc='upper right')

# Axis labels and grid
axes[1].set_title("MCC Std Dev (Bootstrap)", fontsize=12, weight='bold')
axes[1].set_xlabel("Gamma (log scale)", fontsize=11)
axes[1].set_ylabel("", fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.5)

# Log scale
axes[1].set_xscale('log')
axes[1].set_yscale('log')

# Colorbar for MCC
cbar2 = fig.colorbar(cf2, ax=axes[1], pad=0.02, shrink=0.95)

cbar2.ax.set_ylabel('Std Dev', fontsize=10)
cbar2.ax.tick_params(labelsize=9)
cbar2.ax.text(1.05, -0.06, 'Stable', ha='left', va='center',
              fontsize=9, transform=cbar2.ax.transAxes)
cbar2.ax.text(1.05, 1.05, 'Unstable', ha='left', va='center',
              fontsize=9, transform=cbar2.ax.transAxes)

# ============================
# Shared Tick Configuration
# ============================

# Tick personalizzati
minor_ticks_x = [0.1, 0.2, 0.5, 0.7, 1, 2]
minor_ticks_y = [0.05, 0.07, 0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


def decimal_formatter(x, pos):
    # Formatta i numeri senza notazione scientifica
    if x >= 1:
        return f"{int(x)}"
    else:
        return f"{x:g}"  # elimina zeri inutili ma evita scientifico


for ax in axes:
    ax.tick_params(axis='both', labelsize=9)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # Major ticks log base 10
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))

    # Minor ticks personalizzati (che trattiamo come major per dimensioni e label)
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minor_ticks_x))
    ax.yaxis.set_minor_locator(mticker.FixedLocator(minor_ticks_y))

    formatter = mticker.FuncFormatter(decimal_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)

    # Stesse dimensioni e lunghezza per major e minor ticks
    ax.tick_params(axis='x', which='major', labelsize=9, length=7)
    ax.tick_params(axis='y', which='major', labelsize=9, length=7)
    ax.tick_params(axis='x', which='minor', labelsize=9,
                   length=7)  # qui minor come major
    ax.tick_params(axis='y', which='minor', labelsize=9, length=7)

    # ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.grid(False)

# ============================
# Final Layout
# ============================
plt.tight_layout()
plt.show()

# --- MEDIA ---

# # CONTOUR PLOT MEAN VALUES
# C_vals = sorted(df_summary['C'].unique())
# gamma_vals = sorted(df_summary['gamma'].unique())
# C_grid, gamma_grid = np.meshgrid(C_vals, gamma_vals, indexing='ij')

# # Reshape mean values for contour plots
# f1_mean_grid = df_summary.pivot(
#     index="C", columns="gamma", values="f1_mean").values
# mcc_mean_grid = df_summary.pivot(
#     index="C", columns="gamma", values="mcc_mean").values

# fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

# # Colormaps to highlight performance (higher = better)
# f1_cmap = plt.cm.Greens
# mcc_cmap = plt.cm.Purples

# # F1 Mean plot
# cf1 = axes[0].contourf(gamma_grid, C_grid, f1_std_grid,
#                        levels=30, cmap=f1_cmap)
# cbar1 = fig.colorbar(cf1, ax=axes[0])
# axes[0].set_title("F1 Mean (Bootstrap)", fontsize=16)
# axes[0].set_xlabel("Gamma")
# axes[0].set_ylabel("C")
# axes[0].plot(best_gamma, best_C, marker='o', color='red',
#              markersize=10, label='Best (C, γ)')
# axes[0].legend()
# axes[0].contour(gamma_grid, C_grid, f1_mean_grid,
#                 levels=10, colors='black', linewidths=0.5)

# # MCC Mean plot
# cf2 = axes[1].contourf(gamma_grid, C_grid, mcc_mean_grid,
#                        levels=30, cmap=mcc_cmap)
# cbar2 = fig.colorbar(cf2, ax=axes[1])
# axes[1].set_title("MCC Mean (Bootstrap)", fontsize=16)
# axes[1].set_xlabel("Gamma")
# axes[1].set_ylabel("")
# axes[1].plot(best_gamma, best_C, marker='o', color='red',
#              markersize=10, label='Best (C, γ)')
# axes[1].legend()
# axes[1].contour(gamma_grid, C_grid, mcc_mean_grid,
#                 levels=10, colors='black', linewidths=0.5)

# plt.tight_layout()
# plt.show()

# --- NOT USED ---
# # Parametro di penalizzazione
# lambda_penalty = 0  # puoi aumentare a 1.5, 2, ecc.

# # Calcolo score penalizzati
# df_summary['f1_penalized'] = df_summary['f1_mean'] - lambda_penalty * df_summary['f1_std']
# df_summary['mcc_penalized'] = df_summary['mcc_mean'] - lambda_penalty * df_summary['mcc_std']

# # Griglie
# C_vals = sorted(df_summary['C'].unique())
# gamma_vals = sorted(df_summary['gamma'].unique())
# C_grid, gamma_grid = np.meshgrid(C_vals, gamma_vals, indexing='ij')

# f1_pen_grid = df_summary.pivot(index="C", columns="gamma", values="f1_penalized").values
# mcc_pen_grid = df_summary.pivot(index="C", columns="gamma", values="mcc_penalized").values

# # Trova massimo penalizzato per indicare il miglior punto
# best_f1 = df_summary.loc[df_summary['f1_penalized'].idxmax()]
# best_mcc = df_summary.loc[df_summary['mcc_penalized'].idxmax()]

# # Plot
# fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)

# cf1 = axes[0].contourf(gamma_grid, C_grid, f1_pen_grid, levels=30, cmap='viridis')
# axes[0].contour(gamma_grid, C_grid, f1_pen_grid, levels=10, colors='black', linewidths=0.5)
# axes[0].plot(best_f1['gamma'], best_f1['C'], 'ro', label='Best penalized (C, γ)')
# axes[0].set_title(f"F1 Mean - λ·Std (λ = {lambda_penalty})", fontsize=16)
# axes[0].set_xlabel("Gamma")
# axes[0].set_ylabel("C")
# axes[0].legend()
# fig.colorbar(cf1, ax=axes[0])

# cf2 = axes[1].contourf(gamma_grid, C_grid, mcc_pen_grid, levels=30, cmap='viridis')
# axes[1].contour(gamma_grid, C_grid, mcc_pen_grid, levels=10, colors='black', linewidths=0.5)
# axes[1].plot(best_mcc['gamma'], best_mcc['C'], 'ro', label='Best penalized (C, γ)')
# axes[1].set_title(f"MCC Mean - λ·Std (λ = {lambda_penalty})", fontsize=16)
# axes[1].set_xlabel("Gamma")
# axes[1].legend()
# fig.colorbar(cf2, ax=axes[1])

# plt.tight_layout()
# plt.show()

# # --- Define Search Grid ---
# C_values = np.arange(100, 1501, 5)  # C ∈ [100, 1500] with step of 5
# # 41 values between 0.004 and 0.012
# gamma_values = np.linspace(0.04, 0.12, num=41)

# # --- Grid Search Over C and Gamma ---
# performance_results = []

# for C in C_values:
#     for gamma in gamma_values:
#         params = {'C': C, 'gamma': gamma}
#         print(f'--- Testing C={C}, gamma={gamma} ---')

#         model, metrics = train_evaluate_final_svm(
#             X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False
#         )

#         performance_results.append({
#             'C': C,
#             'gamma': gamma,
#             **metrics
#         })

# # --- Convert to DataFrame ---
# df_performance = pd.DataFrame(performance_results)

# # --- Aggregate Performance Metrics by C ---
# def summarize_by_metric(metric):
#     return df_performance.groupby("C")[metric].agg([
#         ("mean", "mean"),
#         ("10th", lambda x: np.percentile(x, 10)),
#         ("25th", lambda x: np.percentile(x, 25)),
#         ("50th", lambda x: np.percentile(x, 50)),
#         ("75th", lambda x: np.percentile(x, 75)),
#         ("90th", lambda x: np.percentile(x, 90)),
#         ("min", "min"),
#         ("max", "max")
#     ]).reset_index()

# df_grouped_mcc = summarize_by_metric("MCC")
# df_grouped_f1 = summarize_by_metric("f1")

# # --- Plotting ---
# plt.figure(figsize=(12, 8))

# # Plot MCC
# plt.plot(df_grouped_mcc["C"], df_grouped_mcc["50th"], color='#1565C0',
#          linewidth=2, label="MCC Median (50%)")
# plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["min"], df_grouped_mcc["max"],
#                  color='#B3E5FC', alpha=0.5, label="MCC Min-Max")
# plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["10th"], df_grouped_mcc["90th"],
#                  color='#81D4FA', alpha=0.75, label="MCC 10th–90th Percentile")
# plt.fill_between(df_grouped_mcc["C"], df_grouped_mcc["25th"], df_grouped_mcc["75th"],
#                  color='#4FC3F7', alpha=1, label="MCC 25th–75th Percentile")

# # Plot F1
# plt.plot(df_grouped_f1["C"], df_grouped_f1["50th"], color='#C62828',
#          linewidth=2, label="F1-score Median (50%)")
# plt.fill_between(df_grouped_f1["C"], df_grouped_f1["min"], df_grouped_f1["max"],
#                  color='#FFCDD2', alpha=0.5, label="F1 Min-Max")
# plt.fill_between(df_grouped_f1["C"], df_grouped_f1["10th"], df_grouped_f1["90th"],
#                  color='#EF9A9A', alpha=0.75, label="F1 10th–90th Percentile")
# plt.fill_between(df_grouped_f1["C"], df_grouped_f1["25th"], df_grouped_f1["75th"],
#                  color='#E57373', alpha=1, label="F1 25th–75th Percentile")

# # Final plot settings
# plt.xlabel("C Value", fontsize=12)
# plt.ylabel("Score", fontsize=12)
# plt.title(
#     "Performance Metrics Across C Values\n(Percentiles over Gamma ∈ [0.004, 0.012])", fontsize=16)
# plt.legend(fontsize=10)
# plt.grid(True, linestyle='dotted')
# plt.tight_layout()
# plt.show()

# # ------------------------------------------------------------------------
# # --- Spline Interpolation and Plateau Detection for C ---
# # ------------------------------------------------------------------------

# # --- Prepare DataFrame for Spline ---
# df = pd.DataFrame({
#     'C': df_grouped_f1['C'],
#     'F1_median': df_grouped_f1['50th'],
#     'MCC_median': df_grouped_mcc['50th']
# })

# # Use log scale for smoother interpolation over wide range of C
# log_C = np.log(df['C'].values)
# f1_values = df['F1_median'].values
# mcc_values = df['MCC_median'].values

# # --- Fit Univariate Spline (smoothed curves) ---
# spline_f1 = UnivariateSpline(log_C, f1_values, s=0.05)
# spline_mcc = UnivariateSpline(log_C, mcc_values, s=0.05)

# # Dense interpolation range for smoother derivative estimation
# log_C_dense = np.linspace(log_C.min(), log_C.max(), 1000)
# f1_dense = spline_f1(log_C_dense)
# mcc_dense = spline_mcc(log_C_dense)

# # --- Compute Derivatives ---
# f1_derivative = np.gradient(f1_dense, log_C_dense)
# mcc_derivative = np.gradient(mcc_dense, log_C_dense)

# # --- Compute Plateaus ---
# C_f1_plateau = detect_plateau(log_C_dense, f1_derivative)
# C_mcc_plateau = detect_plateau(log_C_dense, mcc_derivative)

# C_f1_plateau_rel = detect_plateau_relative(log_C_dense, f1_dense)
# C_mcc_plateau_rel = detect_plateau_relative(log_C_dense, mcc_dense)

# # Average metrics
# avg_dense = (f1_dense + mcc_dense) / 2
# avg_derivative = np.gradient(avg_dense, log_C_dense)
# C_avg_plateau = detect_plateau(log_C_dense, avg_derivative)
# C_avg_plateau_rel = detect_plateau_relative(log_C_dense, avg_dense)

# # --- Plot ---
# plt.figure(figsize=(12, 6))

# # F1 Curve
# plt.plot(df['C'], f1_values, 'o', alpha=0.3,
#          color='red', label='F1 (original)')
# plt.plot(np.exp(log_C_dense), f1_dense, '-',
#          color='red', label='F1 spline')
# plt.axvline(C_f1_plateau_rel, linestyle='--', color='red',
#             label=f'F1 plateau @ C ≈ {int(C_f1_plateau_rel)}')

# # MCC Curve
# plt.plot(df['C'], mcc_values, 'o', alpha=0.3,
#          color='blue', label='MCC (original)')
# plt.plot(np.exp(log_C_dense), mcc_dense, '-',
#          color='blue', label='MCC spline')
# plt.axvline(C_mcc_plateau_rel, linestyle='--', color='blue',
#             label=f'MCC plateau @ C ≈ {int(C_mcc_plateau_rel)}')

# # Average Curve
# avg_original = (f1_values + mcc_values) / 2
# plt.plot(df['C'], avg_original, 's', alpha=0.5,
#          color='green', label='Avg (F1 + MCC)')
# plt.plot(np.exp(log_C_dense), avg_dense, '-',
#          color='green', label='Avg spline')
# plt.axvline(C_avg_plateau_rel, linestyle='--', color='green',
#             label=f'Avg plateau @ C ≈ {int(C_avg_plateau_rel)}')

# # Final Plot Settings
# plt.xscale('linear')
# plt.xlabel("C value")
# plt.ylabel("Score")
# plt.title(
#     "Optimizing C: Spline-Based Plateau Detection on F1 and MCC Scores", fontsize=15)
# plt.grid(True, linestyle='dotted')
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()

# # --- Output Results ---
# print(f"📌 F1 plateau detected at C ≈ {C_f1_plateau_rel:.2f}")
# print(f"📌 MCC plateau detected at C ≈ {C_mcc_plateau_rel:.2f}")
# print(f"📌 Average (F1 + MCC) plateau at C ≈ {C_avg_plateau_rel:.2f}")

# # --- Optimal C Choice ---
# C_optimal = C_avg_plateau_rel
# C_optimal = 500
# # ------------------------------------------------------------------------
# # --- Extended Gamma Sweep (Fixed C) ---
# # ------------------------------------------------------------------------

# # Define gamma search space (broader than initial range)
# gamma_values_expanded = np.linspace(0.04, 0.12, num=161)
# performance_gamma = []

# # Evaluate performance across gamma values at fixed optimal C
# for gamma in gamma_values_expanded:
#     print(
#         f'--- Testing gamma = {gamma:.5f}, fixed C = {C_optimal:.2f} ---')

#     params = {'C': C_optimal, 'gamma': gamma}
#     classifier_SVM, evaluation_metrics = train_evaluate_final_svm(
#         X_train_scaled, y_train, X_test_scaled, y_test, params, display_plot=False
#     )

#     performance_gamma.append({
#         'gamma': gamma,
#         'accuracy': evaluation_metrics['accuracy'],
#         'precision': evaluation_metrics['precision'],
#         'recall': evaluation_metrics['recall'],
#         'f1': evaluation_metrics['f1'],
#         'MCC': evaluation_metrics['MCC']
#     })

# # Convert to DataFrame
# df_gamma = pd.DataFrame(performance_gamma)

# # ------------------------------------------------------------------------
# # --- Spline Interpolation and Plateau Detection for gamma ---
# # ------------------------------------------------------------------------

# # --- Prepare DataFrame for Spline ---
# df = pd.DataFrame({
#     'gamma': df_gamma['gamma'],
#     'f1': df_gamma['f1'],
#     'MCC': df_gamma['MCC']
# })

# # Use log scale for smoother interpolation over wide range of C
# log_gamma = np.log(df['gamma'].values)
# f1_values = df['f1'].values
# mcc_values = df['MCC'].values

# # --- Fit Univariate Spline (smoothed curves) ---
# spline_f1 = UnivariateSpline(log_gamma, f1_values, s=0.05)
# spline_mcc = UnivariateSpline(log_gamma, mcc_values, s=0.05)

# # Dense interpolation range for smoother derivative estimation
# log_gamma_dense = np.linspace(log_gamma.min(), log_gamma.max(), 1000)
# f1_dense = spline_f1(log_gamma_dense)
# mcc_dense = spline_mcc(log_gamma_dense)

# # --- Compute Derivatives ---
# f1_derivative = np.gradient(f1_dense, log_gamma_dense)
# mcc_derivative = np.gradient(mcc_dense, log_gamma_dense)

# # --- Compute Plateaus ---

# gamma_f1_plateau_rel = detect_plateau_relative(
#     log_gamma_dense, f1_dense, threshold=0.95)
# gamma_mcc_plateau_rel = detect_plateau_relative(
#     log_gamma_dense, mcc_dense, threshold=0.95)

# # Average metrics
# avg_dense = (f1_dense + mcc_dense) / 2
# avg_derivative = np.gradient(avg_dense, log_gamma_dense)
# gamma_avg_plateau_rel = detect_plateau_relative(
#     log_gamma_dense, avg_dense, threshold=0.95)

# # --- Plot ---
# plt.figure(figsize=(12, 6))

# # F1 Curve
# plt.plot(df['gamma'], f1_values, 'o', alpha=0.3,
#          color='red', label='F1 (original)')
# plt.plot(np.exp(log_gamma_dense), f1_dense, '-',
#          color='red', label='F1 spline')
# plt.axvline(gamma_f1_plateau_rel, linestyle='--', color='red',
#             label=f'F1 plateau @ gamma ≈ {gamma_f1_plateau_rel:.4f}')

# # MCC Curve
# plt.plot(df['gamma'], mcc_values, 'o', alpha=0.3,
#          color='blue', label='MCC (original)')
# plt.plot(np.exp(log_gamma_dense), mcc_dense, '-',
#          color='blue', label='MCC spline')
# plt.axvline(gamma_mcc_plateau_rel, linestyle='--', color='blue',
#             label=f'MCC plateau @ gamma ≈ {gamma_mcc_plateau_rel:.4f}')

# # Average Curve
# avg_original = (f1_values + mcc_values) / 2
# plt.plot(df['gamma'], avg_original, 's', alpha=0.5,
#          color='green', label='Avg (F1 + MCC)')
# plt.plot(np.exp(log_gamma_dense), avg_dense, '-',
#          color='green', label='Avg spline')
# plt.axvline(gamma_avg_plateau_rel, linestyle='--', color='green',
#             label=f'Avg plateau @ gamma ≈ {gamma_avg_plateau_rel:.4f}')

# # Final Plot Settings
# plt.xscale('linear')
# plt.xlabel("gamma value")
# plt.ylabel("Score")
# plt.title(
#     "Optimizing Gamma: Spline-Based Plateau Detection on F1 and MCC Scores", fontsize=15)
# plt.grid(True, linestyle='dotted')
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()

# # --- Output Results ---
# print(f"📌 F1 plateau detected at gamma ≈ {gamma_f1_plateau_rel:.4f}")
# print(f"📌 MCC plateau detected at gamma ≈ {gamma_mcc_plateau_rel:.4f}")
# print(
#     f"📌 Average (F1 + MCC) plateau at gamma ≈ {gamma_avg_plateau_rel:.4f}")

# # --- Optimal C Choice ---
# gamma_optimal = gamma_avg_plateau_rel

# # ---------------------------------------------------------------------
# # Evaluate model with optimal C and optimal gamma
# # ---------------------------------------------------------------------

# classifier, evaluation_metrics = train_evaluate_final_svm(
#     X_train_scaled, y_train, X_test_scaled, y_test, {'C': C_optimal, 'gamma': gamma_optimal})

# # Original results
# original = res_shap16[2].copy()
# original['source'] = 'GridSearch'

# # New evaluation
# new_eval = evaluation_metrics.copy()
# new_eval['source'] = 'Spline+Plateau'

# # Normalize best_params into flat keys
# for d in (original, new_eval):
#     bp = d.pop('best_params')
#     d['C'] = bp['C']
#     d['gamma'] = bp['gamma']

# # Combine into DataFrame
# df_comparison = pd.DataFrame([original, new_eval])
# df_comparison = df_comparison.set_index('source')
