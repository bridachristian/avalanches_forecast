# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:29:57 2025

@author: Christian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import svm
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
from imblearn.under_sampling import NearMiss

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

from scripts.svm.eda_analysis import run_eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


def compare_distributions(original, resampled, feature, save_path=None):
    plt.figure(figsize=(8, 4))
    sns.kdeplot(original[feature], label='Original', fill=True, alpha=0.5)
    sns.kdeplot(resampled[feature], label='Resampled', fill=True, alpha=0.5)
    plt.title(f'Distribution Comparison: {feature}')
    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}/{feature}_dist_comparison.png")
    plt.show()


def ks_test_features(original, resampled):
    print("\n📊 Kolmogorov-Smirnov Test Results:")
    results = []
    for col in original.columns:
        try:
            stat, p = ks_2samp(original[col], resampled[col])
            results.append((col, stat, p))
        except Exception as e:
            print(f"{col}: error - {e}")
    results_df = pd.DataFrame(
        results, columns=['Feature', 'KS_Statistic', 'p-value'])
    return results_df.sort_values('KS_Statistic', ascending=False)


def mean_difference(original, resampled):
    diff = (original.mean() - resampled.mean()).abs()
    return diff.sort_values(ascending=False)


def check_binarity_violations(df, binary_cols):
    print("\n🚨 Binarity Violations:")
    for col in binary_cols:
        unique_vals = df[col].dropna().unique()
        if not np.all(np.isin(unique_vals, [0, 1])):
            print(f"⚠️  {col}: Not binary! Unique values = {unique_vals}")

# USAGE EXAMPLE (put this at the bottom of your script or in a notebook):


if __name__ == '__main__':
    # Assume you already have these:
    # - X: original features (before resampling)
    # - X_resampled: after cluster centroids
    # - binary_columns: manually defined if needed
    # Load and clean data
    # .........................................................................
    #  TEST 1: HSnum vs HN_3d, full avalanche dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    # filepath = common_path / 'mod1_newfeatures.csv'
    filepath = common_path / 'mod1_certified.csv'
    # results_path = Path(
    #     'C:\\Users\\Christian\\OneDrive\\Desktop\\Family
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type

    # --- AFTER RESAMPLING ---
    feature = ['TaG', 'TminG', 'TmaxG', 'HSnum',
               'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d',
               'HS_delta_2d', 'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
               'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
               'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
               'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
               'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
               'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
               'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
               'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
               'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
               'Precip_5d', 'Penetration_ratio',
               # 'WetSnow_CS', 'WetSnow_Temperature',
               'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d',
               'Tsnow_delta_2d', 'Tsnow_delta_3d', 'Tsnow_delta_5d',
               'ConsecWetSnowDays', 'MF_Crust_Present',
               'New_MF_Crust', 'ConsecCrustDays']

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']
    mod1_clean = mod1_clean.dropna()

    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    X_resampled, y_resampled = undersampling_clustercentroids(X_new, y)

    df_resampled = X_resampled
    df_resampled['AvalDay'] = y_resampled

    binary_columns = ['New_MF_Crust', 'MF_Crust_Present'
                      # 'WetSnow_CS','WetSnow_Temperature'
                      ]  # update manually!

    print("🔍 Checking binarity...")
    check_binarity_violations(X_resampled, binary_columns)

    print("\n📉 Calculating mean differences...")
    mean_diffs = mean_difference(X, X_resampled)
    print(mean_diffs)

    print("\n🧪 Running KS tests...")
    ks_results = ks_test_features(X, X_resampled)
    print(ks_results.head(10))

    print("\n📈 Generating distribution plots...")
    for feature in mean_diffs.index:  # just top 5 most changed
        compare_distributions(X, X_resampled, feature)
