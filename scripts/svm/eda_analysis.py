# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:10:15 2025

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
from scipy.stats import ttest_ind


def check_feature_consistency(df, expected_binary_features):
    """
    Controlla se le feature binarie sono effettivamente binarie (0/1).
    """
    for col in expected_binary_features:
        unique_vals = sorted(df[col].dropna().unique())
        if set(unique_vals) != {0, 1}:
            print(
                f"⚠️ Feature '{col}' non è binaria: valori trovati = {unique_vals}")


def plot_class_distribution(df, target='AvalDay', title='Class Distribution'):
    df[target].value_counts().sort_index().plot(kind='bar')
    plt.title(title)
    plt.xticks([0, 1], ['No Avalanche', 'Avalanche'], rotation=0)
    plt.ylabel("Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()


def univariate_analysis(df, target='AvalDay'):
    for col in df.columns:
        if col != target:
            plt.figure(figsize=(6, 3))
            sns.kdeplot(data=df, x=col, hue=target,
                        fill=True, common_norm=False, alpha=0.5)
            plt.title(f'Distribution of {col} by {target}')
            plt.tight_layout()
            plt.show()


def boxplot_comparison(df, target='AvalDay'):
    for col in df.columns:
        if col != target:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f'{col} vs {target}')
            plt.tight_layout()
            plt.show()


def statistical_tests(df, target='AvalDay'):
    ttest_results = []
    for col in df.columns:
        if col != target:
            group0 = df[df[target] == 0][col].dropna()
            group1 = df[df[target] == 1][col].dropna()
            if len(group0) > 1 and len(group1) > 1:
                stat, p = ttest_ind(group0, group1, equal_var=False)
                interpretation = (
                    "Very strong difference" if p < 0.01 else
                    "Significant difference" if p < 0.05 else
                    "Possible difference" if p < 0.1 else
                    "No significant difference"
                )
                ttest_results.append({
                    'Feature': col,
                    'p-value': round(p, 4),
                    'Interpretation': interpretation
                })
    ttest_df = pd.DataFrame(ttest_results).sort_values('p-value')
    print(ttest_df.to_string(index=False))


def run_eda(df, title='Original Data', binary_features_to_check=None):
    print(f"\n🔍 Running EDA on: {title}")
    plot_class_distribution(df, title=f"{title} - Class Distribution")
    if binary_features_to_check:
        check_feature_consistency(df, binary_features_to_check)
    univariate_analysis(df)
    boxplot_comparison(df)
    statistical_tests(df)
