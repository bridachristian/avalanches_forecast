# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:44:30 2025

@author: Christian
"""
from IPython.display import display
import shap
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
                                               undersampling_tomeklinks, undersampling_clustercentroids_v2)
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                                    plot_roc_curve, permutation_ranking, evaluate_svm_with_feature_selection)
from scripts.svm.utils import (save_outputfile, get_adjacent_values, PermutationImportanceWrapper,
                               remove_correlated_features, remove_low_variance, select_k_best)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.interpolate import UnivariateSpline


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

    feature_list = ['HS_delta_3d',
                    'HS_delta_2d',
                    'TmaxG_delta_3d',
                    'TempGrad_HS',
                    'DayOfSeason',
                    'TempAmplitude_5d',
                    'HS_delta_1d',
                    'TmaxG_delta_5d',
                    'TminG_delta_5d',
                    'TmaxG_delta_1d',
                    'TaG_delta_3d',
                    'T_mean',
                    'HS_delta_5d',
                    'TH03G',
                    'Precip_3d_bin',
                    'HSnum',
                    'TaG_delta_2d',
                    'Precip_5d_bin',
                    'HNnum_bin',
                    'Precip_2d']
    # feature_list = ['TaG_delta_5d',
    #                 'TminG_delta_3d',
    #                 'HS_delta_5d',
    #                 # 'WetSnow_Temperature',
    #                 'New_MF_Crust',
    #                 'Precip_3d',
    #                 'Precip_2d',
    #                 'TempGrad_HS',
    #                 'Tsnow_delta_3d',
    #                 'TmaxG_delta_3d',
    #                 'HSnum',
    #                 'TempAmplitude_2d',
    #                 # 'WetSnow_CS',
    #                 'TaG',
    #                 'Tsnow_delta_2d',
    #                 'DayOfSeason']
    res_shap16 = evaluate_svm_with_feature_selection(mod1, feature_list)
    res_shap16 = evaluate_svm_with_feature_selection(mod1, RFECV)

    available_features = [col for col in feature_list if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']

    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']

# Add target variable to the feature list
# feature_with_target = feature_list + ['AvalDay']

# # Data preprocessing: filter relevant features and drop missing values
# clean_data = mod1[feature_with_target].dropna()

# # Extract features and target variable
# X = clean_data[feature_list]
# y = clean_data['AvalDay']

features_to_remove = remove_correlated_features(X, y)

X = X.drop(columns=features_to_remove)

X_resampled, y_resampled = undersampling_clustercentroids_v2(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42)

# scaler = MinMaxScaler()
# X_train = pd.DataFrame(scaler.fit_transform(
#     X_train), columns=X_train.columns, index=X_train.index)
# X_test = pd.DataFrame(scaler.transform(
#     X_test), columns=X_test.columns, index=X_test.index)

# Step 6: Train the final model with the best hyperparameters and evaluate it
classifier, evaluation_metrics = train_evaluate_final_svm(
    # X_train, y_train, X_test, y_test, {'C': 200, 'gamma': 0.3})
    X_train, y_train, X_test, y_test, {'C': 400, 'gamma': 0.004})

# Use SHAP KernelExplainer for non-tree-based models like SVM
# model = svm.SVC(C=200, gamma=0.3, probability=True, random_state=42)
model = svm.SVC(C=400, gamma=0.004, probability=True, random_state=42)
model.fit(X_train, y_train)

explainer = shap.KernelExplainer(model.predict_proba, X_train)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

shap_values_class1 = shap_values[:, :, 1]
shap.summary_plot(shap_values_class1, X_test)

shap.dependence_plot("WetSnow_Temperature", shap_values_class1, X_test)

shap_exp = shap.Explanation(
    values=shap_values[:, :, 1],       # SHAP values for class 1
    base_values=explainer.expected_value[1],  # Base value for class 1
    data=X_test.values,                # Feature values
    feature_names=X_test.columns       # Column names
)

y_test_np = y_test.to_numpy() if hasattr(
    y_test, "to_numpy") else np.array(y_test)

# Now create cohorts using boolean indexing
cohorts = {
    "No Avalanche": shap_exp[y_test_np == 0],
    "Avalanche": shap_exp[y_test_np == 1]
}
shap.plots.bar(cohorts, max_display=16)

shap.summary_plot(cohorts["Avalanche"], X_test[y_test == 1])
shap.summary_plot(cohorts["No Avalanche"], X_test[y_test == 0])

# Replace 'shap_values' with the correct key if needed
shap.plots.violin(cohorts['Avalanche'])
# Replace 'shap_values' with the correct key if needed
shap.plots.violin(cohorts['No Avalanche'])

# for i in range(len(shap_exp)):  # Loop through the indices of shap_exp
#     shap.plots.waterfall(shap_exp[i])


# ✅ 1. Global Feature Importance (Summary Plot)
shap.summary_plot(shap_values[:, :, 1], X_test)

clustering = shap.utils.hclust(X_test, y_test)
shap.plots.bar(shap_values[:, :, 1], clustering=clustering)

# Crea Explanation object (classe positiva: 1)
explanation1 = shap.Explanation(
    values=shap_values[:, :, 1],
    base_values=explainer.expected_value[1],
    data=X_test,
    feature_names=X_test.columns
)

explanation0 = shap.Explanation(
    values=shap_values[:, :, 0],
    base_values=explainer.expected_value[0],
    data=X_test,
    feature_names=X_test.columns
)


# Ora puoi usare il bar plot
shap.plots.bar(explanation1, max_display=16)
shap.plots.bar(explanation0, max_display=16)


# ✅ 2. Summary Bar Plot
shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar")

# ✅ 3. Force Plot (Local Explanation for One Prediction)
i = 0  # Index of the sample to explain
shap.initjs()  # Enable JS visualizations

shap.force_plot(
    explainer.expected_value[1],   # base value for class 1
    shap_values[i, :, 1],          # SHAP values for instance i and class 1
    X_test.iloc[i]                 # Feature values for instance i
)


force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values[i, :, 1],
    X_test.iloc[i]
)
display(force_plot)  # <- This makes the plot appear in the notebook
# Create the force plot
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_values[10, :, 1],
    X_test.iloc[i]
)

# Save it to an HTML file
shap.save_html("shap_force_plot.html", force_plot)


# ✅ 4. Decision Plot
shap.decision_plot(
    explainer.expected_value[1],
    shap_values[:2, :, 1],        # first 5 samples
    X_test.iloc[:2]
)

# ✅ 5. Dependence Plot
shap.dependence_plot(
    "Precip_3d",               # replace with a column name from X_test
    shap_values[:, :, 1],
    X_test
)

shap.dependence_plot(
    "Precip_3d",
    shap_values[:, :, 1],
    X_test,
    interaction_index="auto"
)

# ✅ 6. Waterfall Plot
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value[1],
    shap_values[96, :, 1],
    X_test.iloc[96]
)


# shap_values_df = pd.DataFrame(
#     shap_values[:, :, 1], columns=X_train.columns)

# shap.plots.bar(shap_values)
