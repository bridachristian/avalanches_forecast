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
    mod1_col = mod1.columns
    print(mod1.dtypes)  # For initial data type inspection

    # -------------------------------------------------------
    # INTERPRETABILITY OF THE CODE USING SHAP TOOLS
    # -------------------------------------------------------

    feature_set = ['HSnum', 'TH01G', 'PR', 'DayOfSeason', 'TmaxG_delta_5d',
                   'HS_delta_5d', 'TH03G', 'HS_delta_1d', 'TmaxG_delta_3d',
                   'Precip_3d', 'TempGrad_HS', 'HS_delta_2d', 'TmaxG_delta_2d',
                   'TminG_delta_5d', 'TminG_delta_3d', 'Tsnow_delta_3d',
                   'TaG_delta_5d', 'Tsnow_delta_1d',
                   'TmaxG_delta_1d', 'Precip_2d']   # SHAP 20 features

    res_shap = evaluate_svm_with_feature_selection(mod1, feature_set)

    # 0. Prepara i dati
    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()

    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']

    # 1. Rimuovi feature correlate
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # 2.  RandomUnderSampler su TUTTO il set dati --> se no CM sbilanciata
    X_train_res, y_train_res = undersampling_random(X_new, y)

    # 3. Split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_res, y_train_res, test_size=0.25, random_state=42)

    # 4. Scaling: fit su train, transform su test
    scaler = MinMaxScaler()
    # Scale the training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert the scaled data into a pandas DataFrame and assign column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled,
                                     columns=X_train.columns, index=X_train.index)

    # Scale the test data (using the same scaler)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled test data into a pandas DataFrame and assign column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled,
                                    columns=X_test.columns, index=X_test.index)

    classifier, evaluation_metrics = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, {'C': 2, 'gamma': 0.5})
    # Ottieni le predizioni dal modello
    y_pred = classifier.predict(X_test_scaled)

    # Confronta con i valori veri
    prediction_status = pd.Series(
        np.where(y_pred == y_test, 'Correct', 'Wrong'),
        index=y_test.index
    )

    # Use SHAP KernelExplainer for non-tree-based models like SVM
    model = svm.SVC(C=2, gamma=0.5, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Use SHAP Kernel Explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)

    # spiegazione sui dati di test
    shap_values = explainer.shap_values(X_test_scaled)

    shap_values_class1 = shap_values[:, :, 1]

    # 1. Visualizzare il summary plot SHAP (feature importance globale)
    shap.summary_plot(shap_values_class1, X_test_scaled_df,
                      feature_names=X_test_scaled_df.columns,
                      max_display=20, plot_type='violin')

    # 2. Analisi di dipendenza (dependence plot) per feature chiave
    shap.dependence_plot('PR', shap_values_class1, X_test_scaled_df)

    # 3. Spiegazione di singole predizioni (force plot)
    shap.force_plot(explainer.expected_value[1],
                    shap_values_class1[0],
                    X_test_scaled_df.iloc[0])

    # 4. Opzionale: Feature interaction
    shap.dependence_plot('Precip_3d', shap_values_class1, X_test_scaled_df,
                         interaction_index='HS_delta_1d')

    # 5. Heatmap SHAP per tutte le osservazioni
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    # Mappa dei colori combinata (truth + correctness)

    def color_map(true, pred):
        if true == 1 and pred == 1:
            return 'green'
        elif true == 1 and pred == 0:
            return 'red'
        elif true == 0 and pred == 0:
            return 'lightgrey'
        elif true == 0 and pred == 1:
            return 'orange'

    color_labels = [color_map(t, p) for t, p in zip(y_test, y_pred)]

    legend_labels = [
        mpatches.Patch(color='green', label='Correct Avalanche'),
        mpatches.Patch(color='red', label='Missed Avalanche'),
        mpatches.Patch(color='orange', label='False Alarm'),
        mpatches.Patch(color='lightgrey', label='Correct No Avalanche')
    ]

    # Trasforma in DataFrame
    shap_df = pd.DataFrame(
        shap_values_class1, columns=X_test_scaled_df.columns,
        index=X_test_scaled_df.index)

    # ordina le feature per importanza
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)
    shap_df = shap_df[mean_abs_shap.index]
    # Estrai solo le prime 6 feature più importanti
    top6_features = mean_abs_shap.index[:6]
    shap_df_top6 = shap_df[top6_features]

    # Allinea y_test all'indice (datetime) della heatmap
    y_test_aligned = y_test.loc[shap_df.index]

    # Crea una mappa colore binaria per la presenza di valanghe
    aval_colors = y_test_aligned.map({0: 'lightgrey', 1: 'red'})

    # Ordina l'indice (datetime) in ordine crescente
    shap_df_top6_sorted = shap_df_top6.sort_index()

    # Riordina anche color_labels in base al nuovo ordine
    sorted_indices = shap_df_top6_sorted.index
    color_labels_sorted = [
        color_labels[shap_df_top6.index.get_loc(idx)] for idx in sorted_indices]

    # Heatmap
    plt.figure(figsize=(9, 16))
    sns.heatmap(shap_df_top6_sorted, cmap='coolwarm', center=0,
                cbar_kws={'label': 'SHAP value'},
                yticklabels=shap_df_top6_sorted.index.strftime('%Y-%m-%d'))

    # Colora le etichette Y
    for ytick, color in zip(plt.gca().get_yticklabels(), color_labels_sorted):
        ytick.set_color(color)

    plt.title('SHAP Values Heatmap (Class 1)\nColor = Prediction Outcome')
    plt.ylabel('Date')
    plt.xlabel('Feature')
    plt.legend(handles=legend_labels, loc='upper right')
    plt.tight_layout()
    plt.show()

    # FULL DATASET
    # Ordina l'indice (datetime) in ordine crescente
    shap_df_sorted = shap_df.sort_index()

    # Riordina anche color_labels in base al nuovo ordine
    sorted_indices = shap_df_sorted.index
    color_labels_sorted = [
        color_labels[shap_df.index.get_loc(idx)] for idx in sorted_indices]

    # Heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(shap_df_sorted, cmap='coolwarm', center=0,
                cbar_kws={'label': 'SHAP value'},
                yticklabels=shap_df_top6_sorted.index.strftime('%Y-%m-%d'))

    # Colora le etichette Y
    for ytick, color in zip(plt.gca().get_yticklabels(), color_labels_sorted):
        ytick.set_color(color)

    plt.title('SHAP Values Heatmap (Class 1)\nColor = Prediction Outcome')
    plt.ylabel('Date')
    plt.xlabel('Feature')
    plt.legend(handles=legend_labels, loc='upper right')
    plt.tight_layout()
    plt.show()

    # # Heatmap SHAP con formato data YYYY-mm-dd
    # plt.figure(figsize=(9, 16))
    # ax = sns.heatmap(shap_df, cmap='coolwarm', center=0)

    # # Converti l'indice in datetime (se necessario)
    # shap_df_top6.index = pd.to_datetime(shap_df_top6.index)

    # # Mostra tutte le date sull'asse Y
    # ax.set_yticks(np.arange(len(shap_df_top6)) + 0.5)
    # ax.set_yticklabels(shap_df.index.strftime('%Y-%m-%d'), rotation=0, fontsize=8)

    # plt.title('SHAP Values Heatmap (Class 1)')
    # plt.ylabel('Date')
    # plt.xlabel('Feature')
    # plt.tight_layout()
    # plt.show()

# shap.dependence_plot("WetSnow_Temperature", shap_values_class1, X_test)

# shap_exp = shap.Explanation(
#     values=shap_values[:, :, 1],       # SHAP values for class 1
#     base_values=explainer.expected_value[1],  # Base value for class 1
#     data=X_test_scaled.values,                # Feature values
#     feature_names=X_test_scaled.columns       # Column names
# )

# y_test_np = X_test_scaled.to_numpy() if hasattr(
#     X_test_scaled, "to_numpy") else np.array(X_test_scaled)

# # Now create cohorts using boolean indexing
# cohorts = {
#     "No Avalanche": shap_exp[y_test_np == 0],
#     "Avalanche": shap_exp[y_test_np == 1]
# }
# shap.plots.bar(cohorts, max_display=16)

# shap.summary_plot(cohorts["Avalanche"], X_test[y_test == 1])
# shap.summary_plot(cohorts["No Avalanche"], X_test[y_test == 0])

# # Replace 'shap_values' with the correct key if needed
# shap.plots.violin(cohorts['Avalanche'])
# # Replace 'shap_values' with the correct key if needed
# shap.plots.violin(cohorts['No Avalanche'])

# # for i in range(len(shap_exp)):  # Loop through the indices of shap_exp
# #     shap.plots.waterfall(shap_exp[i])


# # ✅ 1. Global Feature Importance (Summary Plot)
# shap.summary_plot(shap_values[:, :, 1], X_test)

# clustering = shap.utils.hclust(X_test, y_test)
# shap.plots.bar(shap_values[:, :, 1], clustering=clustering)

# # Crea Explanation object (classe positiva: 1)
# explanation1 = shap.Explanation(
#     values=shap_values[:, :, 1],
#     base_values=explainer.expected_value[1],
#     data=X_test,
#     feature_names=X_test.columns
# )

# explanation0 = shap.Explanation(
#     values=shap_values[:, :, 0],
#     base_values=explainer.expected_value[0],
#     data=X_test,
#     feature_names=X_test.columns
# )


# # Ora puoi usare il bar plot
# shap.plots.bar(explanation1, max_display=16)
# shap.plots.bar(explanation0, max_display=16)


# # ✅ 2. Summary Bar Plot
# shap.summary_plot(shap_values[:, :, 1], X_test, plot_type="bar")

# # ✅ 3. Force Plot (Local Explanation for One Prediction)
# i = 0  # Index of the sample to explain
# shap.initjs()  # Enable JS visualizations

# shap.force_plot(
#     explainer.expected_value[1],   # base value for class 1
#     shap_values[i, :, 1],          # SHAP values for instance i and class 1
#     X_test.iloc[i]                 # Feature values for instance i
# )


# force_plot = shap.force_plot(
#     explainer.expected_value[1],
#     shap_values[i, :, 1],
#     X_test.iloc[i]
# )
# display(force_plot)  # <- This makes the plot appear in the notebook
# # Create the force plot
# force_plot = shap.force_plot(
#     explainer.expected_value[1],
#     shap_values[10, :, 1],
#     X_test.iloc[i]
# )

# # Save it to an HTML file
# shap.save_html("shap_force_plot.html", force_plot)


# # ✅ 4. Decision Plot
# shap.decision_plot(
#     explainer.expected_value[1],
#     shap_values[:2, :, 1],        # first 5 samples
#     X_test.iloc[:2]
# )

# # ✅ 5. Dependence Plot
# shap.dependence_plot(
#     "Precip_3d",               # replace with a column name from X_test
#     shap_values[:, :, 1],
#     X_test
# )

# shap.dependence_plot(
#     "Precip_3d",
#     shap_values[:, :, 1],
#     X_test,
#     interaction_index="auto"
# )

# # ✅ 6. Waterfall Plot
# shap.plots._waterfall.waterfall_legacy(
#     explainer.expected_value[1],
#     shap_values[96, :, 1],
#     X_test.iloc[96]
# )


# # shap_values_df = pd.DataFrame(
# #     shap_values[:, :, 1], columns=X_train.columns)

# # shap.plots.bar(shap_values)
