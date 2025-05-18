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
    y_pred = pd.Series(classifier.predict(X_test_scaled), index=X_test.index)

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
    base_value = explainer.expected_value[1]

    shap_values_class1 = shap_values[:, :, 1]

    # 1. Visualizzare il summary plot SHAP (feature importance globale)
    shap.summary_plot(shap_values_class1, X_test_scaled_df,
                      feature_names=X_test_scaled_df.columns,
                      max_display=20, plot_type='violin')

    # 2. Analisi di dipendenza (dependence plot) per feature chiave
    # Calcolo delle 6 feature più importanti
    shap_abs_mean = np.abs(shap_values_class1).mean(axis=0)
    # Indici delle top 6 feature
    top_features = shap_abs_mean.argsort()[-6:][::-1]
    top_feature_names = [X.columns[i] for i in top_features]

    # 1. Calcolo dei limiti comuni delle SHAP values
    y_min = np.min(shap_values_class1)
    y_max = np.max(shap_values_class1)

    # 2. Creazione dei subplot
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    # 3. Plot dei dependence plot per le 6 feature principali
    for i, feature in enumerate(top_feature_names):
        shap.dependence_plot(
            feature,
            shap_values_class1,
            X_test,
            ax=axes[i],
            show=False,
            interaction_index='DayOfSeason'
        )
        axes[i].set_title(f"Dependence plot: {feature}", fontsize=12)
        axes[i].set_ylim([y_min, y_max])  # Imposta la stessa scala y per tutti

    fig.suptitle("SHAP Dependence Plots for Top 6 Features",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 4. Heatmap SHAP per tutte le osservazioni
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

    # Allinea y_test all'indice (datetime) della heatmap
    y_test_aligned = y_test.loc[shap_df.index]
    y_pred_aligned = y_pred[shap_df.index]

    # FULL DATASET
    # Ordina l'indice (datetime) in ordine crescente
    shap_df_sorted = shap_df.sort_index()
    # Calcola la somma dei valori SHAP per ogni riga
    shap_df_sorted['Sum of SHAP'] = shap_df_sorted.sum(axis=1)
    first_col = shap_df_sorted.pop('Sum of SHAP')
    shap_df_sorted.insert(0, 'Sum of SHAP', first_col)
    shap_df_sorted.insert(1, ' ', np.nan)

    # Riordina anche color_labels in base al nuovo ordine
    sorted_indices = shap_df_sorted.index
    color_labels_sorted = [
        color_labels[shap_df.index.get_loc(idx)] for idx in sorted_indices]

    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib import cm

    # Base colormap (coolwarm con 256 step)
    base_cmap = cm.get_cmap(shap.plots.colors.red_blue, 256)
    colors = base_cmap(np.linspace(0, 1, 256))

    # Applica trasparenza massima al centro (valori SHAP ~0), decrescente verso gli estremi
    for i in range(256):
        # Distanza dal centro (128)
        dist = abs(i - 128) / 128  # Normalizza da 0 a 1
        # 0 in centro (max trasparenza), 1 agli estremi (nessuna trasparenza)
        alpha = dist
        colors[i, -1] = alpha

    # Colormap con alfa lineare centrata su zero
    fade_cmap = ListedColormap(colors)

    base_cmap = cm.get_cmap(shap.plots.colors.red_blue, 256)
    # Define color gradients: lightblue to blue (negative), lightred to red (positive)
    neg_colors = cm.get_cmap('Blues_r', 128)(
        np.linspace(0, 1, 128))   # skip very light blue
    pos_colors = cm.get_cmap('Reds', 128)(
        np.linspace(0.02, 1, 128))    # skip very light red

    # Combine the two colormaps with a hard break at zero
    combined_colors = np.vstack((neg_colors, pos_colors))
    custom_cmap = ListedColormap(combined_colors)

    # Heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(shap_df_sorted, cmap=custom_cmap, center=0,
                cbar_kws={'label': 'SHAP value'},
                yticklabels=shap_df_sorted.index.strftime('%Y-%m-%d'))

    # Colora le etichette Y
    for ytick, color in zip(plt.gca().get_yticklabels(), color_labels_sorted):
        ytick.set_color(color)

    # Grassetto alla prima colonna
    xticklabels = plt.gca().get_xticklabels()
    if xticklabels:
        xticklabels[0].set_fontweight('bold')

    plt.title('SHAP Values Heatmap (Class 1)\nColor = Prediction Outcome')
    plt.ylabel('Date')
    plt.xlabel('Feature')
    plt.legend(handles=legend_labels, loc='upper right')
    plt.tight_layout()
    plt.show()

    # Mappa colore per somma SHAP: positivo (blu), negativo (arancio), zero (grigio)

    # 1. Scegli l'indice della predizione da analizzare
    date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']

    for data_target in date_list:
        if data_target in X_test.index:
            i = X_test.index.get_loc(data_target)
            prob = model.predict_proba(X_test_scaled)[i, 1]
            pred = 'Valanga' if prob > 0.5 else 'Non valanga'
            true = 'Valanga' if y_test.loc[data_target] == 1 else 'Non valanga'

            print(
                f"\n📅 {data_target} - Probabilità: {prob:.3f} | Pred: {pred} | Reale: {true}")

            shap.force_plot(
                base_value=explainer.expected_value[1],
                shap_values=shap_values[:, :, 1][i],
                features=X_test.iloc[i],
                feature_names=X_test.columns,
                matplotlib=True  # oppure rimuovilo per output interattivo
            )
            # plt.title(f"SHAP - {data_target}")
            plt.show()

    # ------------------------
    date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']
    html_blocks = []
    base_value = explainer.expected_value[1]

    x_min_fixed = 0.0
    x_max_fixed = 1.15

    # Mappa colori e descrizioni come heatmap
    color_map = {
        ('Avalanche', 'Avalanche'): ('green', '✅ Correct avalanche prediction'),
        ('No Avalanche', 'No Avalanche'): ('gray', '✅ Correct no avalanche prediction'),
        ('No Avalanche', 'Avalanche'): ('red', '❌ Missed avalanche prediction'),
        ('Avalanche', 'No Avalanche'): ('orange', '⚠️ False alarm')
    }

    for data_target in date_list:
        if data_target in X_test.index:
            i = X_test.index.get_loc(data_target)
            prob = model.predict_proba(X_test_scaled)[i, 1]
            pred = 'Avalanche' if prob > 0.5 else 'No Avalanche'
            true = 'Avalanche' if y_test.loc[data_target] == 1 else 'No Avalanche'

            color, label = color_map[(pred, true)]

            # Blocco descrizione con bordo colorato
            description = f"""
            <div style='border-left: 10px solid {color}; padding-left: 10px; margin: 15px 0; background-color: #f0f0f0'>
                <h3>{data_target} | SHAP SUM: {prob:.3f} | Prediction: {pred} | Obe: {true} → <span style='color:{color}; font-weight:bold'>{label}</span></h3>
            </div>
            """
            # Force plot con asse x fissato
            force = shap.force_plot(
                base_value=base_value,
                shap_values=shap_values[i, :, 1],
                features=X_test.iloc[i],
                feature_names=X_test.columns,
                show=False
            )

            html = force.html()
            html = html.replace(
                '"plot_cmap":', f'"xMin": {x_min_fixed}, "xMax": {x_max_fixed}, "plot_cmap":'
            )

            html_blocks.append(description + html)

    # HTML finale
    final_html = "<html><head><meta charset='utf-8'>" + \
        shap.getjs() + "</head><body>" + "".join(html_blocks) + "</body></html>"

    # Salva su file
    output_file = Path("shap_force_4_cases.html")
    output_file.write_text(final_html, encoding='utf-8')
    print(f"✅ File salvato: {output_file.absolute()}")
