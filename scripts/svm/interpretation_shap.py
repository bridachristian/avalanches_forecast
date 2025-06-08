# -*- coding: utf-8 -*-
"""
Created on Wed May  7 14:44:30 2025

@author: Christian
"""
from matplotlib import cm
import shap  # Assumendo che shap e model siano già definiti altrove
import cairosvg
from bs4 import BeautifulSoup
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

    # --- DEPENDENCE PLOT ---

    # 2. Analisi di dipendenza (dependence plot) per feature chiave
    # # Calcolo delle 6 feature più importanti
    # shap_abs_mean = np.abs(shap_values_class1).mean(axis=0)
    # # Indici delle top 6 feature
    # top_features = shap_abs_mean.argsort()[-6:][::-1]
    # top_feature_names = [X.columns[i] for i in top_features]

    # === Configuration for thesis figure aesthetics ===
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300
    })

    # === Step 1: Identify top 6 most important features ===
    shap_abs_mean = np.abs(shap_values_class1).mean(axis=0)
    top_features = shap_abs_mean.argsort()[-6:][::-1]
    top_feature_names = [X_test.columns[i] for i in top_features]

    # === Step 2: Determine shared SHAP value y-axis limits ===
    y_min = np.min(shap_values_class1)
    y_max = np.max(shap_values_class1)

    # === Step 3: Create 2x3 subplot grid ===
    fig_width = 17 / 2.54  # ≈ 6.7 inches
    fig_height = 24 / 2.54  # ≈ 9.45 inches
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # === Step 4: Plot dependence plots ===

    top_feature_names = ["DayOfSeason", "HSnum",
                         "Precip_3d", "PR", "TH01G", "HS_delta_1d"]

    custom_interactions = {
        "DayOfSeason": "HSnum",
        "HSnum": "DayOfSeason",
        "Precip_3d": "TH01G",
        "PR": "DayOfSeason",
        "TH01G": "Precip_3d",
        "HS_delta_1d": "HSnum"
    }

    for i, feature in enumerate(top_feature_names):
        interaction = custom_interactions.get(feature, 'DayOfSeason')
        shap.dependence_plot(
            feature,
            shap_values_class1,
            X_test,
            ax=axes[i],
            show=False,
            interaction_index=interaction
        )
        # shap.dependence_plot(
        #     feature,
        #     shap_values_class1,
        #     X_test,
        #     ax=axes[i],
        #     show=False,
        #     interaction_index='DayOfSeason'
        # )
        axes[i].set_title(f"{feature}", fontsize=12, weight='bold')
        axes[i].set_xlabel(f"{feature} value", fontsize=11)
        axes[i].set_ylabel("SHAP value", fontsize=11)
        axes[i].set_ylim([y_min, y_max])

    # === Step 5: Final layout and figure title ===
    fig.suptitle("SHAP Dependence Plots for Top 6 Influential Features",
                 fontsize=14, weight='bold')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.88)  # Make room for suptitle
    plt.show()

    # --- HEATMAP ---
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

    # Colori negativi: da bianco (0) a blu intenso (1)
    fucsia_shap = np.array([216, 27, 96]) / 255
    # Azzurro SHAP (negativo): #4393c3 (azzurro bluastro)
    azzurro_shap = np.array([67, 147, 195]) / 255

    # Numero di gradazioni per lato
    n_colors = 128

    # Gradienti da bianco a colore shap
    # def gradient_white_to_color(color, n):
    #     white = np.array([1, 1, 1])
    #     # Interpolazione lineare da bianco (0) al colore shap (1)
    #     return np.linspace(white, color, n)

    def gradient_white_to_color(color, n, gamma=0.5):
        white = np.array([1, 1, 1])
        # Genera valori normalizzati 0..1
        x = np.linspace(0, 1, n)
        # Applica potenza gamma per interpolazione non lineare
        x_gamma = x ** gamma
        # Interpola da bianco a colore shap con curva gamma
        return np.outer(1 - x_gamma, white) + np.outer(x_gamma, color)

    # Gradienti negativi (azzurro)
    neg_colors = gradient_white_to_color(azzurro_shap, n_colors)

    # Gradienti positivi (fucsia)
    pos_colors = gradient_white_to_color(fucsia_shap, n_colors)

    # Unisci i due gradienti (negativo + positivo)
    combined_colors = np.vstack((neg_colors[::-1], pos_colors))

    # Crea la colormap
    custom_cmap = ListedColormap(combined_colors)

    # Heatmap

    fig_width = 17 / 2.54  # ≈ 6.7 inches
    fig_height = 24 / 2.54  # ≈ 9.45 inches

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(
        shap_df_sorted, cmap=custom_cmap, center=0,
        cbar_kws={'label': 'SHAP value'},
        yticklabels=shap_df_sorted.index.strftime('%Y-%m-%d'),
        linewidth=0  # elimina linee griglia
    )

    # Disabilita griglia matplotlib (per sicurezza)
    plt.grid(False)

    # Colora le etichette Y
    for ytick, color in zip(plt.gca().get_yticklabels(), color_labels_sorted):
        ytick.set_color(color)
        ytick.set_fontsize(6)  # più piccolo

    # Grassetto alla prima colonna
    xticklabels = plt.gca().get_xticklabels()
    if xticklabels:
        xticklabels[0].set_fontweight('bold')

    plt.title('SHAP Values Heatmap (Class 1)', fontsize=14, weight='bold')
    plt.ylabel('Date', fontsize=11)
    plt.xlabel('Feature', fontsize=11)
    plt.legend(handles=legend_labels, loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- FORCE PLOTS ---

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

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # Prendi la colormap
# cmap = cm.get_cmap(shap.plots.colors.red_blue, 256)

# # Primo colore (valore 0)
# blue_shap = cmap(0)

# # Ultimo colore (valore 1)
# red_shap = cmap(255)


# def seaborn_force_like_stacked_multi(dates, X_test, shap_values, explainer, max_labels=10):
#     # Dimensioni in cm convertite in pollici per tutta la figura
#     width_in = 17 / 2.54
#     height_in = 24 / 2.54  # altezza moltiplicata per numero di date

#     fig, axs = plt.subplots(len(dates), 1, figsize=(
#         width_in, height_in), constrained_layout=True)

#     if len(dates) == 1:
#         axs = [axs]  # per uniformità

#     for ax, data_target in zip(axs, dates):
#         if data_target not in X_test.index:
#             ax.text(
#                 0.5, 0.5, f"{data_target} non presente nei dati", ha='center', va='center')
#             ax.axis('off')
#             continue

#         i = X_test.index.get_loc(data_target)
#         shap_vals = shap_values[i]
#         feature_vals = X_test.iloc[i]
#         base_value = explainer.expected_value[1]
#         pred_value = base_value + shap_vals.sum()

#         df = pd.DataFrame({
#             'Feature': feature_vals.index,
#             'Value': feature_vals.values,
#             'SHAP': shap_vals
#         })

#         df['abs_SHAP'] = np.abs(df['SHAP'])
#         df = df.sort_values(
#             by='abs_SHAP', ascending=False).reset_index(drop=True)

#         df_top = df.head(max_labels).copy()
#         df_top['Color'] = df_top['SHAP'].apply(
#             lambda x: red_shap if x > 0 else blue_shap)
#         df_top['Label'] = df_top['Feature'] + ' = ' + \
#             df_top['Value'].round(2).astype(str)

#         left = base_value
#         right = base_value
#         positions = []
#         for _, row in df_top.iterrows():
#             if row['SHAP'] >= 0:
#                 start = right
#                 end = right + row['SHAP']
#                 right = end
#             else:
#                 end = left
#                 start = left + row['SHAP']
#                 left = start
#             positions.append((start, end))
#         df_top['Start'], df_top['End'] = zip(*positions)
#         df_top['Mid'] = [(s + e) / 2 for s, e in positions]

#         # Plot barre
#         for idx, row in df_top.iterrows():
#             ax.barh(
#                 y=0.5, width=row['End'] - row['Start'], left=row['Start'],
#                 height=0.4, color=row['Color'], edgecolor='white', linewidth=2.5
#             )

#         # Baseline e predizione
#         ax.axvline(base_value, color='black', linestyle='--',
#                    linewidth=3, label='Baseline' if ax == axs[0] else "")
#         ax.axvline(pred_value, color='red', linestyle='-',
#                    linewidth=2, label='Prediction' if ax == axs[0] else "")

#         # Linee di collegamento e label orizzontali sotto
#         label_y = 0.15
#         for idx, row in enumerate(df_top.itertuples()):
#             width = abs(row.End - row.Start)
#             ax.barh(
#                 y=0.5, width=row.End - row.Start, left=row.Start,
#                 height=0.4, color=row.Color, edgecolor='white', linewidth=2.5
#             )
#             if width >= 0.03:
#                 y_pos = 0.5 if idx % 2 == 0 else 0.2
#                 label_text = f"{row.Feature}\n{row.Value:.2f}"
#                 ax.text(
#                     x=row.Mid,
#                     y=y_pos,
#                     s=label_text,
#                     ha='center',
#                     va='center',
#                     fontsize=9,
#                     color='white',
#                     fontweight='bold',
#                     wrap=True
#                 )

#         ax.set_ylim(0, 1)
#         ax.set_xlim(0, 1)
#         ax.set_yticks([])
#         ax.set_xlabel('Output Model', fontsize=11)
#         # ax.set_title(f"📅 {data_target} - SHAP Stacked Plot\nPrediction: {pred_value:.3f} | Baseline: {base_value:.3f}", fontsize=12)

#     # Solo una legenda per tutto il plot
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', fontsize=11)

#     plt.show()


# # Lista date da plottare
# date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']

# seaborn_force_like_stacked_multi(
#     dates=date_list,
#     X_test=X_test,
#     shap_values=shap_values[:, :, 1],
#     explainer=explainer,
#     max_labels=10
# )


# date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']
# shap_vals = shap_values[:, :, 1]  # Valori SHAP per la classe "valanga"


# for data_target in date_list:
#     if data_target in X_test.index:
#         i = X_test.index.get_loc(data_target)
#         prob = model.predict_proba(X_test_scaled)[i, 1]
#         pred = 'Valanga' if prob > 0.5 else 'Non valanga'
#         true = 'Valanga' if y_test.loc[data_target] == 1 else 'Non valanga'

#         print(
#             f"\n📅 {data_target} - Probabilità: {prob:.3f} | Pred: {pred} | Reale: {true}")

#         force_plot_html = shap.force_plot(
#             base_value=explainer.expected_value[1],
#             shap_values=shap_values[:, :, 1][i],
#             features=X_test.iloc[i],
#             feature_names=X_test.columns
#         )

#         shap.save_html(f"shap_force_{data_target}.html", force_plot_html)

#     # ------------------------
#     date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']
#     html_blocks = []
#     base_value = explainer.expected_value[1]

#     x_min_fixed = 0.0
#     x_max_fixed = 1.15

#     # Mappa colori e descrizioni come heatmap
#     color_map = {
#         ('Avalanche', 'Avalanche'): ('green', '✅ Correct avalanche prediction'),
#         ('No Avalanche', 'No Avalanche'): ('gray', '✅ Correct no avalanche prediction'),
#         ('No Avalanche', 'Avalanche'): ('red', '❌ Missed avalanche prediction'),
#         ('Avalanche', 'No Avalanche'): ('orange', '⚠️ False alarm')
#     }

#     for data_target in date_list:
#         if data_target in X_test.index:
#             i = X_test.index.get_loc(data_target)
#             prob = model.predict_proba(X_test_scaled)[i, 1]
#             pred = 'Avalanche' if prob > 0.5 else 'No Avalanche'
#             true = 'Avalanche' if y_test.loc[data_target] == 1 else 'No Avalanche'

#             color, label = color_map[(pred, true)]

#             # Blocco descrizione con bordo colorato
#             description = f"""
#             <div style='border-left: 10px solid {color}; padding-left: 10px; margin: 15px 0; background-color: #f0f0f0'>
#                 <h3>{data_target} | SHAP SUM: {prob:.3f} | Prediction: {pred} | Obe: {true} → <span style='color:{color}; font-weight:bold'>{label}</span></h3>
#             </div>
#             """
#             # Force plot con asse x fissato
#             force = shap.force_plot(
#                 base_value=base_value,
#                 shap_values=shap_values[i, :, 1],
#                 features=X_test.iloc[i],
#                 feature_names=X_test.columns,
#                 show=False
#             )

#             html = force.html()
#             html = html.replace(
#                 '"plot_cmap":', f'"xMin": {x_min_fixed}, "xMax": {x_max_fixed}, "plot_cmap":'
#             )

#             html_blocks.append(description + html)

#     # HTML finale
#     final_html = "<html><head><meta charset='utf-8'>" + \
#         shap.getjs() + "</head><body>" + "".join(html_blocks) + "</body></html>"

#     # Salva su file
#     output_file = Path("shap_force_4_cases.html")
#     output_file.write_text(final_html, encoding='utf-8')
#     print(f"✅ File salvato: {output_file.absolute()}")

    date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']
    html_blocks = []
    base_value = explainer.expected_value[1]

    # Limiti asse X
    x_min_fixed = 0.0
    x_max_fixed = 1.15

    # Colori e descrizioni per la legenda
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

            # Testo descrittivo migliorato, righe separate
            description = f"""
            <div style='border-left: 10px solid {color}; padding-left: 10px; margin: 15px 0;
                        background-color: #f9f9f9; font-family:Arial,sans-serif; font-size:14px'>
                <h3 style='margin-bottom: 6px'>{data_target} | SHAP SUM: {prob:.3f}</h3>
                <p style='margin: 0'><strong>Predicted:</strong> {pred} - <strong>Observed:</strong> {true}</p>
                <p style='margin: 6px 0'><strong style='color:{color}'>{label}</strong></p>
            </div>
            """

            # SHAP force plot interattivo
            force = shap.force_plot(
                base_value=base_value,
                shap_values=shap_values[i, :, 1],
                features=X_test.iloc[i],
                feature_names=X_test.columns,
                show=False
            )

            # Inserimento HTML e modifica dimensioni e margini del grafico SVG
            html = force.html()

            # Fissa asse x e aumenta dimensione SVG per evitare sormonto
            html = html.replace(
                '"plot_cmap":',
                f'"xMin": {x_min_fixed}, "xMax": {x_max_fixed}, "plot_cmap":'
            ).replace(
                'width:100%;', 'width:100%; min-width:570px; max-width:100%;'
            ).replace(
                '"height":60', '"height":150'  # aumenta altezza elementi SHAP
            )

            html_blocks.append(description + html)

    # CSS globale per il file HTML
    style = """
    <style>
      body {
        margin: 40px auto;
        padding: 0 40px;
        font-family: Arial, sans-serif;
        font-size: 18px;
        line-height: 1.6;
        background-color: #ffffff;
        color: #000000;
      }
      h3 {
        font-size: 20px;
        margin-bottom: 10px;
      }
      p {
        font-size: 18px;
        margin: 6px 0;
      }
      div {
        margin-bottom: 30px;
      }
    </style>
    """

    # Composizione finale HTML
    final_html = f"<html><head><meta charset='utf-8'>{style}{shap.getjs()}</head><body>" + \
        "".join(html_blocks) + "</body></html>"

    # Salva il file HTML
    output_file = Path("shap_force_tesi_finale_2.html")
    output_file.write_text(final_html, encoding='utf-8')
    print(f"✅ File salvato: {output_file.absolute()}")

    # 1. Crea il force plot e salva l'HTML
    import os
    date_list = ['2009-03-06', '2005-01-10', '2018-01-20', '2016-03-29']
    html_blocks = []
    base_value = explainer.expected_value[1]

    # Cartella per salvare immagini, creala se non esiste
    output_dir = "force_plots_TEST"
    os.makedirs(output_dir, exist_ok=True)

    for data_target in date_list:
        if data_target in X_test.index:
            i = X_test.index.get_loc(data_target)
            # Crea plot con matplotlib=True e show=False per poterlo personalizzare
            shap.plots.force(
                base_value=base_value,
                shap_values=shap_values[i, :, 1],
                features=X_test.iloc[i],
                feature_names=X_test.columns,
                matplotlib=True,
                show=False,
                figsize=(12, 6),
                contribution_threshold=0.15,
                text_rotation=90
            )

            ax = plt.gca()

            # Imposta limiti orizzontali (asse x)
            ax.set_xlim(0, 1.2)

            # Riduci dimensione delle label degli assi
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)

            for text in ax.texts:
                text.set_fontsize(16)  # puoi aumentare a 12, 14, ecc.

            # Salva file con nome unico per data
            filename = f"force_plot_{data_target}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()

    # --- VALORI TIPICI ---

    # Calcola media assoluta dei valori SHAP per ogni feature (stessi shap_values)
    shap_abs_mean = np.abs(shap_values_class1).mean(axis=0)
    X_test_df = pd.DataFrame(
        X_test, columns=X_test.columns, index=X_test.index)
    feature_importance = pd.Series(shap_abs_mean, index=X_test_df.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("Feature Importance (SHAP Mean):\n", feature_importance)

    # Calcolo statistiche per le prime N feature
N = 20
stats_list = []

for i in range(N):
    feature_name = feature_importance.index[i]
    feature_values = X_test_df[feature_name]  # <-- NON SCALATO
    shap_vals = shap_values_class1[:, X_test_df.columns.get_loc(feature_name)]

    mask_pos = shap_vals > 0
    mask_neg = shap_vals <= 0

    for label, mask in [('SHAP > 0', mask_pos), ('SHAP ≤ 0', mask_neg)]:
        vals = feature_values[mask]

        stats = {
            'Feature': feature_name,
            'SHAP Sign': label,
            'Count': len(vals),
            'Mean': vals.mean() if len(vals) > 0 else np.nan,
            'Median': vals.median() if len(vals) > 0 else np.nan,
            'Std': vals.std() if len(vals) > 0 else np.nan,
            'Min': vals.min() if len(vals) > 0 else np.nan,
            'Max': vals.max() if len(vals) > 0 else np.nan
        }

        stats_list.append(stats)

df_stats = pd.DataFrame(stats_list)
print("\nFeature Statistics by SHAP Sign:\n", df_stats)

thresholds = []

for i in range(N):
    feature_name = feature_importance.index[i]
    feature_values = X_test_df[feature_name]
    shap_vals = shap_values_class1[:, X_test_df.columns.get_loc(feature_name)]

    # Valori con SHAP > 0 (valanga)
    mask_pos = shap_vals > 0
    vals_pos = feature_values[mask_pos]

    # Valori con SHAP <= 0 (non valanga)
    mask_neg = shap_vals <= 0
    vals_neg = feature_values[mask_neg]

    if len(vals_pos) > 0 and len(vals_neg) > 0:
        q25_pos = np.percentile(vals_pos, 25)
        q75_neg = np.percentile(vals_neg, 75)

        # Threshold = media tra q25_pos e q75_neg
        threshold = (q25_pos + q75_neg) / 2

        # Errore = distanza assoluta tra q25_pos e q75_neg
        error = abs(q25_pos - q75_neg)

        thresholds.append({
            'Feature': feature_name,
            'Suggested Threshold': threshold,
            'Lower Bound (q25 valanga)': q25_pos,
            'Upper Bound (q75 non valanga)': q75_neg,
            'Error (distance)': error
        })

df_thresholds = pd.DataFrame(thresholds)
print("\nThreshold estimates based on average between valanga q25 and non-valanga q75:\n")
print(df_thresholds)

# --- Plotting le distribuzioni con valori reali ---

fig_width_cm = 15
fig_height_cm = 20
fig_width = fig_width_cm / 2.54
fig_height = fig_height_cm / 2.54

base_cmap = cm.get_cmap(shap.plots.colors.red_blue, 256)
red_shap = base_cmap(0)
blue_shap = base_cmap(255)

sns.set(style='whitegrid', font_scale=1.1)

fig, axes = plt.subplots(3, 2, figsize=(fig_width, fig_height))
axes = axes.flatten()

for i in range(N):
    feature_name = feature_importance.index[i]
    shap_vals = shap_values_class1[:, X_test_df.columns.get_loc(feature_name)]
    feature_values = X_test_df[feature_name]

    plot_df = pd.DataFrame({
        'Feature Value': feature_values,
        'SHAP Sign': np.where(shap_vals > 0, 'Avalanches', 'No Avalanches')
    })

    ax = axes[i]
    sns.boxplot(
        x='SHAP Sign',
        y='Feature Value',
        data=plot_df,
        order=['No Avalanches', 'Avalanches'],
        palette=[red_shap, blue_shap],
        fliersize=2,
        linewidth=1.2,
        ax=ax
    )

    # Prendi threshold e errore per questa feature
    threshold_row = df_thresholds[df_thresholds['Feature'] == feature_name]
    if not threshold_row.empty:
        threshold = threshold_row['Suggested Threshold'].values[0]
        lower = threshold_row['Lower Bound (q25 valanga)'].values[0]
        upper = threshold_row['Upper Bound (q75 non valanga)'].values[0]

        # Disegna linea orizzontale per il threshold
        ax.axhline(threshold, color='green', linestyle='--',
                   linewidth=1.5, label='Threshold')

        # Aggiungi area ombreggiata tra lower e upper bound
        ax.fill_between(
            x=[-0.5, 1.5],  # estendi un po' oltre le categorie di boxplot
            y1=lower,
            y2=upper,
            color='green',
            alpha=0.2,
            label='Error range'
        )

    ax.set_title(f'{feature_name}', fontsize=10, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(feature_name, fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    # ax.legend(fontsize=8, loc='upper right')

# Rimuovi eventuali assi vuoti
for j in range(N, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.97])  # lascia spazio per il titolo
fig.suptitle('Feature Distribution per Avalanche Classification',
             fontsize=14, weight='bold')

plt.show()
