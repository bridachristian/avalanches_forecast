# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:37:12 2025

@author: Christian
"""

import os
from pathlib import Path
from scripts.svm.data_loading import load_data
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import json
import shap
from scripts.svm.undersampling_methods import undersampling_random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scripts.svm.svm_training import (train_evaluate_final_svm,
                                      tune_train_evaluate_svm)
from sklearn import svm
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

if __name__ == '__main__':
    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\REAL_CASE_24_25\\')

    filepath = common_path / 'Season_24_25_NEW.csv'
    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\07_Model\\')

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

    # # 0. Prepara i dati
    # available_features = [col for col in feature_set if col in mod1.columns]
    # feature_plus = available_features + ['AvalDay']
    # mod1_clean = mod1[feature_plus].dropna()

    # X = mod1_clean.drop(columns=['AvalDay'])
    # y = mod1_clean['AvalDay']

    # # # 1. Rimuovi feature correlate
    # # features_correlated = remove_correlated_features(X, y)
    # # X_new = X.drop(columns=features_correlated)

    # # 2.  RandomUnderSampler su TUTTO il set dati --> se no CM sbilanciata
    # X_train_res, y_train_res = undersampling_random(X, y)

    # # 3. Split stratificato
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_train_res, y_train_res, test_size=0.25, random_state=42)

    # # 4. Scaling: fit su train, transform su test
    # scaler = MinMaxScaler()
    # # Scale the training data
    # X_train_scaled = scaler.fit_transform(X_train)
    # # Convert the scaled data into a pandas DataFrame and assign column names
    # X_train_scaled_df = pd.DataFrame(X_train_scaled,
    #                                  columns=X_train.columns, index=X_train.index)

    # # Scale the test data (using the same scaler)
    # X_test_scaled = scaler.transform(X_test)
    # # Convert the scaled test data into a pandas DataFrame and assign column names
    # X_test_scaled_df = pd.DataFrame(X_test_scaled,
    #                                 columns=X_test.columns, index=X_test.index)
    # # Tuning of parameter C and gamma for SVM classification
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

    # res = tune_train_evaluate_svm(
    #     X_train_scaled, y_train, X_test_scaled, y_test, param_grid,
    #     resampling_method='Random Undersampling')

    # classifier, evaluation_metrics = train_evaluate_final_svm(
    #     X_train_scaled, y_train, X_test_scaled, y_test, {'C': 2, 'gamma': 0.5})

    # # Use SHAP KernelExplainer for non-tree-based models like SVM
    # model = svm.SVC(C=2, gamma=0.5, probability=True, random_state=42)
    # model.fit(X_train_scaled, y_train)

    # # Use SHAP Kernel Explainer
    # explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)

    # # === Esportazione modello, scaler e features ===
    # joblib.dump(classifier, results_path / "svm_model.joblib")
    # joblib.dump(scaler, results_path / "scaler.joblib")
    # joblib.dump(X_train_scaled_df, results_path / "shap_background.joblib")

    # features_used = X_train.columns.tolist()
    # with open(results_path / "svm_features.json", "w") as f:
    #     json.dump(features_used, f)

    # # Opzionale: esporta anche metriche di valutazione
    # with open(results_path / "evaluation_metrics.json", "w") as f:
    #     json.dump(evaluation_metrics, f, indent=4)

    # print("✅ Modello SVM, scaler e features esportati correttamente!")

    # TEST ON 'NEW' DATA IMPORT
    svm_model = joblib.load(results_path / "svm_model.joblib")
    scaler = joblib.load(results_path / "scaler.joblib")
    X_background = joblib.load(results_path / "shap_background.joblib")

    with open(results_path / "svm_features.json", "r") as f:
        features_used = json.load(f)

    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()

    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']

    # Create explainer from background
    explainer = shap.KernelExplainer(svm_model.predict_proba, X_background)

    ordine_shap = ['DayOfSeason', 'HSnum', 'Precip_3d', 'PR', 'TH01G',
                   'HS_delta_1d', 'TminG_delta_3d', 'TminG_delta_5d',
                   'TmaxG_delta_2d', 'TmaxG_delta_3d', 'HS_delta_2d',
                   'TaG_delta_5d', 'TH03G', 'TmaxG_delta_1d', 'Precip_2d',
                   'Tsnow_delta_1d', 'HS_delta_5d', 'Tsnow_delta_3d',
                   'TmaxG_delta_5d', 'TempGrad_HS']

    X_ordered = X[ordine_shap]

    X_scaled = scaler.transform(X)
    # 1. Previsione
    y_pred = svm_model.predict(X_scaled)
    y_pred_df = pd.DataFrame(y_pred, index=X.index)

    # 2. Probabilità di classe positiva (AvalDay = 1)
    # colonna indice 1 → classe positiva
    y_proba = svm_model.predict_proba(X_scaled)[:, 1]

    # 3. Costruisci il DataFrame dei risultati
    results_df = pd.DataFrame({
        'True_AvalDay': y,
        'Predicted_AvalDay': y_pred,
        'Prob_AvalDay': y_proba
    }, index=y.index)

    # Assicurati che l'indice sia datetime
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        results_df.index = pd.to_datetime(results_df.index)

    threshold = 0.5

    # Usa uno stile professionale
    sns.set_style("whitegrid")

    # Parametri dimensione figura in cm convertiti in pollici
    fig_width_cm = 15
    fig_height_cm = 12
    fig_width = fig_width_cm / 2.54
    fig_height = fig_height_cm / 2.54

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,          # da 14 a 12
        'axes.titlesize': 14,     # da 18 a 16
        'axes.labelsize': 12,     # da 16 a 14
        'legend.fontsize': 8,    # da 12 a 10
        'xtick.labelsize': 8,    # da 12 a 10
        'ytick.labelsize': 8,    # da 12 a 10
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })

    plt.figure(figsize=(fig_width, fig_height))

    plt.plot(results_df.index, results_df['Prob_AvalDay'],
             label='Model Probability',
             color='#1f77b4', linewidth=2.2, zorder=2)

    plt.scatter(results_df.index, results_df['Prob_AvalDay'],
                color='#1f77b4', s=15, alpha=0.9,
                edgecolors='face', linewidth=0.3, zorder=3)

    plt.scatter(results_df.index[results_df['True_AvalDay'] == 1],
                results_df['Prob_AvalDay'][results_df['True_AvalDay'] == 1],
                color='#b07d02', label='Small avalanches obs.',
                marker='o', s=40, edgecolor='black', linewidth=0.7, zorder=5)

    plt.scatter(results_df.index[results_df['True_AvalDay'] == 2],
                results_df['Prob_AvalDay'][results_df['True_AvalDay'] == 2],
                color='#d62728', label='Medium-large avalanches obs.',
                marker='o', s=70, edgecolor='black', linewidth=0.9, zorder=6)

    threshold = 0.5
    plt.axhline(y=threshold, color='#7f7f7f', linestyle='--',
                linewidth=1.5, label='Probability threshold ($p = 0.5$)', zorder=1)

    above_thresh = results_df['Prob_AvalDay'] > threshold
    in_period = False
    start = None
    span_added = False

    for i in range(len(results_df)):
        if above_thresh.iloc[i] and not in_period:
            start = results_df.index[i]
            in_period = True
        elif not above_thresh.iloc[i] and in_period:
            end = results_df.index[i]
            if not span_added:
                plt.axvspan(start - pd.Timedelta(hours=12),
                            end - pd.Timedelta(hours=12),
                            color='#e0e0e0', alpha=0.75, label='Period with $p > 0.5$', zorder=0)
                span_added = True
            else:
                plt.axvspan(start - pd.Timedelta(hours=12),
                            end - pd.Timedelta(hours=12),
                            color='#e0e0e0', alpha=0.75, zorder=0)
            in_period = False

    if in_period:
        end = results_df.index[-1]
        if not span_added:
            plt.axvspan(start - pd.Timedelta(hours=12),
                        end - pd.Timedelta(hours=12),
                        color='#e0e0e0', alpha=0.75, label='Period with $p > 0.5$', zorder=0)
        else:
            plt.axvspan(start - pd.Timedelta(hours=12),
                        end - pd.Timedelta(hours=12),
                        color='#e0e0e0', alpha=0.75, zorder=0)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    plt.xticks(rotation=90, ha='center')

    plt.title("SVM predicted probability – Season 2024–2025")
    plt.xlabel("Date")
    plt.ylabel("Model probability")
    plt.ylim(0, 1.05)

    plt.legend(
        loc='upper right',            # sposta in alto a destra, più pulito
        frameon=True,                 # mantiene il bordo
        fancybox=True,                # bordi arrotondati
        shadow=False,                 # niente ombra per pulizia
        # borderpad=0.5,                # padding più stretto
        # borderaxespad=0.5,            # padding esterno
        facecolor='white',            # sfondo bianco
        edgecolor='gray',             # bordo grigio chiaro
        framealpha=0.5,              # leggera trasparenza
        fontsize=8                  # font un po' più piccolo
    )
    plt.tight_layout()
    plt.show()

    # ---- SHAP ANALYSIS ---------------------------------------------

    # 1. Scala i dati reali
    X_real_scaled = scaler.transform(X)
    X_real_scaled_df = pd.DataFrame(
        X_real_scaled, columns=X.columns, index=results_df.index)

    # 2. Calcola SHAP
    shap_values = explainer.shap_values(X_real_scaled)
    shap_values_class1 = shap_values[:, :, 1]

    shap.summary_plot(shap_values_class1, X_real_scaled_df)

    # 3. SHAP dataframe
    shap_df = pd.DataFrame(
        shap_values_class1, columns=X.columns, index=results_df.index)

    # 4. Ordina feature per importanza media assoluta
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)
    shap_df = shap_df[mean_abs_shap.index]

    # 5. Ordina per data
    shap_df_sorted = shap_df.sort_index()

    # 6. Somma SHAP
    shap_df_sorted['Sum of SHAP'] = shap_df_sorted.sum(axis=1)
    first_col = shap_df_sorted.pop('Sum of SHAP')
    shap_df_sorted.insert(0, 'Sum of SHAP', first_col)
    shap_df_sorted.insert(1, ' ', np.nan)

    # 7. Allinea `results_df` all’indice shap_df_sorted
    results_df = results_df.set_index(pd.to_datetime(results_df.index))
    results_aligned = results_df.loc[shap_df_sorted.index]
    results_aligned = results_aligned.dropna(
        subset=['True_AvalDay', 'Predicted_AvalDay'])

    # 8. Aggiorna shap_df_sorted con solo le date valide
    shap_df_sorted = shap_df_sorted.loc[results_aligned.index]

    # 9. Funzione mappatura colore
    def color_map(true, pred):
        if true in [1, 2] and pred == 1:
            return 'green'       # Valanga corretta
        elif true in [1, 2] and pred == 0:
            return 'red'         # Valanga mancata
        elif true == 0 and pred == 0:
            return 'lightgrey'   # No-valanga corretta
        elif true == 0 and pred == 1:
            return 'orange'      # Falso allarme
        return 'black'

    # 10. Etichette colorate
    color_labels_sorted = [
        color_map(t, p) for t, p in zip(results_aligned['True_AvalDay'], results_aligned['Predicted_AvalDay'])
    ]

    # 11. Colormap SHAP personalizzata
    neg_colors = plt.colormaps['Blues_r'](np.linspace(0, 1, 128))
    pos_colors = plt.colormaps['Reds'](np.linspace(0.02, 1, 128))
    combined_colors = np.vstack((neg_colors, pos_colors))
    custom_cmap = ListedColormap(combined_colors)

    # 12. Heatmap
    plt.figure(figsize=(12, 16))
    sns.heatmap(
        shap_df_sorted,
        cmap=custom_cmap,
        center=0,
        cbar_kws={'label': 'SHAP value'},
        yticklabels=shap_df_sorted.index.strftime('%Y-%m-%d')
    )

    # 13. Colora etichette Y
    yticks = plt.gca().get_yticklabels()
    for label, color in zip(yticks, color_labels_sorted):
        label.set_color(color)

    # 14. Grassetto prima colonna X
    xticklabels = plt.gca().get_xticklabels()
    if xticklabels:
        xticklabels[0].set_fontweight('bold')

    # 15. Legenda
    legend_labels = [
        mpatches.Patch(color='green', label='Correct Avalanche'),
        mpatches.Patch(color='red', label='Missed Avalanche'),
        mpatches.Patch(color='orange', label='False Alarm'),
        mpatches.Patch(color='lightgrey', label='Correct No Avalanche')
    ]

    # 16. Titolo e salvataggio
    plt.title('SHAP Values Heatmap (Class 1)\nColor = Prediction Outcome')
    plt.ylabel('Date')
    plt.xlabel('Feature')
    plt.legend(handles=legend_labels, loc='upper right')
    plt.tight_layout()
    plt.show()

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

    # FORCE PLOT DI ALCUNE DATA SELEZIONATE

    date_list = ['2025-01-28', '2025-02-04', '2025-02-19', '2025-03-12']
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
        if data_target in results_df.index:
            i = results_df.index.get_loc(data_target)
            # prob = model.predict_proba(X_test_scaled)[i, 1]
            prob = results_df.iloc[i, 2]
            pred = 'Avalanche' if prob > 0.5 else 'No Avalanche'
            true = 'Avalanche' if y.loc[data_target] in [
                1, 2] else 'No Avalanche'

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
                features=X.iloc[i],
                feature_names=X.columns,
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

    date_list = ['2025-01-28', '2025-02-04', '2025-02-08', '2025-03-12']
    html_blocks = []
    base_value = explainer.expected_value[1]

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
        if data_target in results_df.index:
            i = results_df.index.get_loc(data_target)
            # prob = model.predict_proba(X_test_scaled)[i, 1]
            prob = results_df.iloc[i, 2]
            pred = 'Avalanche' if prob > 0.5 else 'No Avalanche'
            true = 'Avalanche' if y.loc[data_target] in [
                1, 2] else 'No Avalanche'

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

            force = shap.force_plot(
                base_value=base_value,
                shap_values=shap_values[i, :, 1],
                features=X.iloc[i],
                feature_names=X.columns,
                show=False
            )

            html = force.html()
            html = html.replace(
                '"plot_cmap":', f'"xMin": {x_min_fixed}, "xMax": {x_max_fixed}, "plot_cmap":'
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
            margin: 10px auto 40px auto;  /* Ridotto spazio superiore */
            padding: 0 40px;
            font-family: Arial, sans-serif;
            font-size: 16px;   /* 14px per tutto il testo come nel div */
            line-height: 1.4;  /* leggermente più compatto */
            background-color: #ffffff;
            color: #000000;
          }
          h3 {
            margin-bottom: 6px;
            font-size: 15px;   /* un po’ più grande, simile a quello del blocco html */
            font-weight: bold;
          }
          p {
            margin: 0;
            font-size: 14px;
          }
          p + p {
            margin: 6px 0;  /* secondo p ha margine verticale */
          }
        </style>
        """

    # Composizione finale HTML
    final_html = f"<html><head><meta charset='utf-8'>{style}{shap.getjs()}</head><body>" + \
        "".join(html_blocks) + "</body></html>"

    # Salva il file HTML
    output_file = Path("shap_force_tesi_realcase.html")
    output_file.write_text(final_html, encoding='utf-8')
    print(f"✅ File salvato: {output_file.absolute()}")

    # ---- quali valori sono importanti ----
    # Calcola la media assoluta dei valori SHAP per ogni feature (importanza globale)
    shap_abs_mean = np.abs(shap_values_class1).mean(
        axis=0)  # media per feature

    # Ordina le feature per importanza decrescente
    feature_importance = pd.Series(
        shap_abs_mean, index=X.columns).sort_values(ascending=False)

    print(feature_importance)

    import scipy.stats as stats_lib

    N = 6  # Numero di feature da analizzare
    stats = []

    for i in range(N):
        feature_name = feature_importance.index[i]
        feature_values = X[feature_name]
        shap_values_feature = shap_values_class1[:, X.columns.get_loc(
            feature_name)]

        positive_idx = shap_values_feature >= 0
        values_positive = feature_values[positive_idx]

        count = values_positive.count()
        mean_val = values_positive.mean()
        median_val = values_positive.median()
        std_val = values_positive.std()
        q25 = values_positive.quantile(0.25)
        q75 = values_positive.quantile(0.75)
        min_val = values_positive.min()
        max_val = values_positive.max()
        skewness = stats_lib.skew(values_positive)
        kurtosis = stats_lib.kurtosis(values_positive)

        stats.append({
            'Feature': feature_name,
            'Count (SHAP>0)': count,
            'Mean (SHAP>0)': mean_val,
            'Median (SHAP>0)': median_val,
            'Std Dev': std_val,
            '25% Quantile': q25,
            '75% Quantile': q75,
            'Min': min_val,
            'Max': max_val,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        })

    df_stats = pd.DataFrame(stats)
    print(df_stats)

import numpy as np
import pandas as pd

thresholds = []

for feature in feature_importance.index[:N]:
    shap_vals = shap_values_class1[:, X.columns.get_loc(feature)]
    feature_vals = X[feature]

    # Gruppo SHAP > 0 (valanga)
    valanga_vals = feature_vals[shap_vals > 0]
    # Gruppo SHAP <= 0 (no valanga)
    no_valanga_vals = feature_vals[shap_vals <= 0]

    # Medie e deviazioni standard
    mean_valanga = valanga_vals.mean()
    std_valanga = valanga_vals.std()
    mean_no_valanga = no_valanga_vals.mean()
    std_no_valanga = no_valanga_vals.std()

    # Soglia: punto medio tra le medie
    soglia = (mean_valanga + mean_no_valanga) / 2

    # Incertezza: somma deviazioni standard
    incertezza = std_valanga + std_no_valanga

    thresholds.append({
        'Feature': feature,
        'Threshold': soglia,
        'Uncertainty': incertezza,
        'Mean Valanga': mean_valanga,
        'Std Valanga': std_valanga,
        'Mean No Valanga': mean_no_valanga,
        'Std No Valanga': std_no_valanga
    })

df_thresholds = pd.DataFrame(thresholds)
print(df_thresholds)

# 1. Crea il force plot e salva l'HTML
date_list = ['2025-01-28', '2025-02-04', '2025-02-08', '2025-03-12']
html_blocks = []
base_value = explainer.expected_value[1]

# Cartella per salvare immagini, creala se non esiste
output_dir = "force_plots"
os.makedirs(output_dir, exist_ok=True)

for data_target in date_list:
    if data_target in results_df.index:
        i = results_df.index.get_loc(data_target)

        # Crea plot con matplotlib=True e show=False per poterlo personalizzare
        shap.plots.force(
            base_value=base_value,
            shap_values=shap_values[i, :, 1],
            features=X.iloc[i],
            feature_names=X.columns,
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
