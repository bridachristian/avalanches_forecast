# -*- coding: utf-8 -*-
"""
Created on Sun May 18 10:37:12 2025

@author: Christian
"""

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
from scripts.svm.svm_training import train_evaluate_final_svm
from sklearn import svm


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

    # 2. Probabilità di classe positiva (AvalDay = 1)
    # colonna indice 1 → classe positiva
    y_proba = svm_model.predict_proba(X_scaled)[:, 1]

    # 3. Costruisci il DataFrame dei risultati
    results_df = pd.DataFrame({
        'True_AvalDay': y,
        'Predicted_AvalDay': y_pred,
        'Prob_AvalDay': y_proba
    }, index=y.index)

    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd

    # Assicurati che l'indice sia datetime
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        results_df.index = pd.to_datetime(results_df.index)

    threshold = 0.5

    # Usa uno stile professionale
    sns.set_style("whitegrid")

    plt.figure(figsize=(15, 8))

    plt.plot(results_df.index, results_df['Prob_AvalDay'],
             label='Model Probability', color='#0077b6', linewidth=2, zorder=1)

    # Punti sulla linea (dietro, ma sopra la linea)
    plt.scatter(results_df.index, results_df['Prob_AvalDay'],
                color='#0077b6', s=10, alpha=0.8, zorder=1)

    # 3. Valanghe piccole – davanti
    plt.scatter(results_df.index[results_df['True_AvalDay'] == 1],
                results_df['Prob_AvalDay'][results_df['True_AvalDay'] == 1],
                color='#ffa600', label='Small avalanches observed', marker='s', s=60, zorder=4)

    # 4. Valanghe medie – davanti
    plt.scatter(results_df.index[results_df['True_AvalDay'] == 2],
                results_df['Prob_AvalDay'][results_df['True_AvalDay'] == 2],
                color='#d62728', label='Medium-large avalanches observed', marker='o', s=120, zorder=5)

    # 2. Linea di soglia con etichetta
    plt.axhline(y=threshold, color='#333333', linestyle='--',
                linewidth=1.2, label='Probability threshold ($p = 0.5$)')

    # 5. Evidenzia periodi oltre soglia (bande rosse trasparenti)
    above_thresh = results_df['Prob_AvalDay'] > threshold
    in_period = False
    start = None
    span_added = False  # Per aggiungere la legenda una sola volta

    for i in range(len(results_df)):
        if above_thresh.iloc[i] and not in_period:
            start = results_df.index[i]
            in_period = True
        elif not above_thresh.iloc[i] and in_period:
            end = results_df.index[i]
            if not span_added:
                plt.axvspan(start - pd.Timedelta(hours=12),
                            end - pd.Timedelta(hours=12),
                            color='#ff6b6b', alpha=0.2, label='Period with $p > 0.5$', zorder=0)
                span_added = True
            else:
                plt.axvspan(start - pd.Timedelta(hours=12),
                            end - pd.Timedelta(hours=12),
                            color='#ff6b6b', alpha=0.2, zorder=0)
            in_period = False

    if in_period:
        end = results_df.index[-1]
        if not span_added:
            plt.axvspan(start - pd.Timedelta(hours=12),
                        end - pd.Timedelta(hours=12),
                        color='#ff6b6b', alpha=0.2, label='Period with $p > 0.5$', zorder=0)
        else:
            plt.axvspan(start - pd.Timedelta(hours=12),
                        end - pd.Timedelta(hours=12),
                        color='#ff6b6b', alpha=0.2, zorder=0)

    # 6. Formattazione asse X
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(
        interval=7))  # un tick ogni 7 giorni
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=90)

    # 7. Titoli ed etichette
    plt.title("SVM predicted probability – Season 2024–2025", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Model Probability")
    plt.ylim(0, 1)

    # 8. Legenda fuori dal grafico
    # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.legend()
    # 9. Ottimizzazione layout e griglia
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.savefig("avalanche_probability_plot.png", dpi=300)  # Salva ad alta risoluzione
    plt.show()

    # SHAP ANALYSIS

    # Real-world data: scale it
    X_real_scaled = scaler.transform(X)
    shap_values = explainer.shap_values(X_real_scaled)

    # Plot example SHAP explanation
    shap.summary_plot(shap_values, X)
