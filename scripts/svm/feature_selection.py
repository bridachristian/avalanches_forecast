import matplotlib.colors as mcolors
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
from scripts.svm.undersampling_methods import (undersampling_random,
                                               undersampling_random_timelimited,
                                               undersampling_nearmiss,
                                               undersampling_cnn,
                                               undersampling_enn,
                                               undersampling_clustercentroids,
                                               undersampling_tomeklinks,
                                               undersampling_clustercentroids_v2,
                                               CustomUndersampler)
from scripts.svm.oversampling_methods import (oversampling_random,
                                              oversampling_smote,
                                              oversampling_adasyn,
                                              oversampling_svmsmote)
from scripts.svm.svm_training import (tune_train_evaluate_svm,
                                      train_evaluate_final_svm)
from scripts.svm.evaluation import (plot_learning_curve,
                                    plot_confusion_matrix,
                                    plot_roc_curve,
                                    permutation_ranking,
                                    evaluate_svm_with_feature_selection)
from scripts.svm.utils import (save_outputfile,
                               get_adjacent_values,
                               PermutationImportanceWrapper,
                               remove_correlated_features,
                               remove_low_variance,
                               select_k_best,
                               plot_threshold_scoring)
from scripts.svm.feature_engineering import (transform_features,
                                             transform_penetration_ratio)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (GridSearchCV,
                                     StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import numpy as np
import shap


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
    # print(mod1.dtypes)  # For initial data type inspection

    # DEFINE INITIAL FEATURE SET
    feature_set = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
               'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason',
               'HS_delta_1d', 'HS_delta_2d',
               'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
               'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
               'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
               'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
               'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
               'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
               'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean',
               'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d',
               'TempGrad_HS', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
               'Tsnow_delta_5d']

    # DEFINE PARAMETER GRID (IN SOME CASES SHOULD BE REDUCED)
    param_grid = {
        'C': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
              0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 20, 30, 40, 50, 60, 70, 80, 90,
              100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'gamma': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                  0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    # ---------------------------------------------------------------
    # --- a) FEATURE SELECTION BASED ON PERMUTATION RANKING       ---
    # ---------------------------------------------------------------

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
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Scale the test data (using the same scaler)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled test data into a pandas DataFrame and assign column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    res_tuning = tune_train_evaluate_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, param_grid,
        resampling_method=f'Random Undersampling')

    classifier = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, res_tuning['best_params'])

    # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    feature_importance_df, important_features = permutation_ranking(
        classifier[0], X_train_scaled_df, y_train)

    # Supponendo che il DataFrame abbia una colonna chiamata 'importance'
    positive_features_df = feature_importance_df[feature_importance_df['Importance_Mean'] > 0.001]

    # Opzionalmente, puoi ordinarle per importanza decrescente
    positive_features_df = positive_features_df.sort_values(
        by='Importance_Mean', ascending=False)

    # Visualizza le feature con importanza positiva
    print(positive_features_df['Feature'].tolist())
    positive_feature_names = positive_features_df['Feature'].tolist()
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\Permutation_ranking\\')

    save_outputfile(feature_importance_df, results_path /
                    'permutation_ranking_cluster_centroids.csv')

    # # ---------------------------------------------------------------
    # # --- b) TEST DIFFERENT CONFIGURATION OF FEATURES  ---
    # # ---------------------------------------------------------------

    # candidate_features = feature_set

    # # Initialize results dictionary
    # results = {}

    # # Loop through each candidate feature and test its performance
    # for feature in candidate_features:
    #     # Define the current set of features to evaluate
    #     # current_features = base_predictors + [feature]
    #     current_features = [feature]
    #     print(current_features)

    #     # Evaluate the model with the selected features
    #     result = evaluate_svm_with_feature_selection(mod1, current_features)

    #     # Store the result in the dictionary
    #     results[feature] = result

    #     # Print the evaluated feature and the result
    #     # print(f"Evaluated Feature: {feature}, Result: {result}")

    # # Identify the best-performing feature based on the evaluation metric
    # # Assuming higher is better; adjust based on metric
    # # Extract the feature with the maximum precision
    # best_feature = max(
    #     results, key=lambda x: results[x][2]['recall'])
    # max_value = results[best_feature][2]['recall']

    # print(
    #     f"Best Feature: {best_feature}, Best Result: {max_value}")

    # data = []
    # for key, (feature, model, metrics) in results.items():
    #     row = {'model': model, 'name': key}
    #     row.update(metrics)  # Merge the performance metrics
    #     data.append(row)

    # # Create the DataFrame
    # df = pd.DataFrame(data)
    # df = df.sort_values(by='recall', ascending=False)

    # save_outputfile(df, common_path / 'precision_features_NEW.csv')

    # ---------------------------------------------------------------
    # --- c) FEATURE SELECTION USING SELECT K BEST AND ANOVA      ---
    # ---------------------------------------------------------------

    # 0. Prepara i dati
    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()

    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']

    # 1. Rimuovi feature correlate
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # Create a range for k from 1 to num_columns (inclusive)
    k_range = list(range(1, X_new.shape[1]+1))
    # k_range = [1, 2, 3, 5, 7, 10, 15, 20, 25]
    results = []
    # Tuning of parameter C and gamma for SVM classification
    # param_grid_short = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #     'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    # }

    for k in k_range:
        # Select the top k features using SelectKBest
        features_selected = select_k_best(X_new, y, k=k)

        print(f'k = {k}, Features Selected: {features_selected}')

        X_selected = X_new[features_selected]

        # 2.  RandomUnderSampler su TUTTO il set dati --> se no CM sbilanciata
        X_resampled, y_resampled = undersampling_random(X_selected, y)

        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.25, random_state=42)

        # 4. Scaling: fit su train, transform su test
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(
            scaler.fit_transform(X_test), columns=X_test.columns)

        # SVM model tuning, training and evaluation
        result_SVM = tune_train_evaluate_svm(
            X_train_scaled, y_train, X_test_scaled, y_test, param_grid, resampling_method=f'feature setup {k}')

        # Final SVM classifier and evaluation
        classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
            X_train_scaled, y_train, X_test_scaled, y_test, result_SVM['best_params'])

        # Store results
        results.append({
            'num_features': k,
            # List of selected features for the current k
            'features_selected': features_selected,
            'precision': evaluation_metrics_SVM['precision'],
            'accuracy': evaluation_metrics_SVM['accuracy'],
            'recall': evaluation_metrics_SVM['recall'],
            'f1': evaluation_metrics_SVM['f1'],
            'MCC': evaluation_metrics_SVM['MCC'],
            'best_params': evaluation_metrics_SVM['best_params']
        })

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\ANOVA\\')

    save_outputfile(results_df, results_path /
                    'anova_feature_selection_new.csv')

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric in ['precision', 'accuracy', 'recall', 'f1', 'MCC']:
        plt.plot(results_df['num_features'],
                 results_df[metric], marker='o', label=metric)

    # Add labels, title, and legend
    plt.title('Feature Subset Performance - Feature selection with ANOVA')
    plt.xlabel('Number of Features')
    plt.ylabel('Score', fontsize=12)
    plt.legend(title='Metrics', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # --- d) FEATURE SELECTION USING SHAP METHOD      ---
    # ---------------------------------------------------------------
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
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Scale the test data (using the same scaler)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled test data into a pandas DataFrame and assign column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    # resampling_method = 'ClusterCentroids'
    resampling_method = 'RandomUndersampling'

    res_svm = tune_train_evaluate_svm(X_train_scaled, y_train,
                                         X_test_scaled, y_test,
                                         param_grid,
                                         resampling_method=resampling_method)
    # res_svm = tune_train_evaluate_svm(X_train_scaled, y_train_res, X_test_scaled,
    #                                   y_test, param_grid,
    #                                   resampling_method=resampling_method,
    #                                   cv=5)    # Train SVM model

    svm = svm.SVC(kernel='rbf', C=res_svm['best_params']['C'],
                  gamma=res_svm['best_params']['gamma'], probability=True)
    svm.fit(X_train_scaled, y_train)

    # Use SHAP Kernel Explainer
    explainer = shap.KernelExplainer(svm.predict_proba, X_train_scaled)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_train_scaled)

    # Ensure SHAP values are correctly formatted
    # Should be <class 'list'> or <class 'numpy.ndarray'>
    print(type(shap_values))
    print(
        f"SHAP values shape: {len(shap_values)}, Feature count: {X_train.shape[1]}")

    shap_values_class1 = shap_values[:, :, 1]  # classe positiva
    feature_importance = np.mean(np.abs(shap_values_class1), axis=0)

    # Crea dataframe ordinato per importanza
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)

    importance_df.plot(kind='barh', x='feature', y='importance',
                       figsize=(9, 16), title='Top SHAP Features for Class 1')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    importance_df['cumulative'] = importance_df['importance'].cumsum()
    importance_df['cumulative'] /= importance_df['importance'].sum()

    # Seleziona feature che coprono almeno l'85% dell'importanza
    selected_features = importance_df[importance_df['cumulative']
        <= 0.85]['feature']

    # feature_counts = np.arange(1, 35, step=1)
    feature_counts = np.arange(1, X_train_scaled.shape[1])
    scores = []

    for k in feature_counts:
        # Seleziona le top-k feature dal DataFrame delle importanze
        top_k_features = importance_df['feature'].head(k).values

        # Valuta il modello su questo sottoinsieme di feature
        res_feat = evaluate_svm_with_feature_selection(mod1, top_k_features)

        # Aggiungi i risultati alla lista
        scores.append({
            'Features': top_k_features,
            'Num Features': len(top_k_features),
            'C': res_feat[2]['best_params']['C'],
            'Gamma': res_feat[2]['best_params']['gamma'],
            'Accuracy': res_feat[2]['accuracy'],
            'Precision': res_feat[2]['precision'],
            'Recall': res_feat[2]['recall'],
            'F1-score': res_feat[2]['f1'],
            'MCC': res_feat[2]['MCC']
        })

    results_df = pd.DataFrame(scores)
    # results_df.sort_values(by='F1-score', ascending=False, inplace=True)

    # Mostra le prime righe
    print(results_df.head())

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric in ['Precision', 'Accuracy', 'Recall', 'F1-score', 'MCC']:
        plt.plot(results_df['Num Features'],
                 results_df[metric], marker='o', label=metric)

    # plt.plot(results_df['Num Features'], results_df['F1-score'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('F1-score')
    plt.title('Feature Selection via SHAP')
    plt.grid(True)
    plt.legend(title='Metrics', fontsize=10)
    plt.tight_layout()
    plt.show()

    features_22 = results_df.loc[19, 'Features']
    print("Selected features (n=22):")
    print(features_22)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\SHAP\\')

    save_outputfile(results_df, results_path /
                    'result_shap_values.csv')

    # --- HEATMAP-------
    shap_values_df = pd.DataFrame(shap_values_class1, columns=X_train.columns)
    shap_values_df = shap_values_df.T

    ordered_features = importance_df['feature'].tolist()
    # Filter to include only the features present in both DataFrames
    ordered_features = [
        f for f in ordered_features if f in shap_values_df.index]

    # Reorder the rows of shap_values_df accordingly
    shap_values_df_ordered = shap_values_df.loc[ordered_features]
    # (Opzionale) Salva i dati SHAP
    # shap_values_df.to_csv(results_path / 'shap_values.csv')
    # shap_values_df.abs().to_csv(results_path / 'shap_values_abs.csv')

    # Calcola limiti min/max per la colorbar
    vmin = shap_values_df_ordered.min().min()
    vmax = shap_values_df_ordered.max().max()

    # Crea colormap diverging centrata su 0
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Setup figura
    plt.figure(figsize=(12, 16))

    # Heatmap con colori centrati su 0
    sns.heatmap(
        shap_values_df_ordered,
        cmap=cmap,
        center=0,
        annot=False,
        fmt=".2g",
        cbar_kws={'label': 'SHAP Value'}
    )

    # Titoli e etichette
    plt.title('SHAP Value Heatmap: Feature Impact on Predictions', fontsize=14)
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Migliora layout e mostra il grafico
    plt.tight_layout()
    plt.show()

    # --- VIOLIN PLOT ---------
    plt.figure(figsize=(12, 16))

    shap.summary_plot(
        shap_values_class1,
        X_train,
        plot_type="violin",        # default: 'dot' = beeswarm
        class_names=['No Avalanche', 'Avalanche'],
        show=True, max_display=39
    )


# -------------------------------------------------------
# SHAP + Forward feature selection directly on SVM
# -------------------------------------------------------
candidate_features = pos_features
candidate_features = ['TaG_delta_5d',
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
                          'DayOfSeason',
                          'Precip_5d',
                          'TH10_tanh',
                          'TempAmplitude_1d',
                          'TaG_delta_2d',
                          'HS_delta_1d',
                          'HS_delta_3d',
                          'TaG_delta_3d']

# Store performance results for each feature set
all_results = []
summary_results = []

 feature_sets = [candidate_features[:i+1]
                  for i in range(len(candidate_features))]
  # feature_sets = feature_sets[1:]

  # Initialize an empty list to store results
  performance_results = []

   for i, feat in enumerate(feature_sets):
        # for i, feat in enumerate(feature_sets[10:20]):
        # for i, feat in enumerate(feature_sets[20:]):
        # feature_plus = feat + ['AvalDay']
        print(f" *** Feature set {i+1}: {feat} *** ")

        res_feat = evaluate_svm_with_feature_selection(mod1, feat)

        # Store the results in a dictionary
        performance_results.append({
            # 'Feature Set': i+1,
            'Features': feat,
            'Num Features': len(feat),
            'C': res_feat[2]['best_params']['C'],
            'Gamma': res_feat[2]['best_params']['gamma'],
            'Accuracy': res_feat[2]['accuracy'],
            'Precision': res_feat[2]['precision'],
            'Recall': res_feat[2]['recall'],
            'F1-score': res_feat[2]['f1'],
            'MCC': res_feat[2]['MCC']
        })

    # Convert results into a DataFrame
    df_performance=pd.DataFrame(performance_results)
    # Sort the DataFrame by the number of features
    df_performance_sorted=df_performance.sort_values(
        by="MCC", ascending=False)

    # out_shap = Path(
    #     'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\SHAP_classifier\\')

    # df_performance_sorted.to_csv(
    #     out_shap / "svm_performance_sorted.csv", index=False, sep =';')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Line for MCC (blu petrolio)
    sns.lineplot(
        x=df_performance_sorted["Num Features"],
        y=df_performance_sorted["MCC"],
        marker="o", linewidth=2, markersize=8,
        color='#1f77b4', label="MCC"  # Blu scuro
    )

    # Line for F1-score (arancione elegante)
    sns.lineplot(
        x=df_performance_sorted["Num Features"],
        y=df_performance_sorted["F1-score"],
        marker="s", linewidth=2, markersize=8,
        color='#ff7f0e', label="F1-score"  # Arancione
    )

    # Add labels and title
    plt.xlabel("Number of Features", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title("SVM Performance vs Number of Features",
              fontsize=15, weight='bold')

    # Grid, legend and ticks
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(df_performance_sorted["Num Features"])

    # Improve layout
    plt.tight_layout()
    plt.show()

    SHAP_6=['TaG_delta_5d',
              'TminG_delta_3d',
              'HS_delta_5d',
              'WetSnow_Temperature',
              'New_MF_Crust',
              'Precip_3d']

    res_shap6=evaluate_svm_with_feature_selection(mod1, SHAP_6)

    SHAP_16=['TaG_delta_5d',
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
    res_shap16=evaluate_svm_with_feature_selection(mod1, SHAP_16)

    # ---------------------------------------------------------------
    # --- d) FEATURE SELECTION USING BACKWARD FEATURE ELIMINATION      ---
    # ---------------------------------------------------------------
    # 0. Prepara i dati
    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    
    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']
    
    # 1. Rimuovi feature correlate
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)
    
    # 3. RandomUnderSampler sul training set
    X_train_res, y_train_res = undersampling_random(X_new, y)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_res, y_train_res, test_size=0.25, random_state=42)
    

    # 4. Scaling: fit su train, transform su test
    scaler = MinMaxScaler()
    # Scale the training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert the scaled data into a pandas DataFrame and assign column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Scale the test data (using the same scaler)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled test data into a pandas DataFrame and assign column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    # resampling_method = 'ClusterCentroids'
    resampling_method = 'RandomUndersampling'
    
    # 5. Parametri per GridSearch
    param_grid_short = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
    
    # 6. SVM con GridSearchCV (fuori dalla pipeline)
    svm_model = SVC(kernel='rbf')
    
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid_short,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1
    )
    
    # 7. Backward Feature Elimination (SFS)
    sfs_BW = SFS(
        estimator=grid_search,
        k_features=(1, X_train_scaled.shape[1]),
        forward=False,         # Backward selection
        floating=False,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2 
    )
    
    # 8. Fit SFS su training set bilanciato e scalato
    sfs_BW.fit(X_train_scaled, y_train)
    
    # 9. Feature selezionate
    selected_feature_indices = sfs_BW.k_feature_idx_
    selected_feature_names = [X_new.columns[i] for i in selected_feature_indices]
    
    print("Selected Features (Backward Elimination):")
    print(selected_feature_names)
    # Retrieve information about subsets
    subsets_BW=sfs_BW.subsets_
    subsets_BW_df=pd.DataFrame(subsets_BW).T
    results_path=Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\BACKWARD_FEATURE_ELIMINATION\\')
    
    save_outputfile(subsets_BW_df, results_path /
                    'BFE_feature_selection_CC.csv')
    
    # Extract the best subset
    best_subset_BW=max(subsets_BW.items(), key=lambda x: x[1]['avg_score'])

    # Retrieve the indices and names of the best features
    best_feature_indices_BW=best_subset_BW[1]['feature_idx']
    if isinstance(X_new, pd.DataFrame):
        best_feature_names_BW=[X_new.columns[i]
                                 for i in best_feature_indices_BW]
    else:
        best_feature_names_BW=list(best_feature_indices_BW)
    
    # Print the results
    print(f"Best Feature Subset Size: {len(best_feature_names_BW)}")
    print(f"Best Features: {best_feature_names_BW}")
    print(f"Best Average Score (F1 Macro): {best_subset_BW[1]['avg_score']}")
    
    # Extract data for visualization
    subset_sizes_BW=[len(subset_BW['feature_idx'])
                       for subset_BW in subsets_BW.values()]
    avg_scores_BW=[subset_BW['avg_score']
                     for subset_BW in subsets_BW.values()]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes_BW, avg_scores_BW, marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Average F1 Macro Score")
    plt.title("Feature Subset Performance - Backward feature elimination")
    plt.grid(True)
    plt.show()
    
    selected_11_indices =(1, 4, 7, 10, 14, 23, 24, 27, 28, 32, 37)
    selected_11_features = [X_train_scaled_df.columns[i] for i in selected_11_indices]
    print(selected_11_features)
   
    # ---------------------------------------------------------------
    # --- d) FEATURE SELECTION USING FORWARD FEATURE SELECTION      ---
    # ---------------------------------------------------------------
    # 0. Prepara i dati
    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    
    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']
    
    # 1. Rimuovi feature correlate
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)
    
    # 3. RandomUnderSampler sul training set
    X_train_res, y_train_res = undersampling_random(X_new, y)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_res, y_train_res, test_size=0.25, random_state=42)
    

    # 4. Scaling: fit su train, transform su test
    scaler = MinMaxScaler()
    # Scale the training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Convert the scaled data into a pandas DataFrame and assign column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    # Scale the test data (using the same scaler)
    X_test_scaled = scaler.transform(X_test)
    # Convert the scaled test data into a pandas DataFrame and assign column names
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    # resampling_method = 'ClusterCentroids'
    resampling_method = 'RandomUndersampling'
    
    # 5. Parametri per GridSearch
    param_grid_short = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
    
    # 6. SVM con GridSearchCV (fuori dalla pipeline)
    svm_model = SVC(kernel='rbf')
    
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid_short,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1
    )
    
    # Perform Sequential Feature Selection (SFS)
    sfs_FW=SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=(1, X_train_scaled.shape[1]),
        # k_features=(1, 10),
        forward=True,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=-1,              # Use all available CPU cores
        verbose=2
    )

    # Fit SFS to the data
    sfs_FW.fit(X_train_scaled, y_train)

    # Retrieve the names of the selected features
    if isinstance(X_new, pd.DataFrame):
        selected_feature_names_FW=[X_train_scaled.columns[i]
                                     for i in sfs_FW.k_feature_idx_]
    else:
        selected_feature_names_FW=list(sfs_FW.k_feature_idx_)

    print("Selected Features:", selected_feature_names_FW)

    # Retrieve information about subsets
    subsets_FW=sfs_FW.subsets_
    subsets_FW_df=pd.DataFrame(subsets_FW)

    results_path=Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\FORWARD_FEATURE_SELECTION\\')

    save_outputfile(subsets_FW_df, results_path /
                    'FFS_feature_selection_2.csv')

    # Extract the best subset
    best_subset_FW=max(subsets_FW.items(), key=lambda x: x[1]['avg_score'])

    # Retrieve the indices and names of the best features
    best_feature_indices_FW=best_subset_FW[1]['feature_idx']
    if isinstance(X_new, pd.DataFrame):
        best_feature_names_FW=[X_new.columns[i]
                                 for i in best_feature_indices_FW]
    else:
        best_feature_names_FW=list(best_feature_indices_FW)

    # Print the results
    print(f"Best Feature Subset Size: {len(best_feature_names_FW)}")
    print(f"Best Features: {best_feature_names_FW}")
    print(f"Best Average Score (F1 Macro): {best_subset_FW[1]['avg_score']}")

    # Extract data for visualization
    subset_sizes_FW=[len(subset_FW['feature_idx'])
                       for subset_FW in subsets_FW.values()]
    avg_scores_FW=[subset_FW['avg_score']
                     for subset_FW in subsets_FW.values()]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes_FW, avg_scores_FW, marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Average F1 Macro Score")
    plt.title("Feature Subset Performance - Forward Selection")
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------------
    # --- e) RECURSIVE FEATURE EXTRACTION: RFE  ---
    # ---------------------------------------------------------------

    # 0. Prepara i dati
    available_features = [col for col in feature_set if col in mod1.columns]
    feature_plus = available_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    
    X = mod1_clean.drop(columns=['AvalDay'])
    y = mod1_clean['AvalDay']
    
    # 1. Rimuovi feature correlate
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)
    
    # 2. Split stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 3. Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Hyperparameter Tuning per SVM con kernel RBF (su dati scalati e sbilanciati)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
    
    base_svc = SVC(kernel='rbf', random_state=42)
    grid_search = GridSearchCV(
        estimator=base_svc,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)
    
    # 5. RFECV con LinearSVC per la selezione delle feature
    linear_svc = LinearSVC(random_state=42, max_iter=10000)
    
    # 6. Pipeline finale: undersampling + RFECV + RBF-SVM
    pipeline = Pipeline([
        ('undersample', RandomUnderSampler()),  # Bilanciamento
        ('scaler', MinMaxScaler()),             # Scaling interno alla pipeline
        ('feature_selection', RFECV(
            estimator=linear_svc,
            step=1,
            cv=StratifiedKFold(n_splits=5),
            scoring='f1_macro',
            n_jobs=-1
        )),
        ('svc', SVC(
            kernel='rbf',
            C=best_params['C'],
            gamma=best_params['gamma'],
            random_state=42
        ))
    ])
    
    # 7. Fit della pipeline sui dati originali (non scalare prima!)
    pipeline.fit(X_train, y_train)
    
    # 8. Valutazione finale sul test set (non scalare perché lo fa la pipeline)
    y_pred = pipeline.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 9. Analisi feature selezionate
    rfecv = pipeline.named_steps['feature_selection']
    selected_features = np.array(X_new.columns)[rfecv.support_]
    
    print("Selected Features:", selected_features)
    print("Feature Rankings:", rfecv.ranking_)
    
    # 10. Plot delle performance in funzione del numero di feature
    mean_cv_scores = rfecv.cv_results_['mean_test_score']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mean_cv_scores) + 1), mean_cv_scores, marker='o')
    plt.title('Feature Selection Process - Recursive Feature Elimination with CV') 
    plt.xlabel('Number of Features')
    plt.ylabel('Mean CV Score (F1 Macro)')
    plt.grid(True)
    plt.show()
    
    # 11. Classifica delle feature
    feature_importance = pd.DataFrame({
        'Feature': np.array(X_new.columns),
        'Ranking': rfecv.ranking_
    })
    
    # Ordina per importanza (ranking più basso = più importante)
    feature_importance_sorted = feature_importance.sort_values(by='Ranking')
    
    print("Selected Features ordered by importance:")
    print(feature_importance_sorted)


   
    results_path=Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\0X_FEATURE_SELECTION\\RECURSIVE_FEATURE_ELIMINATION\\')

    save_outputfile(feature_importance_sorted, results_path /
                    'RFE_feature_selection_CC.csv')

    # ---------------------------------------------------------------
    # --- D) FEATURE EXTRACTION USING LINEAR DISCRIMINANT ANALYSIS (LDA)
    #        on SELECTED FEATURES ---
    # ---------------------------------------------------------------
    # 1. LDA on full feature dataset
    # Data preparation
    feature_plus=feature_set + ['AvalDay']
    mod1_clean=mod1[feature_plus]
    mod1_clean=mod1_clean.dropna()
    mod1_transformed=transform_features(mod1_clean.copy())

    X=mod1_transformed[feature_set]
    y=mod1_transformed['AvalDay']

    X_resampled, y_resampled=undersampling_clustercentroids(X, y)

    # Split into training and test set
    X_train, X_test, y_train, y_test=train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    common_indices=X_train.index.intersection(X_test.index)

    # # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # X_train_scaled = pd.DataFrame(
    #     X_train_scaled, columns=X_train.columns, index=X_train.index)
    # X_test_scaled = pd.DataFrame(
    #     X_test_scaled, columns=X_test.columns, index=X_test.index)

    result_SVM=tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test,
        param_grid, resampling_method='Cluster Centroids')

    classifier_SVM, evaluation_metrics_SVM=train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, result_SVM['best_params'])

    # Apply LDA for dimensionality reduction
    lda=LDA(n_components=1)

    X_train_lda=lda.fit_transform(X_train, y_train)
    X_test_lda=lda.transform(X_test)

    # Convert back to DataFrame for compatibility
    X_train_lda=pd.DataFrame(X_train_lda, index=X_train.index)
    X_test_lda=pd.DataFrame(X_test_lda, index=X_test.index)

    # Train and evaluate SVM on LDA-transformed features
    result_SVM_LDA=tune_train_evaluate_svm(
        X_train_lda, y_train, X_test_lda, y_test, param_grid, resampling_method='Cluster Centroids')

    classifier_SVM_LDA, evaluation_metrics_SVM_LDA=train_evaluate_final_svm(
        X_train_lda, y_train, X_test_lda, y_test, result_SVM_LDA['best_params'])

    # 2. SHAP without LDA

    SHAP=['TaG_delta_5d',
            'TminG_delta_3d',
            'HS_delta_5d',
            'Precip_3d',
            'Precip_2d',
            'TempGrad_HS',
            'Tsnow_delta_3d',
            'TmaxG_delta_3d',
            'HSnum',
            'TempAmplitude_2d',
            'TaG',
            'Tsnow_delta_2d',
            'DayOfSeason']

    # best_features = list(set(BestFeatures_FW_20 + BestFeatures_BW_27))

    # Data preparation
    feature_plus=SHAP + ['AvalDay']
    mod1_clean=mod1[feature_plus].dropna()
    X=mod1_clean[SHAP]
    y=mod1_clean['AvalDay']

    X_resampled, y_resampled=undersampling_clustercentroids(X, y)

    # Split into training and test set
    X_train, X_test, y_train, y_test=train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    common_indices=X_train.index.intersection(X_test.index)

    # # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # X_train_scaled = pd.DataFrame(
    #     X_train_scaled, columns=X_train.columns, index=X_train.index)
    # X_test_scaled = pd.DataFrame(
    #     X_test_scaled, columns=X_test.columns, index=X_test.index)

    result_SVM=tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, param_grid, resampling_method='Cluster Centroids')

    classifier_SVM, evaluation_metrics_SVM=train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, result_SVM['best_params'])

    # 3. SHAP + LDA

    # Apply LDA for dimensionality reduction
    lda=LDA(n_components=1)

    X_train_lda=lda.fit_transform(X_train, y_train)
    X_test_lda=lda.transform(X_test)

    # Convert back to DataFrame for compatibility
    X_train_lda=pd.DataFrame(X_train_lda, index=X_train.index)
    X_test_lda=pd.DataFrame(X_test_lda, index=X_test.index)

    # Train and evaluate SVM on LDA-transformed features
    result_SVM_LDA=tune_train_evaluate_svm(
        X_train_lda, y_train, X_test_lda, y_test, param_grid, resampling_method='Cluster Centroids')

    classifier_SVM_LDA, evaluation_metrics_SVM_LDA=train_evaluate_final_svm(
        X_train_lda, y_train, X_test_lda, y_test, result_SVM_LDA['best_params'])

    # ---------------------------------------------------------------
    # --- E) COMPARE FEATURE SELECTIONS ---
    # ---------------------------------------------------------------

    FULL=feature_set
    res_FULL=evaluate_svm_with_feature_selection(mod1, FULL)

    ANOVA=['HSnum', 'TH01G', 'DayOfSeason', 'HS_delta_1d', 'Tmin_2d', 'TaG_delta_5d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
             'T_mean', 'DegreeDays_Pos', 'Precip_2d', 'Precip_3d', 'Precip_5d', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'SnowConditionIndex']
    res_ANOVA=evaluate_svm_with_feature_selection(mod1, ANOVA)

    BFE=['TaG', 'HNnum', 'TH01G', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_3d',
           'HS_delta_5d', 'DaysSinceLastSnow', 'TempAmplitude_1d',
           'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d',
           'TaG_delta_1d', 'TaG_delta_2d', 'TaG_delta_3d', 'TaG_delta_5d',
           'TminG_delta_1d', 'TminG_delta_2d', 'TminG_delta_3d', 'TminG_delta_5d',
           'TmaxG_delta_1d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
           'T_mean', 'DegreeDays_Pos', 'Precip_1d', 'WetSnow_CS', 'TH10_tanh',
           'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
           'Tsnow_delta_5d', 'ConsecWetSnowDays', 'ConsecCrustDays']
    res_BFE=evaluate_svm_with_feature_selection(mod1, BFE)

    FFS=['HS_delta_3d', 'HS_delta_5d', 'DaysSinceLastSnow', 'TempAmplitude_1d',
           'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d', 'TaG_delta_3d',
           'TaG_delta_5d', 'TminG_delta_2d', 'TminG_delta_3d', 'TmaxG_delta_3d',
           'TmaxG_delta_5d', 'DegreeDays_Pos', 'Precip_3d', 'Precip_5d',
           'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh',
           'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d', 'Tsnow_delta_5d',
           'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays']
    res_FFS=evaluate_svm_with_feature_selection(mod1, FFS)

    RFE=['TaG', 'New_MF_Crust', 'TaG_delta_5d', 'TH10_tanh', 'TH30_tanh',
           'TmaxG_delta_2d', 'TempAmplitude_5d', 'TempAmplitude_3d', 'Tsnow_delta_1d',
           'TempAmplitude_1d', 'TminG_delta_5d', 'TmaxG_delta_3d', 'HS_delta_5d',
           'HS_delta_3d', 'Tsnow_delta_3d', 'HS_delta_1d', 'DayOfSeason', 'PR',
           'Penetration_ratio', 'Precip_2d', 'HSnum', 'WetSnow_Temperature',
           'TminG_delta_3d']
    res_RFE=evaluate_svm_with_feature_selection(mod1, RFE)

    Permutation_ranking=['AvalDay_2d',
                           'TH10_tanh',
                           'TH30_tanh',
                           'DegreeDays_cumsum_2d',
                           'TH01G',
                           'PR',
                           'HS_delta_2d',
                           'DayOfSeason',
                           'TmaxG_delta_1d',
                           'TaG',
                           'ConsecWetSnowDays',
                           'HS_delta_1d',
                           'Precip_2d',
                           'TempAmplitude_1d',
                           'TminG_delta_1d',
                           'Tmin_3d',
                           'T_mean',
                           'TminG_delta_2d',
                           'TmaxG_delta_5d',
                           'TminG_delta_5d',
                           'TempAmplitude_2d',
                           'HS_delta_3d',
                           'TaG_delta_2d',
                           'TaG_delta_1d',
                           'Precip_5d',
                           'Tsnow_delta_5d',
                           'TempAmplitude_5d',
                           'Tsnow_delta_2d',
                           'TminG_delta_3d',
                           'TempAmplitude_3d',
                           'TmaxG_delta_2d',
                           'HS_delta_5d',
                           'Tsnow_delta_3d',
                           'Precip_3d',
                           'TaG_delta_5d',
                           'Precip_1d',
                           'Tsnow_delta_1d']

    res_PR=evaluate_svm_with_feature_selection(mod1, Permutation_ranking)

    SHAP_16=['TaG_delta_5d',
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

    res_SHAP=evaluate_svm_with_feature_selection(mod1, SHAP_16)

    results_dict={
        'All features': res_FULL,
        'ANOVA': res_ANOVA,
        'BFE': res_BFE,
        'FFS': res_FFS,
        'RFE': res_RFE,
        'Permutation': res_PR,
        'SHAP': res_SHAP
    }

   # Step 1: Create DataFrame from results_dict
    results_df=pd.DataFrame.from_dict(results_dict, orient='index')

    # Step 2: Set correct column names (3 columns)
    results_df.columns=['Selected Features', 'Model', 'Metrics']

    # Step 3: Move method names from index to a column
    results_df=results_df.reset_index().rename(columns={'index': 'Method'})

    # Step 4: Expand 'Metrics' dictionary
    metrics_expanded=results_df['Metrics'].apply(pd.Series)

    # Step 5: Expand 'best_params' dictionary (inside metrics)
    best_params_expanded=metrics_expanded['best_params'].apply(pd.Series)

    # Step 6: Add feature count by counting the length of 'Selected Features'
    results_df['Feature Count']=results_df['Selected Features'].apply(len)

    # Step 7: Concatenate everything
    df_expanded=pd.concat([
        results_df.drop(columns=['Metrics', 'Model']),
        metrics_expanded.drop(columns=['best_params']),
        best_params_expanded
    ], axis=1)

    # Step 8: Rearrange if desired
    df_expanded=df_expanded[['Method', 'Selected Features', 'Feature Count',
                               'precision', 'recall', 'f1', 'accuracy', 'MCC', 'C', 'gamma']]

    # Step 9: Optional sort by MCC
    df_expanded=df_expanded.sort_values(
        by='MCC', ascending=False).reset_index(drop=True)

    # Final output
    print(df_expanded)

    outpath=results_path / 'df_expanded.csv'
    df_expanded.to_csv(outpath, index=False)
