import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


if __name__ == '__main__':
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    filepath = common_path / 'mod1_newfeatures_NEW.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # ---------------------------------------------------------------
    # --- a) FEATURE SELECTION BASED ON PERMUTATION RANKING       ---
    # ---------------------------------------------------------------

    feature = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason',
        'HS_delta_1d', 'HS_delta_2d', 'HS_delta_3d', 'HS_delta_5d',
        'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d',
        'Penetration_ratio',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d',
        'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'ConsecWetSnowDays',
        'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    ]

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # features_low_variance = remove_low_variance(X, threshold=0.1)
    # X = X.drop(columns=features_low_variance)

    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # Tuning of parameter C and gamma for SVM classification
    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #     'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    # }

    # param_grid = {
    #     'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    #     'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # }

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
                  10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }

    # X_nm, y_nm = undersampling_nearmiss(X_new, y, version=3, n_neighbors=10)
    X_nm, y_nm = undersampling_clustercentroids(X_new, y)

    features_low_variance = remove_low_variance(X_nm, threshold=0.1)
    X_nm = X_nm.drop(columns=features_low_variance)

    X_train, X_test, y_train, y_test = train_test_split(
        X_nm, y_nm, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(
        X_test), columns=X_test.columns, index=X_test.index)

    res_tuning = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, param_grid,
        resampling_method=f'ClusterCentroids')

    classifier = train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, res_tuning['best_params'])

    # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    feature_importance_df = permutation_ranking(
        classifier[0], X_test, y_test)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\Permutation_ranking\\')

    save_outputfile(feature_importance_df, results_path /
                    'permutation_ranking_cluster_centroids.csv')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TEST ON PERFORMANCE OF MODEL WITH ONLY FEATURES WITH POSITIVE IMPORTANCE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Filter the DataFrame to include only positive importance values
    positive_features = feature_importance_df[feature_importance_df['Importance_Mean'] > 0]

    # ordered_features = feature_importance_df['Feature'].iloc[::-1]

    # # Get only the feature names
    features_plus_aval = positive_features['Feature'].tolist() + ['AvalDay']
    # features_plus_aval = ordered_features.tolist() + ['AvalDay']

    # # --- NEW SVM MODEL WITH FEATURES SELECTED ---

    # # mod1_clean = mod1.dropna()  # Keep the clean DataFrame with the datetime index
    # # X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    X_new = X[positive_features['Feature'].tolist()]
    y_new = y

    # --- SPLIT TRAIN AND TEST ---

    # ... 4. Condensed Nearest Neighbour Undersampling ...
    resampling_method = 'NearMiss3'
    X_resampled, y_resampled = undersampling_nearmiss(
        X_new, y_new, version=3, n_neighbors=10)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)
    res_new = tune_train_evaluate_svm(
        X_train_new, y_train_new, X_test_new, y_test_new, param_grid,
        resampling_method='NearMiss3 Undersampling')

    classifier_new = train_evaluate_final_svm(
        X_train_new, y_train_new, X_test_new, y_test_new, res_new['best_params'])

    # ---------------------------------------------------------------
    # --- b) TEST DIFFERENT CONFIGURATION OF FEATURES  ---
    # ---------------------------------------------------------------

    candidate_features = [
        'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
        'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    ]

    # Base predictors
    # base_predictors = ['HSnum']

    # Initialize results dictionary
    results = {}

    # Loop through each candidate feature and test its performance
    for feature in candidate_features:
        # Define the current set of features to evaluate
        # current_features = base_predictors + [feature]
        current_features = [feature]
        print(current_features)

        # Evaluate the model with the selected features
        result = evaluate_svm_with_feature_selection(mod1, current_features)

        # Store the result in the dictionary
        results[feature] = result

        # Print the evaluated feature and the result
        # print(f"Evaluated Feature: {feature}, Result: {result}")

    # Identify the best-performing feature based on the evaluation metric
    # Assuming higher is better; adjust based on metric
    # Extract the feature with the maximum precision
    best_feature = max(
        results, key=lambda x: results[x][2]['recall'])
    max_value = results[best_feature][2]['recall']

    print(
        f"Best Feature: {best_feature}, Best Result: {max_value}")

    data = []
    for key, (feature, model, metrics) in results.items():
        row = {'model': model, 'name': key}
        row.update(metrics)  # Merge the performance metrics
        data.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by='recall', ascending=False)

    save_outputfile(df, common_path / 'precision_features_NEW.csv')

    # ---------------------------------------------------------------
    # --- c) FEATURE SELECTION USING SELECT K BEST AND ANOVA      ---
    # ---------------------------------------------------------------

    # candidate_features = ['HSnum', 'HN_3d']
    # List of candidate features
    candidate_features = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
    ]

    # Data preparation
    feature_plus = candidate_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[candidate_features]
    y = mod1_clean['AvalDay']

    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # Create a range for k from 1 to num_columns (inclusive)
    k_range = list(range(1, X_new.shape[1]+1))
    # k_range = [1, 2, 3, 5, 7, 10, 15, 20, 25]
    results = []
    # Tuning of parameter C and gamma for SVM classification
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
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

    for k in k_range:
        # Select the top k features using SelectKBest
        features_selected = select_k_best(X_new, y, k=k)

        print(f'k = {k}, Features Selected: {features_selected}')

        X_selected = X_new[features_selected]
        # Apply random undersampling
        # X_resampled, y_resampled = undersampling_nearmiss(
        #     X_selected, y, version=3, n_neighbors=10)
        X_resampled, y_resampled = undersampling_clustercentroids(
            X_selected, y)

        # Remove correlated features and with low variance
        features_low_variance = remove_low_variance(X_resampled)
        X_resampled = X_resampled.drop(columns=features_low_variance)

        # Split into training and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.25, random_state=42)

        # Normalization
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
            'best_params': evaluation_metrics_SVM['best_params']
        })

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\ANOVA\\')

    save_outputfile(results_df, results_path /
                    'anova_feature_selection_new.csv')

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each metric
    for metric in ['precision', 'accuracy', 'recall', 'f1']:
        plt.plot(results_df['num_features'],
                 results_df[metric], marker='o', label=metric)

    # Add labels, title, and legend
    plt.title('Metrics Comparison Across Experiments', fontsize=14)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(results_df['num_features'])
    plt.legend(title='Metrics', fontsize=10)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # --- d) FEATURE SELECTION USING SHAP METHOD      ---
    # ---------------------------------------------------------------

    # candidate_features = ['HSnum', 'HN_3d']
    # List of candidate features
    candidate_features = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
    ]

    # Data preparation
    feature_plus = candidate_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[candidate_features]
    y = mod1_clean['AvalDay']

    features_correlated = remove_correlated_features(X, y)
    X = X.drop(columns=features_correlated)

    import shap

    # X_resampled, y_resampled = undersampling_nearmiss(
    #     X, y, version=3, n_neighbors=10)
    X_resampled, y_resampled = undersampling_clustercentroids(
        X, y)

    # Remove correlated features and with low variance
    features_low_variance = remove_low_variance(X_resampled)
    X_resampled = X_resampled.drop(columns=features_low_variance)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # Normalization
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        scaler.fit_transform(X_test), columns=X_test.columns)

    # param_grid = {
    #     'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    #     'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # }
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
                  10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    resampling_method = 'ClusterCentroids'

    res_svm = tune_train_evaluate_svm(X_train_scaled, y_train, X_test_scaled,
                                      y_test, param_grid, resampling_method, cv=5)    # Train SVM model

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
        f"SHAP values shape: {len(shap_values)}, Feature count: {X_train_scaled.shape[1]}")

    shap_values_df = pd.DataFrame(
        shap_values[:, :, 1], columns=X_train_scaled.columns)

    mean_shap_values = shap_values[:, :, 0].mean(axis=0)
    # Convert to pandas Series with feature names
    mean_shap_values = pd.Series(
        mean_shap_values, index=X_train_scaled.columns)

    # Sort SHAP values to make the plot clearer
    mean_shap_values_sorted = mean_shap_values.sort_values(ascending=False)

    pos_values = list(
        mean_shap_values_sorted[mean_shap_values_sorted > 0].items())
    # Extracting only the feature names
    pos_features = [feature for feature, _ in pos_values]

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 15))
    mean_shap_values_sorted.plot(kind='barh', color='skyblue')
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Features')
    plt.title('Feature Importance Based on SHAP Values')
    plt.show()

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\SHAP\\')

    save_outputfile(mean_shap_values_sorted, results_path /
                    'mean_shap_values.csv')

    shap_values_abs_df = shap_values_df.abs()
    save_outputfile(shap_values_abs_df, results_path /
                    'shap_values_abs.csv')

    save_outputfile(shap_values_df, results_path /
                    'shap_values.csv')

    mean_shap = shap_values_df.mean(axis=0).sort_values(ascending=False)

    import matplotlib.colors as mcolors

    # Create a custom diverging colormap (red, white, blue)
    cmap = mcolors.TwoSlopeNorm(vmin=shap_values_df.min(
    ).min(), vcenter=0, vmax=shap_values_df.max().max())
    # Red to Blue with white as the midpoint
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 15))

    # Create the heatmap with the custom color map
    sns.heatmap(shap_values_df.T, cmap='RdBu_r', annot=False,
                fmt='.2g', center=0, cbar_kws={'label': 'SHAP Value'})

    # Add title and labels
    plt.title('Heatmap of SHAP Values with Red-White-Blue Color Map')
    plt.xlabel('Samples')
    plt.ylabel('Features')

    # Show the plot
    plt.tight_layout()
    plt.show()

    #
    # ---------------------------------------------------------------
    # --- d) FEATURE SELECTION USING BACKWARD FEATURE ELIMINATION      ---
    # ---------------------------------------------------------------

    # candidate_features = ['HSnum', 'HN_3d']
    # List of candidate features
    candidate_features = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
    ]

    # Data preparation
    feature_plus = candidate_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[candidate_features]
    y = mod1_clean['AvalDay']

    # Remove correlated features
    features_correlated = remove_correlated_features(X, y)

    # features_to_remove_2 = remove_low_variance(X)

    X_new = X.drop(columns=features_correlated)

    # undersample = NearMiss(version=3, n_neighbors=10)

    # X_resampled, y_resampled = undersampling_nearmiss(
    #     X_new, y, version=3, n_neighbors=10)

    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'svc__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
    # Create a pipeline with undersampling and SVC
    pipeline = Pipeline([
        # ('undersample', NearMiss(version=3, n_neighbors=10)),  # Apply NearMiss
        ('undersample', ClusterCentroids(random_state=42)),  # Apply NearMiss
        ('svc', svm.SVC(kernel='rbf'))
    ])

    # Use GridSearchCV to tune hyperparameters during SFS
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=6
    )

    # Perform Sequential Feature Selection (SFS)
    sfs_BW = SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=(1, X_new.shape[1]),
        forward=False,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=6              # Use all available CPU cores
    )

    # Fit SFS to the data
    # sfs_BW.fit(X_resampled, y_resampled)
    sfs_BW.fit(X_new, y)

    # Retrieve the names of the selected features
    if isinstance(X_new, pd.DataFrame):
        selected_feature_names_BW = [X_new.columns[i]
                                     for i in sfs_BW.k_feature_idx_]
    else:
        selected_feature_names_BW = list(sfs_BW.k_feature_idx_)

    print("Selected Features:", selected_feature_names_BW)

    # Retrieve information about subsets
    subsets_BW = sfs_BW.subsets_
    subsets_BW_df = pd.DataFrame(subsets_BW)
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\BACKWARD_FEATURE_ELIMINATION\\')

    save_outputfile(subsets_BW_df, results_path /
                    'BFE_feature_selection_CC.csv')

    # Extract the best subset
    best_subset_BW = max(subsets_BW.items(), key=lambda x: x[1]['avg_score'])

    # Retrieve the indices and names of the best features
    best_feature_indices_BW = best_subset_BW[1]['feature_idx']
    if isinstance(X_new, pd.DataFrame):
        best_feature_names_BW = [X_new.columns[i]
                                 for i in best_feature_indices_BW]
    else:
        best_feature_names_BW = list(best_feature_indices_BW)

    # Print the results
    print(f"Best Feature Subset Size: {len(best_feature_names_BW)}")
    print(f"Best Features: {best_feature_names_BW}")
    print(f"Best Average Score (F1 Macro): {best_subset_BW[1]['avg_score']}")

    # Extract data for visualization
    subset_sizes_BW = [len(subset_BW['feature_idx'])
                       for subset_BW in subsets_BW.values()]
    avg_scores_BW = [subset_BW['avg_score']
                     for subset_BW in subsets_BW.values()]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes_BW, avg_scores_BW, marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Average F1 Macro Score")
    plt.title("Feature Subset Performance - Backward feature elimination")
    plt.grid(True)
    plt.show()

    # BestFeatures_BW_27 = ['N', 'TaG', 'HNnum', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_3d', 'HS_delta_5d', 'DaysSinceLastSnow', 'Tmin_2d', 'TempAmplitude_1d', 'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_2d',
    #                       'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'DegreeDays_Pos', 'Precip_1d', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_3d', 'Tsnow_delta_5d', 'ConsecWetSnowDays']

    # # NEW evaluation based on the best 6 feature (BASED ON PLOT)
    # # Perform Sequential Feature Selection (SFS)
    # sfs_BW_6 = SFS(
    #     estimator=grid_search,
    #     # k_features=10,          # Select the top 10 features
    #     # Explore all possible subset sizes
    #     k_features=6,
    #     forward=False,         # Forward selection
    #     floating=False,        # Disable floating step
    #     cv=5,                  # 5-fold cross-validation
    #     scoring='f1_macro',    # Use F1 macro as the scoring metric
    #     n_jobs=-1              # Use all available CPU cores
    # )

    # # Fit SFS to the data
    # # sfs_BW.fit(X_resampled, y_resampled)
    # sfs_BW_6.fit(X_new, y)

    # # Retrieve the names of the selected features
    # if isinstance(X_new, pd.DataFrame):
    #     selected_feature_names_BW_6 = [X_new.columns[i]
    #                                    for i in sfs_BW_6.k_feature_idx_]
    # else:
    #     selected_feature_names_BW_6 = list(sfs_BW_6.k_feature_idx_)

    # print("Selected Features:", selected_feature_names_BW_6)

    # BestFeatures_BW_6 = ['TaG', 'DayOfSeason', 'Tmin_2d',
    #                      'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_5d']

    # ---------------------------------------------------------------
    # --- e) FEATURE SELECTION USING FORWARD FEATURE ELIMINATION      ---
    # ---------------------------------------------------------------

    # candidate_features = ['HSnum', 'HN_3d']
    # List of candidate features
    candidate_features = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
    ]

    # Data preparation
    feature_plus = candidate_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[candidate_features]
    y = mod1_clean['AvalDay']

    # Remove correlated features
    features_correlated = remove_correlated_features(X, y)
    # features_to_remove_2 = remove_low_variance(X)
    # combined_list = features_to_remove + \
    #     features_to_remove_2  # Concatenate the two lists

    X_new = X.drop(columns=features_correlated)

    # X_resampled, y_resampled = undersampling_nearmiss(
    #     X_new, y, version=3, n_neighbors=10)

    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'svc__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }

    # Create a pipeline with SVC
    pipeline = Pipeline([
        # ('undersample', NearMiss(version=3, n_neighbors=10)),  # Apply NearMiss
        ('undersample', ClusterCentroids(random_state=42)),  # Apply NearMiss
        ('svc', svm.SVC(kernel='rbf'))
    ])

    # Use GridSearchCV to tune hyperparameters during SFS
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=5,
        n_jobs=-1
    )

    # Perform Sequential Feature Selection (SFS)
    sfs = SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=(1, X_new.shape[1]),
        # k_features=(1, 10),
        forward=True,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=-1              # Use all available CPU cores
    )

    # Fit SFS to the data
    sfs.fit(X_new, y)

    # Retrieve the names of the selected features
    if isinstance(X_new, pd.DataFrame):
        selected_feature_names_FW = [X_new.columns[i]
                                     for i in sfs.k_feature_idx_]
    else:
        selected_feature_names_FW = list(sfs.k_feature_idx_)

    print("Selected Features:", selected_feature_names_FW)

    # Retrieve information about subsets
    subsets_FW = sfs.subsets_
    subsets_FW_df = pd.DataFrame(subsets_FW)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\FORWARD_FEATURE_SELECTION\\')

    save_outputfile(subsets_FW_df, results_path /
                    'FFS_feature_selection_2.csv')

    # Extract the best subset
    best_subset_FW = max(subsets_FW.items(), key=lambda x: x[1]['avg_score'])

    # Retrieve the indices and names of the best features
    best_feature_indices_FW = best_subset_FW[1]['feature_idx']
    if isinstance(X_new, pd.DataFrame):
        best_feature_names_FW = [X_new.columns[i]
                                 for i in best_feature_indices_FW]
    else:
        best_feature_names_FW = list(best_feature_indices_FW)

    # Print the results
    print(f"Best Feature Subset Size: {len(best_feature_names_FW)}")
    print(f"Best Features: {best_feature_names_FW}")
    print(f"Best Average Score (F1 Macro): {best_subset_FW[1]['avg_score']}")

    # Extract data for visualization
    subset_sizes_FW = [len(subset_FW['feature_idx'])
                       for subset_FW in subsets_FW.values()]
    avg_scores_FW = [subset_FW['avg_score']
                     for subset_FW in subsets_FW.values()]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes_FW, avg_scores_FW, marker='o')
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Average F1 Macro Score")
    plt.title("Feature Subset Performance - Forward Selection")
    plt.grid(True)
    plt.show()

    # BestFeatures_FW_20 = ['N', 'V', 'HNnum', 'PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'Precip_1d', 'Precip_2d', 'Penetration_ratio',
    #                       'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex', 'MF_Crust_Present', 'New_MF_Crust']

    # sfs_FW_11 = SFS(
    #     estimator=grid_search,
    #     # k_features=10,          # Select the top 10 features
    #     # Explore all possible subset sizes
    #     k_features=11,
    #     forward=True,         # Forward selection
    #     floating=False,        # Disable floating step
    #     cv=5,                  # 5-fold cross-validation
    #     scoring='f1_macro',    # Use F1 macro as the scoring metric
    #     n_jobs=-1              # Use all available CPU cores
    # )

    # # Fit SFS to the data
    # # sfs_BW.fit(X_resampled, y_resampled)
    # sfs_FW_11.fit(X_new, y)

    # # Retrieve the names of the selected features
    # if isinstance(X_new, pd.DataFrame):
    #     selected_feature_names_FW_11 = [X_new.columns[i]
    #                                     for i in sfs_FW_11.k_feature_idx_]
    # else:
    #     selected_feature_names_FW_11 = list(sfs_FW_11.k_feature_idx_)

    # print("Selected Features:", selected_feature_names_FW_11)

    # BestFeatures_FW_11 = ['PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'WetSnow_Temperature',
    #                       'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex']

    # ---------------------------------------------------------------
    # --- e) RECURSIVE FEATURE EXTRACTION: RFE  ---
    # ---------------------------------------------------------------
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import ClusterCentroids
    from sklearn.svm import SVC, LinearSVC
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
    import numpy as np

    # List of candidate features
    candidate_features = [
        'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
        'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
        'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
        'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
    ]

    # Data Preparation
    feature_plus = candidate_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[candidate_features]
    y = mod1_clean['AvalDay']

    # Remove correlated features
    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.25, random_state=42
    )

    # Step 1: Hyperparameter Tuning for RBF Kernel SVM
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
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Step 2: Define a LinearSVC for RFECV
    linear_svc = LinearSVC(random_state=42, max_iter=10000)

    # Step 3: Define the final pipeline
    pipeline = Pipeline([
        ('undersample', ClusterCentroids(random_state=42)),  # Undersampling
        ('scaler', MinMaxScaler()),                          # Scaling
        ('feature_selection', RFECV(
            estimator=linear_svc,                            # Use LinearSVC for RFE
            step=1,                                          # Remove one feature at a time
            # 5-fold CV for RFE
            cv=StratifiedKFold(n_splits=5),
            scoring='f1_macro',                              # Scoring metric
            n_jobs=-1                                        # Use all CPU cores
        )),
        ('svc', SVC(
            kernel='rbf',                                    # Final RBF SVM
            C=best_params['C'],
            gamma=best_params['gamma'],
            random_state=42
        ))
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Evaluate the final pipeline
    y_pred = pipeline.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))

    # Access Selected Features
    rfecv = pipeline.named_steps['feature_selection']

    # Get mean cross-validation scores for each number of features
    mean_cv_scores = rfecv.cv_results_['mean_test_score']
    std_cv_scores = rfecv.cv_results_['std_test_score']

    # Plot the cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mean_cv_scores) + 1), mean_cv_scores, marker='o')
    plt.title('RFECV - Feature Selection Process')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Cross-Validation Score (F1 Macro)')
    plt.grid(True)
    plt.show()

    print("Selected Features:", np.array(X_new.columns)[rfecv.support_])
    print("Feature Rankings:", rfecv.ranking_)

    ranking = rfecv.ranking_
    top_11_features_indices = np.argsort(ranking)[:11]
    top_11_features = np.array(X_new.columns)[top_11_features_indices]

    top_23_features_indices = np.argsort(ranking)[:23]
    top_23_features = np.array(X_new.columns)[top_23_features_indices]

    print("Top 23 Features by Ranking:", top_23_features)
    print("Top 11 Features by Ranking:", top_11_features)

    # Get the selected features (those marked as True in support_)
    selected_features = np.array(X_new.columns)

    # Create a DataFrame with selected features and their ranking
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Ranking': ranking
    })

    # Sort the DataFrame by ranking (ascending order as lower rank means higher importance)
    feature_importance_sorted = feature_importance.sort_values(by='Ranking')

    # Display the sorted features by their importance ranking
    print("Selected Features ordered by importance:")
    print(feature_importance_sorted)

    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\RECURSIVE_FEATURE_ELIMINATION\\')

    save_outputfile(feature_importance_sorted, results_path /
                    'RFE_feature_selection_CC.csv')

    # ---------------------------------------------------------------
    # --- D) FEATURE EXTRACTION USING LINEAR DISCRIMINANT ANALYSIS (LDA)
    #        on SELECTED FEATURES ---
    # ---------------------------------------------------------------

    SHAP = ['TaG_delta_5d',
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

    # best_features = list(set(BestFeatures_FW_20 + BestFeatures_BW_27))

    # Data preparation
    feature_plus = SHAP + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[SHAP]
    y = mod1_clean['AvalDay']

    X_resampled, y_resampled = undersampling_clustercentroids(X, y)

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

    param_grid = {
        'C': [0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
            0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500,
            750, 1000],
        'gamma': [100, 75, 50, 30, 20, 15, 10, 7.5, 5, 3, 2, 1.5, 1,
            0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008,
            0.007, 0.005, 0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0007, 0.0005, 0.0003, 0.0002,
            0.00015, 0.0001]
    }

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # **Applicazione di LDA per Feature Extraction**
    # Ridurre a 1 dimensione, ad esempio
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index)

    X_train_lda = pd.DataFrame(X_train_lda, columns=['LDA'])
    X_test_lda = pd.DataFrame(X_test_lda, columns=['LDA'])

    result_SVM = tune_train_evaluate_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, param_grid, resampling_method='Cluster Centroids')

    classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, result_SVM['best_params'])

    # --add loop
    C_values = np.logspace(0, 3, num=1000)  # 50 values from 0.1 to 1000
    C_values = np.round(C_values, 5)  # Round for better readability

    gamma_values = np.linspace(0.002, 0.01, 41)  # 10 values in this range

    # Store results
    performance_results = []

    # Loop through different C values
    for C in C_values:
        for gamma in gamma_values:
            # Use the best found gamma, only varying C
            params = {'C': C, 'gamma': gamma}
            # params = {'C': C, 'gamma': result_SVM['best_params']['gamma']}

            # print(f"Training SVM with C = {C} and gamma = {params['gamma']}")

            # Train and evaluate the model with the current C value
            classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
                X_train_scaled, y_train, X_test_scaled, y_test, params
            )

            # Store results
            performance_results.append({
                'C': C,
                'gamma': gamma,
                'accuracy': evaluation_metrics_SVM['accuracy'],
                'precision': evaluation_metrics_SVM['precision'],
                'recall': evaluation_metrics_SVM['recall'],
                'f1': evaluation_metrics_SVM['f1']
            })

    # Convert results into a DataFrame for easy analysis
    df_performance = pd.DataFrame(performance_results)

    # Sort by C value
    df_performance = df_performance.sort_values(by='C')

    # Display results
    print(df_performance)

    shap_regular_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\04_SVM\\01_FEATURE_SELECTION\\SHAP_regularized\\svm_performance_results.csv')

    df_performance.to_csv(shap_regular_path, index=False)

    # # Optional: Plot performance trends
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 5))
    # for metric in ['accuracy', 'precision', 'recall', 'f1']:
    #     plt.plot(df_performance['C'],
    #              df_performance[metric], marker='o', label=metric)

    # plt.xscale('log')  # Use logarithmic scale for better visualization
    # plt.xlabel("C value (log scale)")
    # plt.ylabel("Performance")
    # plt.title("SVM Performance for Different C Values")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Group by C and compute statistics
    df_grouped = df_performance.groupby("C")["f1"].agg([
        ("mean", "mean"),
        ("10th", lambda x: np.percentile(x, 10)),
        ("25th", lambda x: np.percentile(x, 25)),
        ("50th", lambda x: np.percentile(x, 50)),  # Median
        ("75th", lambda x: np.percentile(x, 75)),
        ("90th", lambda x: np.percentile(x, 90)),
        ("min", "min"),
        ("max", "max")
    ]).reset_index()

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Plot median line
    plt.plot(df_grouped["C"], df_grouped["50th"], color='#1565C0',
             linestyle='-', linewidth=2, label="Median (50%)")

    # Add shading for percentiles
    plt.fill_between(df_grouped["C"], df_grouped["min"], df_grouped["max"],
                     color='#B3E5FC', alpha=0.5, label="min-max")
    plt.fill_between(df_grouped["C"], df_grouped["10th"], df_grouped["90th"],
                     color='#81D4FA', alpha=0.75, label="10th-90th Percentile")
    plt.fill_between(df_grouped["C"], df_grouped["25th"], df_grouped["75th"],
                     color='#4FC3F7', alpha=1, label="25th-75th Percentile")

    # Log scale for C
    plt.xscale("log")
    plt.xlabel("C value (log scale)")
    plt.ylabel("F1-score")
    plt.title("SVM Performance (C vs F1-score with Percentile Ranges)")

    # Add legend
    plt.legend(loc="upper left")

    # Add grid
    plt.grid(True, linestyle='dotted')

    # Show the plot
    plt.show()

    # -- end loop

    params = {'C': 53.7, 'gamma': 0.0058}
    # params = {'C': 50, 'gamma': 0.01}

    # print(f"Training SVM with C = {C} and gamma = {params['gamma']}")

    # Train and evaluate the model with the current C value
    classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, params
    )

    # Evaluate model with selected features
    result_LDA = tune_train_evaluate_svm(
        X_train_lda, y_train, X_test_lda, y_test, param_grid, resampling_method='Cluster Centroids')

    classifier_LDA, evaluation_metrics_LDA = train_evaluate_final_svm(
        X_train_lda, y_train, X_test_lda, y_test, result_LDA['best_params'])

    # -------------------------------------------------------
    # TEST FEATURES PERFORMANCE
    # -------------------------------------------------------

    ANOVA = ['HSnum', 'TH01G', 'DayOfSeason', 'HS_delta_1d', 'Tmin_2d', 'TaG_delta_5d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
        'T_mean', 'DegreeDays_Pos', 'Precip_2d', 'Precip_3d', 'Precip_5d', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'SnowConditionIndex']
    res_ANOVA = evaluate_svm_with_feature_selection(mod1, ANOVA)

# BFE = ['TaG', 'DayOfSeason', 'TempAmplitude_2d', 'TmaxG_delta_5d', 'TH30_tanh']
# res_BFE = evaluate_svm_with_feature_selection(mod1, BFE)

BFE = ['TaG', 'HNnum', 'TH01G', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_3d',
       'HS_delta_5d', 'DaysSinceLastSnow', 'TempAmplitude_1d',
       'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d',
       'TaG_delta_1d', 'TaG_delta_2d', 'TaG_delta_3d', 'TaG_delta_5d',
       'TminG_delta_1d', 'TminG_delta_2d', 'TminG_delta_3d', 'TminG_delta_5d',
       'TmaxG_delta_1d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
       'T_mean', 'DegreeDays_Pos', 'Precip_1d', 'WetSnow_CS', 'TH10_tanh',
       'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
       'Tsnow_delta_5d', 'ConsecWetSnowDays', 'ConsecCrustDays']
res_BFE = evaluate_svm_with_feature_selection(mod1, BFE)


# FFS = ['HS_delta_3d', 'TaG_delta_3d', 'DegreeDays_Pos', 'Tsnow_delta_3d']
# res_FFS = evaluate_svm_with_feature_selection(mod1, FFS)

FFS = ['HS_delta_3d', 'HS_delta_5d', 'DaysSinceLastSnow', 'TempAmplitude_1d',
       'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d', 'TaG_delta_3d',
       'TaG_delta_5d', 'TminG_delta_2d', 'TminG_delta_3d', 'TmaxG_delta_3d',
       'TmaxG_delta_5d', 'DegreeDays_Pos', 'Precip_3d', 'Precip_5d',
       'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh',
       'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d', 'Tsnow_delta_5d',
       'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays']
res_FFS = evaluate_svm_with_feature_selection(mod1, FFS)

RFE = ['TaG', 'New_MF_Crust', 'TaG_delta_5d', 'TH10_tanh', 'TH30_tanh',
        'TmaxG_delta_2d', 'TempAmplitude_5d', 'TempAmplitude_3d', 'Tsnow_delta_1d',
        'TempAmplitude_1d', 'TminG_delta_5d', 'TmaxG_delta_3d', 'HS_delta_5d',
        'HS_delta_3d', 'Tsnow_delta_3d', 'HS_delta_1d', 'DayOfSeason', 'PR',
        'Penetration_ratio', 'Precip_2d', 'HSnum', 'WetSnow_Temperature',
        'TminG_delta_3d']
res_RFE = evaluate_svm_with_feature_selection(mod1, RFE)

Permutation_ranking = ['AvalDay_2d',
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

res_PR = evaluate_svm_with_feature_selection(mod1, Permutation_ranking)

SHAP = ['TaG_delta_5d',
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

res_SHAP = evaluate_svm_with_feature_selection(mod1, SHAP)

physics = ['TaG_delta_5d',
        'TminG_delta_3d',
        'HS_delta_5d',
        'Precip_3d',
        'Precip_2d',
        'TempGrad_HS',
        'Tsnow_delta_3d', 'DayOfSeason']
res_physics = evaluate_svm_with_feature_selection(mod1, physics)


results_features = [res_ANOVA, res_BFE, res_FFS,
    res_RFE, res_PR, res_SHAP, res_physics]
label_feature = ['ANOVA', 'BFE', 'FFS', 'RFE', 'PR', 'SHAP', 'physics']
# Extract the metrics and create a DataFrame
data_res = []
for i, res in enumerate(results_features, 1):
     feature_set = ', '.join(res[0])  # Combine feature names as a string
     metrics = res[2]
     data_res.append({
         # 'Configuration': f"res{i}: {feature_set}",
         'Configuration': f"{label_feature[i-1]}",
         'Features': f"{feature_set}",
         'Precision': metrics['precision'],
         'Accuracy': metrics['accuracy'],
         'Recall': metrics['recall'],
         'F1': metrics['f1']
     })

# Create the DataFrame
df_res = pd.DataFrame(data_res)

# Plotting a line plot for precision
plt.figure(figsize=(8, 5))
plt.plot(df_res['Configuration'], df_res['F1'],
         marker='x', linestyle='-.', color='red', label='F1 Score')
plt.plot(df_res['Configuration'], df_res['Recall'],
         marker='s', linestyle='--', color='green', label='Recall')
plt.plot(df_res['Configuration'], df_res['Accuracy'],
         marker='d', linestyle=':', label='Accuracy')
plt.plot(df_res['Configuration'], df_res['Recall'],
         marker='o', linestyle='-', label='Recall')
plt.title('Scores for Snowfall features in different configuration')
plt.xlabel('Feature Configuration')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

df_res = df_res.sort_values(by='Recall', ascending=False)


# ....... 1. SNOW LOAD DUE SNOWFALL ...........................
[
    'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
    'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
    'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
    'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
    'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
    'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
    'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
    'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
    'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
    'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
    'SnowDrift_1d', 'SnowDrift_2d', 'SnowDrift_3d', 'SnowDrift_5d',
    'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
    'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
    'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
    'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
    'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
    'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
]

BestFeatures_BW_27 = ['N', 'TaG', 'HNnum', 'DayOfSeason',
                      'HS_delta_1d', 'HS_delta_3d', 'HS_delta_5d',
                      'DaysSinceLastSnow',
                      'Tmin_2d', 'TempAmplitude_1d', 'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d',
                      'TaG_delta_2d', 'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d',
                      'DegreeDays_Pos', 'Precip_1d', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_3d', 'Tsnow_delta_5d', 'ConsecWetSnowDays']
BestFeatures_BW_6 = ['TaG', 'DayOfSeason', 'Tmin_2d',
                     'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_5d']

BestFeatures_FW_20 = ['N', 'V', 'HNnum', 'PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'Precip_1d', 'Precip_2d', 'Penetration_ratio',
                      'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex', 'MF_Crust_Present', 'New_MF_Crust']
BestFeatures_FW_11 = ['PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'WetSnow_Temperature',
                      'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex']

Best_RFE = ['HSnum', 'TH03G', 'HS_delta_1d', 'HS_delta_2d', 'HN_2d', 'Tmax_2d',
            'Tmin_5d', 'TempAmplitude_1d', 'TaG_delta_1d', 'TaG_delta_5d',
            'TminG_delta_3d', 'TmaxG_delta_1d', 'TmaxG_delta_2d', 'Precip_2d',
            'Tsnow_delta_2d']

BestFeatures_Combined = list(set(BestFeatures_BW_6 + BestFeatures_FW_11))

print(BestFeatures_Combined)  # s0 = ['HS_delta_2d', 'TaG_delta_2d']

# resBW27 = evaluate_svm_with_feature_selection(mod1, BestFeatures_BW_27)
resBW6 = evaluate_svm_with_feature_selection(mod1, BestFeatures_BW_6)

# resFW20 = evaluate_svm_with_feature_selection(mod1, BestFeatures_FW_20)
resFW11 = evaluate_svm_with_feature_selection(mod1, BestFeatures_FW_11)

resComb = evaluate_svm_with_feature_selection(mod1, BestFeatures_Combined)

resRFE = evaluate_svm_with_feature_selection(mod1, Best_RFE)

# --- Test Pairwise combination with features selected in BW_6 ----

s1 = ['TaG', 'DayOfSeason']
res1 = evaluate_svm_with_feature_selection(mod1, s1)

s2 = ['TaG', 'Tmin_2d']
res2 = evaluate_svm_with_feature_selection(mod1, s2)

s3 = ['TaG', 'TaG_delta_3d']
res3 = evaluate_svm_with_feature_selection(mod1, s3)

s4 = ['TaG', 'TaG_delta_5d']
res4 = evaluate_svm_with_feature_selection(mod1, s4)

s5 = ['TaG', 'TminG_delta_5d']
res5 = evaluate_svm_with_feature_selection(mod1, s5)

s6 = ['DayOfSeason', 'Tmin_2d']
res6 = evaluate_svm_with_feature_selection(mod1, s6)

s7 = ['DayOfSeason', 'TaG_delta_3d']
res7 = evaluate_svm_with_feature_selection(mod1, s7)

s8 = ['DayOfSeason', 'TaG_delta_5d']
res8 = evaluate_svm_with_feature_selection(mod1, s8)

s9 = ['DayOfSeason', 'TminG_delta_5d']
res9 = evaluate_svm_with_feature_selection(mod1, s9)

s10 = ['Tmin_2d', 'TaG_delta_3d']
res10 = evaluate_svm_with_feature_selection(mod1, s10)

s11 = ['Tmin_2d', 'TaG_delta_5d']
res11 = evaluate_svm_with_feature_selection(mod1, s11)

s12 = ['Tmin_2d', 'TminG_delta_5d']
res12 = evaluate_svm_with_feature_selection(mod1, s12)

s13 = ['TaG_delta_3d', 'TaG_delta_5d']
res13 = evaluate_svm_with_feature_selection(mod1, s13)

s14 = ['TaG_delta_3d', 'TminG_delta_5d']
res14 = evaluate_svm_with_feature_selection(mod1, s14)

s15 = ['TaG_delta_5d', 'TminG_delta_5d']
res15 = evaluate_svm_with_feature_selection(mod1, s15)

 results_features = [res1, res2, res3, res4, res5,
                      res6, res7, res8, res9, res10,
                      res11, res12, res13, res14, res15]

  # Extract the metrics and create a DataFrame
  data_res = []
   for i, res in enumerate(results_features, 1):
        feature_set = ', '.join(res[0])  # Combine feature names as a string
        metrics = res[2]
        data_res.append({
            # 'Configuration': f"res{i}: {feature_set}",
            'Configuration': f"conf.{i}",
            'Features': f"{feature_set}",
            'Precision': metrics['precision'],
            'Accuracy': metrics['accuracy'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })

    # Create the DataFrame
    df_res= pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res['Configuration'], df_res['F1'],
             marker='x', linestyle='-.', color='red', label='F1 Score')
    plt.plot(df_res['Configuration'], df_res['Recall'],
             marker='s', linestyle='--', color='green', label='Recall')
    plt.plot(df_res['Configuration'], df_res['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res['Configuration'], df_res['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Snowfall features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    df_res= df_res.sort_values(by='Recall', ascending=False)
    # save_outputfile(df_res, common_path / 'config_snowload_features.csv')
