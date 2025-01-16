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


if __name__ == '__main__':
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures_NEW.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # output_filepath = common_path / 'mod1_undersampling.csv'
    # output_filepath2 = common_path / 'mod1_oversampling.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- FEATURES SELECTION ---
    feature = ['HN_3d', 'HSnum']

    # feature = [
    #     'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
    #     'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
    #     'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
    #     'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
    #     'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
    #     'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
    #     'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
    #     'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
    #     'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
    #     'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
    #     'SnowDrift_1d', 'SnowDrift_2d', 'SnowDrift_3d', 'SnowDrift_5d',
    #     'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
    #     'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
    #     'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
    #     'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
    #     'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
    #     'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    # ]

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- TUNING SVM PARAMETERS ---

    # Tuning of parameter C and gamma for SVM classification
    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #     'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    # }

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
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

    # --- UNDERSAMPLING ---

    # ... 1. Random undersampling ...

    X_rand, y_rand = undersampling_random(X, y)

    X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
        X_rand, y_rand, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_rand_train = pd.DataFrame(scaler.fit_transform(
        X_rand_train), columns=X_rand_train.columns, index=X_rand_train.index)
    X_rand_test = pd.DataFrame(scaler.transform(
        X_rand_test), columns=X_rand_test.columns, index=X_rand_test.index)

    res_rand = tune_train_evaluate_svm(
        X_rand_train, y_rand_train, X_rand_test, y_rand_test, param_grid,
        resampling_method='Random undersampling')

    # ... 2. Random undersampling N days before ...
    Ndays = 10
    X_rand_10d, y_rand_10d = undersampling_random_timelimited(
        X, y, Ndays=Ndays)

    X_rand_10d_train, X_rand_10d_test, y_rand_10d_train, y_rand_10d_test = train_test_split(
        X_rand_10d, y_rand_10d, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_rand_10d_train = pd.DataFrame(scaler.fit_transform(
        X_rand_10d_train), columns=X_rand_10d_train.columns, index=X_rand_10d_train.index)
    X_rand_10d_test = pd.DataFrame(scaler.transform(
        X_rand_10d_test), columns=X_rand_10d_test.columns, index=X_rand_10d_test.index)

    res_rand_10d = tune_train_evaluate_svm(
        X_rand_10d_train, y_rand_10d_train, X_rand_10d_test, y_rand_10d_test, param_grid,
        resampling_method=f'Random undersampling {Ndays} days before')

    # ... 3. Nearmiss undersampling ...

    # vers = [1, 2, 3]
    vers = [3]
    n_neig = [1, 3, 5, 10, 50]

    # List to store results
    res_list = []
    for v in vers:
        for n in n_neig:
            X_nm, y_nm = undersampling_nearmiss(X, y, version=v, n_neighbors=n)

            X_nm_train, X_nm_test, y_nm_train, y_nm_test = train_test_split(
                X_nm, y_nm, test_size=0.25, random_state=42)

            scaler = MinMaxScaler()
            X_nm_train = pd.DataFrame(scaler.fit_transform(
                X_nm_train), columns=X_nm_train.columns, index=X_nm_train.index)
            X_nm_test = pd.DataFrame(scaler.transform(
                X_nm_test), columns=X_nm_test.columns, index=X_nm_test.index)

            res_nm = tune_train_evaluate_svm(
                X_nm_train, y_nm_train, X_nm_test, y_nm_test, param_grid,
                resampling_method=f'NearMiss_v{v}_nn{n}')

            res_list.append(
                {'sampling_method': f'NearMiss_v{v}_nn{n}', **res_nm})

    # ... 4. Condensed Nearest Neighbour Undersampling ...

    X_cnn, y_cnn = undersampling_cnn(X, y)

    X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = train_test_split(
        X_cnn, y_cnn, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_cnn_train = pd.DataFrame(scaler.fit_transform(
        X_cnn_train), columns=X_cnn_train.columns, index=X_cnn_train.index)
    X_cnn_test = pd.DataFrame(scaler.transform(
        X_cnn_test), columns=X_cnn_test.columns, index=X_cnn_test.index)

    res_cnn = tune_train_evaluate_svm(
        X_cnn_train, y_cnn_train, X_cnn_test, y_cnn_test, param_grid,
        resampling_method='Condensed Nearest Neighbour Undersampling')

    # ... 5. Edited Nearest Neighbour Undersampling ...

    X_enn, y_enn = undersampling_enn(X, y)

    X_enn_train, X_enn_test, y_enn_train, y_enn_test = train_test_split(
        X_enn, y_enn, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_enn_train = pd.DataFrame(scaler.fit_transform(
        X_enn_train), columns=X_enn_train.columns, index=X_enn_train.index)
    X_enn_test = pd.DataFrame(scaler.transform(
        X_enn_test), columns=X_enn_test.columns, index=X_enn_test.index)

    res_enn = tune_train_evaluate_svm(
        X_enn_train, y_enn_train, X_enn_test, y_enn_test, param_grid,
        resampling_method='Edited Nearest Neighbour Undersampling')

    # ... 6. Cluster Centroids Undersampling ...

    X_cc, y_cc = undersampling_clustercentroids(X, y)

    X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(
        X_cc, y_cc, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_cc_train = pd.DataFrame(scaler.fit_transform(
        X_cc_train), columns=X_cc_train.columns, index=X_cc_train.index)
    X_cc_test = pd.DataFrame(scaler.transform(
        X_cc_test), columns=X_cc_test.columns, index=X_cc_test.index)

    res_cc = tune_train_evaluate_svm(
        X_cc_train, y_cc_train, X_cc_test, y_cc_test, param_grid,
        resampling_method='Cluster Centroids Undersampling')

    # ... 7. Tomek Links Undersampling ...

    X_tl, y_tl = undersampling_tomeklinks(X, y)

    X_tl_train, X_tl_test, y_tl_train, y_tl_test = train_test_split(
        X_tl, y_tl, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_tl_train = pd.DataFrame(scaler.fit_transform(
        X_tl_train), columns=X_tl_train.columns, index=X_tl_train.index)
    X_tl_test = pd.DataFrame(scaler.transform(
        X_tl_test), columns=X_tl_test.columns, index=X_tl_test.index)

    res_tl = tune_train_evaluate_svm(
        X_tl_train, y_tl_train, X_tl_test, y_tl_test, param_grid,
        resampling_method='Tomek Links Undersampling')

    # --- OVERSAMPLING ---
    # --- SPLIT TRAIN AND TEST ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(
        X_test), columns=X_test.columns, index=X_test.index)

    print("Original class distribution:", Counter(y))
    print("Original class distribution training set:", Counter(y_train))
    print("Original class distribution test set:", Counter(y_test))

    # ... 1. Random oversampling ...

    X_ros, y_ros = oversampling_random(X_train, y_train)
    res_ros = tune_train_evaluate_svm(X_ros, y_ros, X_test, y_test, param_grid,
                                      resampling_method='Random oversampling')
    # ... 2. SMOTE oversampling ...

    X_sm, y_sm = oversampling_smote(X_train, y_train)
    res_sm = tune_train_evaluate_svm(X_sm, y_sm, X_test, y_test, param_grid,
                                     resampling_method='SMOTE oversampling')

    # ... 3. ADASYN oversampling ...

    X_adas, y_adas = oversampling_adasyn(X_train, y_train)
    res_adas = tune_train_evaluate_svm(
        X_adas, y_adas, X_test, y_test, param_grid,
        resampling_method='ADASYN oversampling')

    # ... 4. SVMSMOTE oversampling ...

    X_svmsm, y_svmsm = oversampling_svmsmote(X_train, y_train)
    res_svmsm = tune_train_evaluate_svm(
        X_svmsm, y_svmsm, X_test, y_test, param_grid,
        resampling_method='SVMSMOTE oversampling')

    # --- STORE RESULTS IN A DATAFRAME ---

    # List to store results
    results_list = []
    final_results_list = []

    # Add each result to the list with the sampling method as an identifier
    results_list.append(
        {'sampling_method': 'Random_Undersampling', **res_rand})
    results_list.append(
        {'sampling_method': 'Random_Undersampling_10d', **res_rand_10d})

    for entry in results_list:
        if isinstance(entry, list):  # Check if an entry is a nested list (like res_list)
            # Add all elements of the list to the final results
            final_results_list.extend(entry)
        else:
            # Add single dictionary entries directly
            final_results_list.append(entry)

    # Add res_list directly (in case it hasn't been already added to results_list)
    final_results_list.extend(res_list)

    final_results_list.append(
        {'sampling_method': 'CNN_Undersampling', **res_cnn})
    final_results_list.append(
        {'sampling_method': 'ENN_Undersampling', **res_enn})
    final_results_list.append(
        {'sampling_method': 'ClusterCentroids_Undersampling', **res_cc})
    final_results_list.append(
        {'sampling_method': 'TomekLinks_Undersampling', **res_tl})

    final_results_list.append(
        {'sampling_method': 'Random_Oversampling', **res_ros})
    final_results_list.append(
        {'sampling_method': 'SMOTE_Oversampling', **res_sm})
    final_results_list.append(
        {'sampling_method': 'ADASYN_Oversampling', **res_adas})
    final_results_list.append(
        {'sampling_method': 'SVMSMOTE_Oversampling', **res_svmsm})

    # Convert list of dictionaries to a DataFrame
    results_df = pd.DataFrame(final_results_list)
    print(results_df)

    save_outputfile(results_df, results_path /
                    'under_oversampling_comparison_normalized_2var_NEW.csv')

    # ---------------------------------------------------------------
    # --- a) DEVELOP SVM FOR Nearmiss3 UNDERSAMPLING ---
    # ---------------------------------------------------------------
    # feature = [
    #     'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
    #     'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason',
    #     'HS_delta_1d', 'HS_delta_2d', 'HS_delta_3d', 'HS_delta_5d',
    #     'HN_2d', 'HN_3d', 'HN_5d',
    #     'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
    #     'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
    #     'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
    #     'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
    #     'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
    #     'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
    #     'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
    #     'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d',
    #     'FreshSWE', 'SeasonalSWE_cum',
    #     'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
    #     'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d',
    #     'Tsnow_delta_2d', 'Tsnow_delta_3d',
    #     'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
    #     'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
    #     'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    # ]
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

    features_low_variance = remove_low_variance(X, threshold=0.1)
    X = X.drop(columns=features_low_variance)

    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    }

    X_nm, y_nm = undersampling_nearmiss(X_new, y, version=3, n_neighbors=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X_nm, y_nm, test_size=0.25, random_state=42)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(
        X_test), columns=X_test.columns, index=X_test.index)

    res_tuning = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, param_grid,
        resampling_method=f'NearMiss_v3_nn10')

    classifier = train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, res_tuning['best_params'])

    # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    feature_importance_df = permutation_ranking(
        classifier[0], X_test, y_test)

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

    feature_importance_df = permutation_ranking(
        classifier_new[0], X_test_new, y_test_new)

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

    # Remove correlated features and with low variance
    features_low_variance = remove_low_variance(X)
    X = X.drop(columns=features_low_variance)

    features_correlated = remove_correlated_features(X, y)
    X_new = X.drop(columns=features_correlated)

    # Create a range for k from 1 to num_columns (inclusive)
    k_range = list(range(1, X_new.shape[1]+1))
    results = []
    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    }

    for k in k_range:
        # Select the top k features using SelectKBest
        features_selected = select_k_best(X_new, y, k=k)

        print(f'k = {k}, Features Selected: {features_selected}')

        X_selected = X_new[features_selected]
        # Apply random undersampling
        X_resampled, y_resampled = undersampling_nearmiss(
            X_selected, y, version=3, n_neighbors=10)

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
            X_train_scaled, y_train, X_test_scaled, y_test, param_grid, resampling_method='Nearmiss3')

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

    # Remove correlated features and with low variance
    features_low_variance = remove_low_variance(X)
    X = X.drop(columns=features_low_variance)

    features_correlated = remove_correlated_features(X, y)
    X = X.drop(columns=features_correlated)

    import shap

    X_resampled, y_resampled = undersampling_nearmiss(
        X, y, version=3, n_neighbors=10)

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # Normalization
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        scaler.fit_transform(X_test), columns=X_test.columns)

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    }
    resampling_method = 'Nearmiss3'

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
    mean_shap_values_sorted = mean_shap_values.sort_values(ascending=True)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 15))
    mean_shap_values_sorted.plot(kind='barh', color='skyblue')
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Features')
    plt.title('Feature Importance Based on SHAP Values')
    plt.show()

    shap_values_abs_df = shap_values_df.abs()

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
    features_to_remove = remove_correlated_features(X, y)
    features_to_remove_2 = remove_low_variance(X)
    combined_list = features_to_remove + \
        features_to_remove_2  # Concatenate the two lists

    X_new = X.drop(columns=combined_list)

    # undersample = NearMiss(version=3, n_neighbors=10)

    # X_resampled, y_resampled = undersampling_nearmiss(
    #     X_new, y, version=3, n_neighbors=10)

    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'svc__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }

    # Create a pipeline with undersampling and SVC
    pipeline = Pipeline([
        ('undersample', NearMiss(version=3, n_neighbors=10)),  # Apply NearMiss
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
    sfs_BW = SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=(1, X_new.shape[1]),
        forward=False,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=-1              # Use all available CPU cores
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

    BestFeatures_BW_27 = ['N', 'TaG', 'HNnum', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_3d', 'HS_delta_5d', 'DaysSinceLastSnow', 'Tmin_2d', 'TempAmplitude_1d', 'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_2d',
                          'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d', 'DegreeDays_Pos', 'Precip_1d', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_3d', 'Tsnow_delta_5d', 'ConsecWetSnowDays']

    # NEW evaluation based on the best 6 feature (BASED ON PLOT)
    # Perform Sequential Feature Selection (SFS)
    sfs_BW_6 = SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=6,
        forward=False,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=-1              # Use all available CPU cores
    )

    # Fit SFS to the data
    # sfs_BW.fit(X_resampled, y_resampled)
    sfs_BW_6.fit(X_new, y)

    # Retrieve the names of the selected features
    if isinstance(X_new, pd.DataFrame):
        selected_feature_names_BW_6 = [X_new.columns[i]
                                       for i in sfs_BW_6.k_feature_idx_]
    else:
        selected_feature_names_BW_6 = list(sfs_BW_6.k_feature_idx_)

    print("Selected Features:", selected_feature_names_BW_6)

    BestFeatures_BW_6 = ['TaG', 'DayOfSeason', 'Tmin_2d',
                         'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_5d']

    # ---------------------------------------------------------------
    # --- e) FEATURE SELECTION USING FORWARD FEATURE ELIMINATION      ---
    # ---------------------------------------------------------------

    # candidate_features = ['HSnum', 'HN_3d']
    # List of candidate features
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
    features_to_remove = remove_correlated_features(X, y)
    features_to_remove_2 = remove_low_variance(X)
    combined_list = features_to_remove + \
        features_to_remove_2  # Concatenate the two lists

    X_new = X.drop(columns=combined_list)

    # X_resampled, y_resampled = undersampling_nearmiss(
    #     X_new, y, version=3, n_neighbors=10)

    param_grid = {
        'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'svc__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    }

    # Create a pipeline with SVC
    pipeline = Pipeline([
        ('undersample', NearMiss(version=3, n_neighbors=10)),  # Apply NearMiss
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

    BestFeatures_FW_20 = ['N', 'V', 'HNnum', 'PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'Precip_1d', 'Precip_2d', 'Penetration_ratio',
                          'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex', 'MF_Crust_Present', 'New_MF_Crust']

    sfs_FW_11 = SFS(
        estimator=grid_search,
        # k_features=10,          # Select the top 10 features
        # Explore all possible subset sizes
        k_features=11,
        forward=True,         # Forward selection
        floating=False,        # Disable floating step
        cv=5,                  # 5-fold cross-validation
        scoring='f1_macro',    # Use F1 macro as the scoring metric
        n_jobs=-1              # Use all available CPU cores
    )

    # Fit SFS to the data
    # sfs_BW.fit(X_resampled, y_resampled)
    sfs_FW_11.fit(X_new, y)

    # Retrieve the names of the selected features
    if isinstance(X_new, pd.DataFrame):
        selected_feature_names_FW_11 = [X_new.columns[i]
                                        for i in sfs_FW_11.k_feature_idx_]
    else:
        selected_feature_names_FW_11 = list(sfs_FW_11.k_feature_idx_)

    print("Selected Features:", selected_feature_names_FW_11)

    BestFeatures_FW_11 = ['PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'WetSnow_Temperature',
                          'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex']
    # ---------------------------------------------------------------
    # --- D) FEATURE EXTRACTION USING LINEAR DISCRIMINANT ANALYSIS (LDA)
    #        on SELECTED FEATURES ---
    # ---------------------------------------------------------------

    BestFeatures_FW_11 = ['PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'WetSnow_Temperature',
                          'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex']

    BestFeatures_BW_6 = ['TaG', 'DayOfSeason', 'Tmin_2d',
                         'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_5d']

    BestFeatures_BW_27 = ['N', 'TaG', 'HNnum', 'DayOfSeason',
                          'HS_delta_1d', 'HS_delta_3d', 'HS_delta_5d',
                          'DaysSinceLastSnow',
                          'Tmin_2d', 'TempAmplitude_1d', 'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d',
                          'TaG_delta_2d', 'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_2d', 'TmaxG_delta_3d',
                          'DegreeDays_Pos', 'Precip_1d', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_3d', 'Tsnow_delta_5d', 'ConsecWetSnowDays']

    BestFeatures_FW_20 = ['N', 'V', 'HNnum', 'PR', 'DayOfSeason', 'HS_delta_3d', 'Tmin_2d', 'TmaxG_delta_3d', 'Precip_1d', 'Precip_2d', 'Penetration_ratio',
                          'WetSnow_CS', 'WetSnow_Temperature', 'TempGrad_HS', 'TH10_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_3d', 'SnowConditionIndex', 'MF_Crust_Present', 'New_MF_Crust']

    best_features = list(set(BestFeatures_FW_20 + BestFeatures_BW_27))

    # Data preparation
    feature_plus = best_features + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[best_features]
    y = mod1_clean['AvalDay']

    X_resampled, y_resampled = undersampling_nearmiss(
        X, y, version=3, n_neighbors=10)

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    }

    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    scaler = StandardScaler()
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
        X_train_scaled, y_train, X_test_scaled, y_test, param_grid, resampling_method='Nearmiss3')

    classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, result_SVM['best_params'])

    # Evaluate model with selected features
    result_LDA = tune_train_evaluate_svm(
        X_train_lda, y_train, X_test_lda, y_test, param_grid, resampling_method='Nearmiss3')

    classifier_LDA, evaluation_metrics_LDA = train_evaluate_final_svm(
        X_train_lda, y_train, X_test_lda, y_test, result_LDA['best_params'])

    # ---------------------------------------------------------------
    # --- e) RECURSIVE FEATURE EXTRACTION: RFE  ---
    # ---------------------------------------------------------------
    from sklearn.svm import SVC
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold

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

    X_resampled, y_resampled = undersampling_nearmiss(
        X, y, version=3, n_neighbors=10)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # Scale the data (important for SVM with RBF kernel)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the LinearSVC estimator
    linear_svm = LinearSVC(penalty="l1", dual=False, max_iter=10000)

    # Initialize RFECV with cross-validation (using Stratified K-Folds)
    selector = RFECV(estimator=linear_svm, step=1,
                     cv=StratifiedKFold(5), scoring='recall_macro')

    # Fit the selector to the data
    selector.fit(X_train_scaled, y_train)

    # Print the optimal number of features
    print(f"Optimal number of features: {selector.n_features_}")

    # Get the selected features
    selected_features = X.columns[selector.support_]
    print("Selected Features:", selected_features)

    # # Train and evaluate the model using the selected features
    # X_train_selected = X_train_scaled[:, selector.support_]
    # X_test_selected = X_test_scaled[:, selector.support_]

    # param_grid = {
    #     'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    #     'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # }
    # # Train a model with the selected features
    # best_params = cross_validate_svm(
    #     X_train_selected, y_train, param_grid, cv=5, title='CV scores', scoring='recall_macro')
    # # svm_rbf = SVC(kernel='rbf', gamma='scale', C=1)
    # svm_rbf = SVC(kernel='rbf', gamma=best_params['best_params']['gamma'],
    #               C=best_params['best_params']['C'])
    # svm_rbf.fit(X_train_selected, y_train)

    # y_pred = svm_rbf.predict(X_test_selected)
    # from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

    # # Evaluate the model
    # accuracy = svm_rbf.score(X_test_selected, y_test)
    # print(f"Model Accuracy with Selected Features: {accuracy}")
    # # Precision
    # precision = precision_score(y_test, y_pred)
    # print("Precision:", precision)

    # # Recall
    # recall = recall_score(y_test, y_pred)
    # print("Recall:", recall)

    # # F1 Score
    # f1 = f1_score(y_test, y_pred)
    # print("F1 Score:", f1)

    # # ROC AUC
    # roc_auc = roc_auc_score(y_test, y_pred)
    # print("ROC AUC:", roc_auc)

    # # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)

    # # Classification Report (summary of precision, recall, F1, support)
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # # Plot the cross-validation scores for each number of features
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
    #          selector.cv_results_['mean_test_score'])
    # plt.xlabel('Number of Features')
    # plt.ylabel('Cross-validation score (Accuracy)')
    # plt.title('RFECV - Cross-validation scores')
    # plt.show()


# .......................

    # Data preparation
    feature_plus = list(selected_features) + ['AvalDay']
    mod1_clean = mod1[feature_plus].dropna()
    X = mod1_clean[selected_features]
    y = mod1_clean['AvalDay']

    X_resampled, y_resampled = undersampling_nearmiss(
        X, y, version=3, n_neighbors=10)

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Scale the data (important for SVM with RBF kernel)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns)

    param_grid = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        'gamma': [100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    }
    result_SVM = tune_train_evaluate_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, param_grid, resampling_method='Nearmiss3')

    classifier_SVM, evaluation_metrics_SVM = train_evaluate_final_svm(
        X_train_scaled, y_train, X_test_scaled, y_test, result_SVM['best_params'])
    # -------------------------------------------------------
    # TEST FEATURES PERFORMANCE
    # -------------------------------------------------------

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
    # save_outputfile(df_res, common_path / 'config_snowload_features.csv')
