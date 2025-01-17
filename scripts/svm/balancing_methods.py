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

    # .........................................................................
    #  TEST 1: HSnum vs HN_3d, full avalanche dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    filepath = common_path / 'mod1_newfeatures.csv'
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

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- TUNING SVM PARAMETERS ---

    # Tuning of parameter C and gamma for SVM classification

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
    n_neig = [1, 3, 5, 10, 25, 50]

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
                    'resampling_HSnum_HN_3d_long.csv')

    # .........................................................................
    #  TEST 2: all features, full avalanche dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    filepath = common_path / 'mod1_newfeatures.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- FEATURES SELECTION ---

    feature = [
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

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- TUNING SVM PARAMETERS ---

    # Tuning of parameter C and gamma for SVM classification
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
    n_neig = [1, 3, 5, 10, 25, 50]

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
                    'resampling_all_long.csv')

    # .........................................................................
    #  TEST 3: HSnum vs HN_3d, only medium-large avalanches dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures_NEW.csv'
    # filepath = common_path / 'mod1_newfeatures.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- FEATURES SELECTION ---
    feature = ['HN_3d', 'HSnum']

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- TUNING SVM PARAMETERS ---

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
    n_neig = [1, 3, 5, 10, 25, 50]

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
                    'resampling_HSnum_HN_3d_short.csv')

    # .........................................................................
    #  TEST 2: all features, full avalanche dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures_NEW.csv'
    # filepath = common_path / 'mod1_newfeatures.csv'
    results_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- FEATURES SELECTION ---

    feature = [
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

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- TUNING SVM PARAMETERS ---

    # Tuning of parameter C and gamma for SVM classification
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
    n_neig = [1, 3, 5, 10, 25, 50]

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
                    'resampling_all_short.csv')
