import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import (undersampling_random, undersampling_random_timelimited,
                                               undersampling_nearmiss, undersampling_cnn,
                                               undersampling_enn, undersampling_clustercentroids,
                                               undersampling_tomeklinks)
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                                    plot_roc_curve, permutation_ranking, evaluate_svm_with_feature_selection)
from scripts.svm.utils import save_outputfile, get_adjacent_values
from sklearn.preprocessing import MinMaxScaler, StandardScaler


if __name__ == '__main__':
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

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
    # feature = ['HSnum', 'HN_3d']
    # feature = [
    #     'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
    #     'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
    #     'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
    #     'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
    #     'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
    #     'TempAmplitude_3d', 'TempAmplitude_5d', 'Ta_delta_1d', 'Ta_delta_2d',
    #     'Ta_delta_3d', 'Ta_delta_5d', 'Tmin_delta_1d', 'Tmin_delta_2d',
    #     'Tmin_delta_3d', 'Tmin_delta_5d', 'Tmax_delta_1d', 'Tmax_delta_2d',
    #     'Tmax_delta_3d', 'Tmax_delta_5d', 'T_mean', 'DegreeDays_Pos',
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

    # --- SPLIT TRAIN AND TEST ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    print("Original class distribution:", Counter(y))
    print("Original class distribution training set:", Counter(y_train))
    print("Original class distribution test set:", Counter(y_test))

    # --- SCALING FEATURES (NORMALIZATION vs STANDARDIZATION) ---

    # NORMALIZATION
    norm = MinMaxScaler().fit(X)
    X_norm = pd.DataFrame(norm.transform(X), columns=X.columns, index=X.index)

    # # STANDARDIZATION
    # stand = StandardScaler().fit(X)
    # X_stand = pd.DataFrame(stand.transform(
    #     X), columns=X.columns, index=X.index)

    summary_X = X.describe()
    summary_X_norm = X_norm.describe()
    # summary_X_stand = X_stand.describe()

    print(summary_X)
    print(summary_X_norm)
    # print(summary_X_stand)

    # save_outputfile(summary_X, results_path / 'summary_X.csv')
    # save_outputfile(summary_X_norm, results_path / 'summary_X_norm.csv')
    # save_outputfile(summary_X_stand, results_path / 'summary_X_stand.csv')

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

    X = X_norm  # normalization

    # ... 1. Random undersampling ...

    X_rand, y_rand = undersampling_random(X, y)

    X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
        X_rand, y_rand, test_size=0.25, random_state=42)

    res_rand = tune_train_evaluate_svm(
        X_rand_train, y_rand_train, X_rand_test, y_rand_test, param_grid,
        resampling_method='Random undersampling')

    # ... 2. Random undersampling N days before ...
    Ndays = 10
    X_rand_10d, y_rand_10d = undersampling_random_timelimited(
        X, y, Ndays=Ndays)

    X_rand_10d_train, X_rand_10d_test, y_rand_10d_train, y_rand_10d_test = train_test_split(
        X_rand_10d, y_rand_10d, test_size=0.25, random_state=42)
    res_rand_10d = tune_train_evaluate_svm(
        X_rand_10d_train, y_rand_10d_train, X_rand_10d_test, y_rand_10d_test, param_grid,
        resampling_method=f'Random undersampling {Ndays} days before')

    # ... 3. Nearmiss undersampling ...

    vers = [1, 2, 3]
    n_neig = [1, 3, 5, 10]

    # List to store results
    res_list = []
    for v in vers:
        for n in n_neig:
            X_nm, y_nm = undersampling_nearmiss(X, y, version=v, n_neighbors=n)

            X_nm_train, X_nm_test, y_nm_train, y_nm_test = train_test_split(
                X_nm, y_nm, test_size=0.25, random_state=42)
            res_nm = tune_train_evaluate_svm(
                X_nm_train, y_nm_train, X_nm_test, y_nm_test, param_grid,
                resampling_method=f'NearMiss_v{v}_nn{n}')

            res_list.append(
                {'sampling_method': f'NearMiss_v{v}_nn{n}', **res_nm})

    # ... 4. Condensed Nearest Neighbour Undersampling ...

    X_cnn, y_cnn = undersampling_cnn(X, y)

    # norm = MinMaxScaler().fit(X)
    # X_norm = pd.DataFrame(norm.transform(
    #     X_cnn), columns=X_cnn.columns, index=X_cnn.index)

    X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = train_test_split(
        X_cnn, y_cnn, test_size=0.25, random_state=42)

    res_cnn = tune_train_evaluate_svm(
        X_cnn_train, y_cnn_train, X_cnn_test, y_cnn_test, param_grid,
        resampling_method='Condensed Nearest Neighbour Undersampling')

    # ... 5. Edited Nearest Neighbour Undersampling ...

    X_enn, y_enn = undersampling_enn(X, y)

    X_enn_train, X_enn_test, y_enn_train, y_enn_test = train_test_split(
        X_enn, y_enn, test_size=0.25, random_state=42)
    res_enn = tune_train_evaluate_svm(
        X_enn_train, y_enn_train, X_enn_test, y_enn_test, param_grid,
        resampling_method='Edited Nearest Neighbour Undersampling')

    # ... 6. Cluster Centroids Undersampling ...

    X_cc, y_cc = undersampling_clustercentroids(X, y)

    X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(
        X_cc, y_cc, test_size=0.25, random_state=42)
    res_cc = tune_train_evaluate_svm(
        X_cc_train, y_cc_train, X_cc_test, y_cc_test, param_grid,
        resampling_method='Cluster Centroids Undersampling')

    # ... 7. Tomek Links Undersampling ...

    X_tl, y_tl = undersampling_tomeklinks(X, y)

    X_tl_train, X_tl_test, y_tl_train, y_tl_test = train_test_split(
        X_tl, y_tl, test_size=0.25, random_state=42)
    res_tl = tune_train_evaluate_svm(
        X_tl_train, y_tl_train, X_tl_test, y_tl_test, param_grid,
        resampling_method='Tomek Links Undersampling')

    # --- OVERSAMPLING ---
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
                    'under_oversampling_comparison_standardized.csv')

    # ---------------------------------------------------------------
    # --- a) DEVELOP SVM FOR NearMiss UNDERSAMPLING ---
    # ---------------------------------------------------------------
    # # Plot training data

    # print(mod1_clean.columns)
    # df = pd.concat([X_nm_train, y_nm_train], axis=1)
    # # Check for NaNs in specific columns and data types
    # print(df[['HSnum', 'HN72h', 'AvalDay']].isna().sum())
    # # Ensure categorical/int data for hue
    # df['AvalDay'] = df['AvalDay'].astype(int)

    # # Plot with Seaborn
    # plt.figure(figsize=(10, 6))
    # # Edgecolor changed to 'w' for white, or remove if not needed
    # sns.scatterplot(data=df, x='HSnum', y='HN72h', hue='AvalDay',
    #                 palette='coolwarm', s=60, edgecolor='w', alpha=1)
    # plt.title(
    #     'Training Data: scatterplot of Features with Avalanche Day Classification')
    # plt.xlabel('HSnum')
    # plt.ylabel('HN72h')
    # plt.legend(title='AvalDay', loc='upper right')
    # plt.show()

    classifier_cnn = train_evaluate_final_svm(
        X_cnn_train, y_cnn_train, X_cnn_test, y_cnn_test, res_cnn['best_params'])

    # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    feature_importance_df = permutation_ranking(
        classifier_cnn[0], X_cnn_test, y_cnn_test)

    # # Filter the DataFrame to include only positive importance values
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

    # X_cnn2, y_cnn2 = undersampling_cnn(X_new, y_new)

    # # norm = MinMaxScaler().fit(X)
    # # X_norm = pd.DataFrame(norm.transform(
    # #     X_cnn), columns=X_cnn.columns, index=X_cnn.index)

    # X_cnn_train2, X_cnn_test2, y_cnn_train2, y_cnn_test2 = train_test_split(
    #     X_cnn2, y_cnn2, test_size=0.25, random_state=42)

    # res_cnn2 = tune_train_evaluate_svm(
    #     X_cnn_train2, y_cnn_train2, X_cnn_test2, y_cnn_test2, param_grid,
    #     resampling_method='Condensed Nearest Neighbour Undersampling')

    # classifier_cnn2 = train_evaluate_final_svm(
    #     X_cnn_train2, y_cnn_train2, X_cnn_test2, y_cnn_test2, res_cnn2['best_params'])

    # mod1_filtered = mod1_filtered.dropna()

    # X_new = mod1_filtered.drop(columns=['AvalDay'])
    # y_new = mod1_filtered['AvalDay']

    # --- SPLIT TRAIN AND TEST ---

    # ... 4. Condensed Nearest Neighbour Undersampling ...
    resampling_method = 'CNN'
    X_cnn, y_cnn = undersampling_cnn(X_new, y_new)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_cnn, y_cnn, test_size=0.25, random_state=42)
    res_new = tune_train_evaluate_svm(
        X_train_new, y_train_new, X_test_new, y_test_new, param_grid,
        resampling_method='Condensed Nearest Neighbour Undersampling')

    # X_nm_new, y_nm_new = undersampling_nearmiss(X_new, y_new)

    # X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    #     X_nm_new, y_nm_new, test_size=0.25, random_state=42)

    # df = pd.concat([X_train_new, y_train_new], axis=1)
    # # Check for NaNs in specific columns and data types
    # print(df[['HSnum', 'HN72h', 'AvalDay']].isna().sum())
    # # Ensure categorical/int data for hue
    # df['AvalDay'] = df['AvalDay'].astype(int)

    # # Plot with Seaborn
    # plt.figure(figsize=(10, 6))
    # # Edgecolor changed to 'w' for white, or remove if not needed
    # sns.scatterplot(data=df, x='HSnum', y='HN72h', hue='AvalDay',
    #                 palette='coolwarm', s=60, edgecolor='w', alpha=1)
    # plt.title('Training data after scaling')
    # plt.xlabel('HSnum')
    # plt.ylabel('HN72h')
    # plt.legend(title='AvalDay', loc='upper right')
    # plt.show()

    # res_nm_new = tune_train_evaluate_svm(
    #     X_train_new, y_train_new, X_test_new, y_test_new)

    res_new_list = []

    # Add each result to the list with the sampling method as an identifier
    res_new_list.append(
        {'Run': '1', **res_new})

    classifier_new = train_evaluate_final_svm(
        X_train_new, y_train_new, X_test_new, y_test_new, res_new['best_params'])

    # Calculate evaluation metrics
    y_predict = classifier_new.predict(X_test_new)
    accuracy = accuracy_score(y_test_new, y_predict)
    precision = precision_score(y_test_new, y_predict)
    recall = recall_score(y_test_new, y_predict)
    f1 = f1_score(y_test_new, y_predict)

    res_2 = {
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'best_params': {'C': classifier_new.C, 'gamma': classifier_new.gamma}
    }

    res_new_list.append(
        {'Run': '2', **res_2})
    res_new_df = pd.DataFrame(res_new_list)

    save_outputfile(res_new_df, common_path /
                    f'all_feature_{resampling_method}.csv')

    feature_importance_df = permutation_ranking(
        classifier_new, X_test_new, y_test_new)

    # # ---------------------------------------------------------------
    # # --- b) DEVELOP SVM FOR SMOTE OVERSAMPLING ---
    # # ---------------------------------------------------------------

    # classifier_sm = train_evaluate_final_svm(
    #     X_sm, y_sm, X_test, y_test, res_sm)

    # # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    # feature_importance_df = permutation_ranking(classifier_sm, X_test, y_test)

    # # Filter the DataFrame to include only positive importance values
    # positive_features = feature_importance_df[feature_importance_df['Importance_Mean'] > 0]

    # # Get only the feature names
    # features_plus_aval = positive_features['Feature'].tolist() + ['AvalDay']

    # # --- NEW SVM MODEL WITH FEATURES SELECTED ---

    # # mod1_clean = mod1.dropna()  # Keep the clean DataFrame with the datetime index
    # # X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    # mod1_filtered = mod1[features_plus_aval]
    # mod1_filtered = mod1_filtered.dropna()

    # X_new = mod1_filtered.drop(columns=['AvalDay'])
    # y_new = mod1_filtered['AvalDay']

    # # --- SCALING ---

    # scaler = StandardScaler()
    # X_new = pd.DataFrame(scaler.fit_transform(X_new),
    #                      columns=X_new.columns,
    #                      index=X_new.index)

    # # --- SPLIT TRAIN AND TEST ---

    # X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    #     X_new, y_new, test_size=0.25, random_state=42)

    # X_sm_new, y_sm_new = oversampling_smote(X_train_new, y_train_new)

    # res_sm_new = train_and_evaluate_svm(
    #     X_sm_new, y_sm_new, X_test_new, y_test_new)

    # classifier_sm_new = train_evaluate_final_svm(
    #     X_sm_new, y_sm_new, X_test_new, y_test_new, res_sm_new)

    # feature_importance_df = permutation_ranking(
    #     classifier_sm_new, X_test_new, y_test_new)

    # ---------------------------------------------------------------
    # --- c) TEST DIFFERENT CONFIGURATION OF  ---
    # ---------------------------------------------------------------

    candidate_features = [
        'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_5d',
        'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
        'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
        'TempAmplitude_3d', 'TempAmplitude_5d', 'Ta_delta_1d', 'Ta_delta_2d',
        'Ta_delta_3d', 'Ta_delta_5d', 'Tmin_delta_1d', 'Tmin_delta_2d',
        'Tmin_delta_3d', 'Tmin_delta_5d', 'Tmax_delta_1d', 'Tmax_delta_2d',
        'Tmax_delta_3d', 'Tmax_delta_5d', 'T_mean', 'DegreeDays_Pos',
        'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
        'SnowDrift_1d', 'SnowDrift_2d', 'SnowDrift_3d', 'SnowDrift_5d',
        'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
        'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
        'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
        'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    ]

    # Base predictors
    base_predictors = ['HSnum']

    # Initialize results dictionary
    results = {}

    # Loop through each candidate feature and test its performance
    for feature in candidate_features:
        # Define the current set of features to evaluate
        current_features = base_predictors + [feature]
        # current_features = [feature]

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

    save_outputfile(df, common_path / 'precision_features.csv')

    # -------------------------------------------------------
    # TEST FEATURES PERFORMANCE
    # -------------------------------------------------------

    # ....... 1. SNOW LOAD DUE SNOWFALL ...........................

    s1 = ['HSnum']

    s1 = ['HSnum', 'HN_5d']
    res1 = evaluate_svm_with_feature_selection(mod1, s1)

    s2 = ['HN_5d', 'HSnum']
    res2 = evaluate_svm_with_feature_selection(mod1, s2)

    s3 = s2 + ['HN_2d']
    res3 = evaluate_svm_with_feature_selection(mod1, s3)

    s4 = s2 + ['HN_3d']
    res4 = evaluate_svm_with_feature_selection(mod1, s4)

    s5 = s4 + ['HN_5d']
    res5 = evaluate_svm_with_feature_selection(mod1, s5)

    s6 = s5 + ['Precip_1d']
    res6 = evaluate_svm_with_feature_selection(mod1, s6)

    s7 = s6 + ['Precip_2d']
    res7 = evaluate_svm_with_feature_selection(mod1, s7)

    s8 = s7 + ['Precip_3d']
    res8 = evaluate_svm_with_feature_selection(mod1, s8)

    s9 = s8 + ['Precip_5d']
    res9 = evaluate_svm_with_feature_selection(mod1, s9)

    s10 = s9 + ['FreshSWE']
    res10 = evaluate_svm_with_feature_selection(mod1, s10)

    s11 = s10 + ['SeasonalSWE_cum']
    res11 = evaluate_svm_with_feature_selection(mod1, s11)

    # PLOTS
    # Combine the results into a list
    results_features = [res1, res2, res3, res4,
                        res5, res6, res7, res8, res9, res10, res11]

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
    # plt.plot(df_res['Configuration'], df_res['F1'],
    #          marker='x', linestyle='-.', color='red', label='F1 Score')
    # plt.plot(df_res['Configuration'], df_res['Recall'],
    #          marker='s', linestyle='--', color='green', label='Recall')
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

    save_outputfile(df_res, common_path / 'config_snowload_features.csv')

    # RESULTS: the best configuration based on Recall is:
    # s11: HSnum, HNnum, HN_2d, HN_3d, HN_5d,
    #      Precip_1d, Precip_2d, Precip_3d, Precip_5d,
    #      FreshSWE, SeasonalSWE_cum

    # ....... 2. SNOW LOAD DUE WIND DRIFT ...........................

    wd4 = s3 + ['SnowDrift_1d']
    res_wd4 = evaluate_svm_with_feature_selection(mod1, wd4)

    wd5 = wd4 + ['SnowDrift_2d']
    res_wd5 = evaluate_svm_with_feature_selection(mod1, wd5)

    wd6 = wd5 + ['SnowDrift_3d']
    res_wd6 = evaluate_svm_with_feature_selection(mod1, wd6)

    wd7 = wd6 + ['SnowDrift_5d']
    res_wd7 = evaluate_svm_with_feature_selection(mod1, wd7)

    results_features = [res3, res_wd4, res_wd5, res_wd6, res_wd7]

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
    df_res_wd = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    # plt.plot(df_res['Configuration'], df_res['F1'],
    #          marker='x', linestyle='-.', color='red', label='F1 Score')
    # plt.plot(df_res['Configuration'], df_res['Recall'],
    #          marker='s', linestyle='--', color='green', label='Recall')
    plt.plot(df_res_wd['Configuration'], df_res_wd['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_wd['Configuration'], df_res_wd['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Snow Drift features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_wd, common_path / 'config_snowdrift_features.csv')

    # RESULTS: the wind drift does not provide improvements to the model.

    # ....... 3. PAST AVALANCHE ACTIVITY ...........................

    a10 = s3 + ['AvalDay_2d']
    res_a10 = evaluate_svm_with_feature_selection(mod1, a10)

    a11 = a10 + ['AvalDay_3d']
    res_a11 = evaluate_svm_with_feature_selection(mod1, a11)

    a12 = a11 + ['AvalDay_5d']
    res_a12 = evaluate_svm_with_feature_selection(mod1, a12)

    results_features = [res3, res_a10, res_a11, res_a12]

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
    df_res_av_act = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res_av_act['Configuration'], df_res_av_act['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_av_act['Configuration'], df_res_av_act['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Avalanche activity features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_av_act, common_path /
                    'config_avalanche_activity_features.csv')

    # RESULTS: the best configuration based on Recall is
    # a12: HSnum, HNnum, HN_2d, HN_3d, HN_5d,
    #      Precip_1d, Precip_2d, Precip_3d, Precip_5d,
    #      AvalDay_2d, AvalDay_3d, AvalDay_5d

    # ....... 4. SNOW TEMPERATURE  ...........................

    ts13 = s3 + ['TH01G']
    res_ts13 = evaluate_svm_with_feature_selection(mod1, ts13)

    ts14 = ts13 + ['Tsnow_delta_1d']
    res_ts14 = evaluate_svm_with_feature_selection(mod1, ts14)

    ts15 = ts14 + ['Tsnow_delta_2d']
    res_ts15 = evaluate_svm_with_feature_selection(mod1, ts15)

    ts16 = ts15 + ['Tsnow_delta_3d']
    res_ts16 = evaluate_svm_with_feature_selection(mod1, ts16)

    ts17 = ts16 + ['Tsnow_delta_5d']
    res_ts17 = evaluate_svm_with_feature_selection(mod1, ts17)

    results_features = [res3, res_ts14,
                        res_ts15, res_ts15, res_ts16, res_ts17]

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
    df_res_ts = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res_ts['Configuration'], df_res_ts['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_ts['Configuration'], df_res_ts['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Snow Temperature features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_ts, common_path /
                    'config_snowtemp_features.csv')

    # ....... 5. WEAK LAYER DETECTION ...........................

    pr1 = s3 + ['Penetration_ratio']
    res_pr1 = evaluate_svm_with_feature_selection(mod1, pr1)

    pr2 = s3 + ['PR']
    res_pr2 = evaluate_svm_with_feature_selection(mod1, pr2)

    pr3 = s3 + ['MF_Crust_Present']
    res_pr3 = evaluate_svm_with_feature_selection(mod1, pr3)

    pr4 = s3 + ['New_MF_Crust']
    res_pr4 = evaluate_svm_with_feature_selection(mod1, pr4)

    pr5 = s3 + ['ConsecCrustDays']
    res_pr5 = evaluate_svm_with_feature_selection(mod1, pr5)

    pr6 = s3 + ['TempGrad_HS']
    res_pr6 = evaluate_svm_with_feature_selection(mod1, pr6)

    results_features = [res3, res_pr1, res_pr2,
                        res_pr3, res_pr4, res_pr5, res_pr6]

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
    df_res_ts = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res_ts['Configuration'], df_res_ts['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_ts['Configuration'], df_res_ts['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Weak Layer features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_ts, common_path /
                    'config_weaklayer_features.csv')

    # ....... COMPARE RESULTS ...........................

    results_features = [res1, res2, res3, res4, res5, res6,
                        res7, res8, res9, res10, res11,
                        res_wd4, res_wd5, res_wd6, res_wd7,
                        res_a10, res_a11, res_a12,
                        res_ts14, res_ts15, res_ts15, res_ts16, res_ts17,
                        res_pr1, res_pr2, res_pr3, res_pr4, res_pr5, res_pr6]

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
    df_res_ts = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res_ts['Configuration'], df_res_ts['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_ts['Configuration'], df_res_ts['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_ts, common_path /
                    'config_test_features.csv')

    # ....... 6. AIR TEMPERATURE  ...........................

    ta1 = s3 + ['TaG']
    res_ta1 = evaluate_svm_with_feature_selection(mod1, ta1)

    ta2 = s3 + ['TminG']
    res_ta2 = evaluate_svm_with_feature_selection(mod1, ta2)

    ta3 = s3 + ['TmaxG']
    res_ta3 = evaluate_svm_with_feature_selection(mod1, ta3)

    ta4 = s4 + ['T_mean']
    res_ta4 = evaluate_svm_with_feature_selection(mod1, ta4)

    ta1 = s3 + ['TaG']
    res_ta1 = evaluate_svm_with_feature_selection(mod1, ta1)

    results_features = [res3, res_ta1, res_ta2, res_ta3, res_ta4]

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
    df_res_ts = pd.DataFrame(data_res)

    # Plotting a line plot for precision
    plt.figure(figsize=(8, 5))
    plt.plot(df_res_ts['Configuration'], df_res_ts['Accuracy'],
             marker='d', linestyle=':', label='Accuracy')
    plt.plot(df_res_ts['Configuration'], df_res_ts['Recall'],
             marker='o', linestyle='-', label='Recall')
    plt.title('Scores for Snow Temperature features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_ts, common_path /
                    'config_airtemp_features.csv')

    # if __name__ == '__main__':
    #     main()
