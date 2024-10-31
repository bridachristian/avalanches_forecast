# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:24:26 2024

@author: Christian
"""

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance


def load_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    # mod1 = mod1.drop(columns=['Stagione'])
    mod1['DataRilievo'] = pd.to_datetime(
        mod1['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index
    mod1.set_index('DataRilievo', inplace=True)

    return mod1


def undersampling_random(X, y):
    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Check the new class distribution
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def undersampling_random_timelimited(X, y, Ndays=10):
    # Convert y to a DataFrame to retain the index
    y_df = pd.DataFrame(y).copy()
    y_df.columns = ['AvalDay']

    # Find indices where avalanche events occur
    avalanche_dates = y_df[y_df['AvalDay'] == 1].index

    # Create a mask to keep data within 10 days before each avalanche event
    mask = pd.Series(False, index=y.index)

    # Mark the 10 days before each avalanche event
    for date in avalanche_dates:
        mask.loc[date - pd.Timedelta(days=Ndays):date] = True

    # Separate the data into non-avalanche events within the 10-day window and other data
    X_window = X[mask]
    y_window = y[mask]
    X_other = X[~mask]
    y_other = y[~mask]

    # Select only non-avalanche events from the window
    non_avalanche_mask = (y_window == 0)
    X_non_avalanche = X_window[non_avalanche_mask]
    y_non_avalanche = y_window[non_avalanche_mask]

    avalanche_mask = (y_window == 1)
    X_avalanche = X_window[avalanche_mask]
    y_avalanche = y_window[avalanche_mask]

    X_new = pd.concat([X_non_avalanche, X_avalanche])
    y_new = pd.concat([y_non_avalanche, y_avalanche])

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_new, y_new)

    # Check the new class distribution
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def undersampling_nearmiss(X, y, version=1, n_neighbors=3):
    """
    Apply NearMiss undersampling algorithm to balance the classes in an imbalanced dataset.

    Parameters:
    - X: pd.DataFrame or np.ndarray, feature matrix
    - y: pd.Series or np.ndarray, target vector with binary classes (0 and 1)
    - version: int, NearMiss version (1, 2, or 3). Default is version 1.
    - n_neighbors: int, number of neighbors to consider for selecting samples. Default is 3.

    Returns:
    - X_res: Resampled feature matrix after NearMiss undersampling
    - y_res: Resampled target vector after NearMiss undersampling
    """
    # Initialize the NearMiss object with the chosen version
    nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)

    # Apply NearMiss undersampling
    try:
        X_res, y_res = nearmiss.fit_resample(X, y)
    except ValueError as e:
        print(f"Error during resampling: {e}")
        # If an error occurs, return the original dataset
        return X, y

    # Display the class distribution after undersampling
    print("NearMiss: Original class distribution:", Counter(y))
    print("NearMiss: Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def oversampling_random(X, y):
    ros = RandomOverSampler(random_state=42)
    # Apply Random oversampling
    X_res, y_res = ros.fit_resample(X, y)

    # Display the class distribution before and after SMOTE
    print("RandomOverSampler: Original class distribution:", Counter(y))
    print("RandomOverSampler: Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def oversampling_smote(X, y):
    smote = SMOTE(sampling_strategy='minority', random_state=42)

    # Apply SMOTE oversampling
    X_res, y_res = smote.fit_resample(X, y)

    # Display the class distribution before and after SMOTE
    print("SMOTE: Original class distribution:", Counter(y))
    print("SMOTE: Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def oversampling_adasyn(X, y):
    # Initialize the ADASYN object
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)

    # Apply ADASYN oversampling
    X_res, y_res = adasyn.fit_resample(X, y)

    # Display the class distribution before and after SMOTE
    print("ADASYN: Original class distribution:", Counter(y))
    print("ADASYN: Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def oversampling_svmsmote(X, y):

    # Initialize the SVMSMOTE object
    svmsmote = SVMSMOTE(sampling_strategy='minority', random_state=42)

    # Apply SVMSMOTE oversampling
    X_res, y_res = svmsmote.fit_resample(X, y)

    # Display the class distribution before and after SVMSMOTE

    print("SVMSMOTE: Original class distribution:", Counter(y))
    print("SVMSMOTE: Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def train_and_evaluate_svm(X, y, X_test, y_test):
    # --- CREATE SVM MODEL ---
    clf = svm.SVC(kernel='rbf')

    # Training SVM model
    clf.fit(X, y)

    # Tuning SVM hyperparameters
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001]}
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, verbose=3)

    # Fit GridSearchCV with X and y
    grid.fit(X, y)
    print(f'Best parameters: {grid.best_params_}')

    # Creating new SVM model with the best parameters
    clf = svm.SVC(kernel='rbf', C=grid.best_params_[
                  'C'], gamma=grid.best_params_['gamma'])

    # Training new SVM model
    clf.fit(X, y)

    # Ensure y_test is a DataFrame or a 2D array with one row per sample
    # Predicting on the test data
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_params': grid.best_params_
    }


def plot_confusion_matrix(y_test, y_pred):
    # Step 1: Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Step 2: Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return cm


def plot_roc_curve(X_test, y_test, clf):
    # Compute ROC curve and ROC area
    y_score = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def develop_svm_nearmiss(X_train, y_train, X_test, y_test, res_nm):
    # Create SVM classifier
    clf = svm.SVC(kernel='rbf')

    # Training initial SVM model
    clf.fit(X_train, y_train)

    print("Training class distribution:", Counter(y_train))
    print("Testing class distribution:", Counter(y_test))

    # Get the best parameters from the res_nm results
    C_value = res_nm['best_params']['C']
    gamma_value = res_nm['best_params']['gamma']

    # Create a range for C and gamma
    C_range = np.linspace(C_value * 0.5, C_value * 1.5, 21)
    gamma_range = np.linspace(gamma_value * 0.5, gamma_value * 1.5, 21)

    # Tuning SVM hyperparameters with GridSearchCV
    param_grid = {'C': C_range, 'gamma': gamma_range}
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, verbose=3)
    grid.fit(X_train, y_train)

    print(f'Best parameters: {grid.best_params_}')

    # Creating new SVM model with the best parameters
    clf = svm.SVC(kernel='rbf', C=grid.best_params_[
        'C'], gamma=grid.best_params_['gamma'])

    # Training the new SVM model
    out = clf.fit(X_train, y_train)

    # Predicting on the test data
    y_pred = clf.predict(X_test)

    # Create confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    # Compute ROC curve and ROC area
    plot_roc_curve(X_test, y_test, clf)
    return out


def permutation_ranking(classifier, X_test, y_test):

    perm_importance = permutation_importance(
        classifier, X_test, y_test, n_repeats=30, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()

    feature_importance_df = pd.DataFrame({
        'Feature': X_test.columns[sorted_idx],
        'Ranking': range(1, len(sorted_idx) + 1),
        'Importance_Mean': perm_importance.importances_mean[sorted_idx],
        'Importance_Std': perm_importance.importances_std[sorted_idx]
    })

    # Plot permutation importance with error bars
    plt.figure(figsize=(10, 16))
    plt.barh(
        range(len(sorted_idx)),
        perm_importance.importances_mean[sorted_idx],
        xerr=perm_importance.importances_std[sorted_idx],  # Adding error bars
        align='center',
        capsize=5,  # Adding caps to error bars for clarity
    )
    plt.yticks(range(len(sorted_idx)), X_test.columns[sorted_idx])
    plt.title("Feature Importance (Permutation Importance)")
    plt.xlabel("Mean Decrease in Accuracy")
    plt.show()

    return feature_importance_df


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures.csv'

    # output_filepath = common_path / 'mod1_undersampling.csv'
    # output_filepath2 = common_path / 'mod1_oversampling.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- FEATURES SELECTION ---
    feature = ['HN72h', 'TH01G', 'Tmin120h']
    # feature = [
    #     'N', 'V', 'TaG', 'TminG', 'TmaxG',
    #     'HSnum', 'HNnum', 'TH01G', 'TH03G', 'PR', 'CS',
    #     'HSdiff24h', 'HSdiff48h', 'HSdiff72h', 'HSdiff120h', 'HN48h', 'HN72h',
    #     'HN120h', 'NewSnowIndex', 'NewSnow_5cm', 'NewSnow_15cm', 'NewSnow_30cm',
    #     'NewSnow_50cm', '3dNewSnow_10cm', '3dNewSnow_30cm', '3dNewSnow_60cm',
    #     '3dNewSnow_100cm', 'DaysSinceLastSnow', 'Tmin48h', 'Tmax48h', 'Tmin72h',
    #     'Tmax72h', 'Tmin120h', 'Tmax120h', 'Tdelta24h', 'Tdelta48h',
    #     'Tdelta72h', 'Tdelta120h', 'Tavg', 'DegreeDays',
    #     'CumulativeDegreeDays48h', 'CumulativeDegreeDays72h',
    #     'CumulativeDegreeDays120h', 'SWEnew', 'SWE_cumulative',
    #     'PSUM24h', 'PSUM48h', 'PSUM72h', 'PSUM120h', 'Penetration_ratio',
    #     'T_gradient', 'AvalDay48h', 'AvalDay72h', 'AvalDay120h'
    # ]

    mod1_clean = mod1[feature]
    mod1_clean = mod1_clean.dropna()

    # X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- SPLIT TRAIN AND TEST ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    print("Original class distribution:", Counter(y))
    print("Original class distribution training set:", Counter(y_train))
    print("Original class distribution test set:", Counter(y_test))

    # --- UNDERSAMPLING ---
    X_rand, y_rand = undersampling_random(X, y)
    X_rand_10d, y_rand_10d = undersampling_random_timelimited(X, y, Ndays=10)
    X_nm, y_nm = undersampling_nearmiss(X, y)

    # --- OVERSAMPLING ---
    X_ros, y_ros = oversampling_random(X_train, y_train)
    X_sm, y_sm = oversampling_smote(X_train, y_train)
    X_adas, y_adas = oversampling_adasyn(X_train, y_train)
    X_svmsm, y_svmsm = oversampling_svmsmote(X_train, y_train)

    # --- CREATE SVM MODEL ---

    # 1. Random undersampling
    X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(
        X_rand, y_rand, test_size=0.25, random_state=42)
    res_rand = train_and_evaluate_svm(
        X_rand_train, y_rand_train, X_rand_test, y_rand_test)

    # 2. Random undersampling N days before
    X_rand_10d_train, X_rand_10d_test, y_rand_10d_train, y_rand_10d_test = train_test_split(
        X_rand_10d, y_rand_10d, test_size=0.25, random_state=42)
    res_rand_10d = train_and_evaluate_svm(
        X_rand_10d_train, y_rand_10d_train, X_rand_10d_test, y_rand_10d_test)

    # 3. Nearmiss undersampling
    X_nm_train, X_nm_test, y_nm_train, y_nm_test = train_test_split(
        X_nm, y_nm, test_size=0.25, random_state=42)
    res_nm = train_and_evaluate_svm(
        X_nm_train, y_nm_train, X_nm_test, y_nm_test)

    # 4. Random oversampling
    res_ros = train_and_evaluate_svm(X_ros, y_ros, X_test, y_test)

    # 5. SMOTE oversampling
    res_sm = train_and_evaluate_svm(X_sm, y_sm, X_test, y_test)

    # 6. adasyn oversampling
    res_adas = train_and_evaluate_svm(X_adas, y_adas, X_test, y_test)

    # 7. SVMSMOTE oversampling
    res_svmsm = train_and_evaluate_svm(X_svmsm, y_svmsm, X_test, y_test)

    # --- STORE RESULTS IN A DATAFRAME ---

    # List to store results
    results_list = []

    # Add each result to the list with the sampling method as an identifier
    results_list.append(
        {'sampling_method': 'Random_Undersampling', **res_rand})
    results_list.append(
        {'sampling_method': 'Random_Undersampling_10d', **res_rand_10d})
    results_list.append(
        {'sampling_method': 'Nearmiss_Undersampling', **res_nm})
    results_list.append({'sampling_method': 'Random_Oversampling', **res_ros})
    results_list.append({'sampling_method': 'SMOTE_Oversampling', **res_sm})
    results_list.append({'sampling_method': 'ADASYN_Oversampling', **res_adas})
    results_list.append(
        {'sampling_method': 'SVMSMOTE_Oversampling', **res_svmsm})

    # Convert list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_list)
    print(results_df)

    # --- DEVELOP SVM FOR NearMiss UNDERSAMPLING ---

    classifier_nm = develop_svm_nearmiss(
        X_nm_train, y_nm_train, X_nm_test, y_nm_test, res_nm)

    # --- PERMUTATION IMPORTANCE FEATURE SELECTION ---

    feature_importance_df = permutation_ranking(classifier_nm, X_test, y_test)

    # Filter the DataFrame to include only positive importance values
    positive_features = feature_importance_df[feature_importance_df['Importance_Mean'] > 0]

    # Get only the feature names
    features_plus_aval = positive_features['Feature'].tolist() + ['AvalDay']

    # --- NEW SVM MODEL WITH FEATURES SELECTED ---

    # mod1_clean = mod1.dropna()  # Keep the clean DataFrame with the datetime index
    # X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    mod1_filtered = mod1[features_plus_aval]
    mod1_filtered = mod1_filtered.dropna()

    X_new = mod1_filtered.drop(columns=['AvalDay'])
    y_new = mod1_filtered['AvalDay']

    # --- SCALING ---

    scaler = StandardScaler()
    X_new = pd.DataFrame(scaler.fit_transform(X_new),
                         columns=X_new.columns,
                         index=X_new.index)

    # --- SPLIT TRAIN AND TEST ---

    X_nm_new, y_nm_new = undersampling_nearmiss(X_new, y_new)

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_nm_new, y_nm_new, test_size=0.25, random_state=42)

    res_nm_new = train_and_evaluate_svm(
        X_train_new, y_train_new, X_test_new, y_test_new)

    classifier_nm_new = develop_svm_nearmiss(
        X_train_new, y_train_new, X_test_new, y_test_new, res_nm_new)

    feature_importance_df = permutation_ranking(
        classifier_nm_new, X_test_new, y_test_new)


if __name__ == '__main__':
    main()
