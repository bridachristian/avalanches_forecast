# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:24:26 2024

@author: Christian
"""

import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


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
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid,
                        scoring='f1_macro', verbose=3)

    # Fit GridSearchCV with X and y
    grid.fit(X, y)
    print(f'Best parameters: {grid.best_params_}')

    # Creating new SVM model with the best parameters
    clf = svm.SVC(kernel='rbf', C=grid.best_params_[
                  'C'], gamma=grid.best_params_['gamma'])

    scores = cross_val_score(clf, X_train, y_train, cv=5)

    print("Average Cross-Validation Score:", scores.mean())
    print("Standard Deviation of Scores:", scores.std())

    clf.fit(X, y)

    test_accuracy = clf.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, val_scores = learning_curve(
        clf, X_train, y_train, cv=10)
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.ylim(0.5, 1)
    plt.legend()
    plt.show()

    # Training new SVM model
    # clf.fit(X, y)
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
        'recall': recall,
        'accuracy': accuracy,
        'precision': precision,
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


def develop_SVM(X_train, y_train, X_test, y_test, res_nm):
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
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid,
                        cv=10, scoring='f1_macro', verbose=3)
    grid.fit(X_train, y_train)

    print(f'Best parameters: {grid.best_params_}')

    # Creating new SVM model with the best parameters
    clf = svm.SVC(kernel='rbf', C=grid.best_params_[
        'C'], gamma=grid.best_params_['gamma'])

    scores = cross_val_score(clf, X_train, y_train, cv=10)

    print("Average Cross-Validation Score:", scores.mean())
    print("Standard Deviation of Scores:", scores.std())

    # Training the new SVM model
    out = clf.fit(X_train, y_train)

    test_accuracy = out.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)

    train_sizes, train_scores, val_scores = learning_curve(
        out, X_train, y_train, cv=10)
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training Score", color='red')
    plt.plot(train_sizes, val_mean, label="Validation Score", color='green')
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.ylim(0.5, 1)
    plt.legend()
    plt.show()

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

    res = {
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'best_params': grid.best_params_
    }
    return out, res


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


def test_features_config(mod1, feature):
    feature_plus = feature + ['AvalDay']

    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    # Extract target and features from dataframe
    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # Undersampling using NearMiss method
    X_nm, y_nm = undersampling_nearmiss(X, y)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_nm, y_nm, test_size=0.25, random_state=42)

    # First Cross Validation and performance evaluation
    result_1iter = train_and_evaluate_svm(X_train, y_train, X_test, y_test)

    # Second Cross Validation and model training
    classifier, result_2iter = develop_SVM(
        X_train, y_train, X_test, y_test, result_1iter)

    return feature, classifier, result_2iter


def save_outputfile(df, output_filepath):
    """Save the mod1_features dataframe to a CSV file."""
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


# def main():
if __name__ == '__main__':
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
    # feature = ['HN_3d', 'HSnum']
    feature = [
        'N', 'V',  'TaG', 'TminG', 'TmaxG', 'HSnum',
        'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
        'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
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
        'TempGrad_HS', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
        'Tsnow_delta_5d', 'SnowConditionIndex', 'ConsecWetSnowDays',
        'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays',
        'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
    ]

    feature_plus = feature + ['AvalDay']
    mod1_clean = mod1[feature_plus]
    mod1_clean = mod1_clean.dropna()

    # X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    X = mod1_clean[feature]
    y = mod1_clean['AvalDay']

    # --- Plot example for 2 features classification ----
    #
    # # Confirm columns in mod1_clean
    # print(mod1_clean.columns)
    # df = mod1_clean
    # # Check for NaNs in specific columns and data types
    # print(df[['HSnum', 'HN72h', 'AvalDay']].isna().sum())
    # # Ensure categorical/int data for hue
    # df['AvalDay'] = df['AvalDay'].astype(int)

    # # Plot with Seaborn
    # plt.figure(figsize=(10, 6))
    # # Edgecolor changed to 'w' for white, or remove if not needed
    # sns.scatterplot(data=df, x='HSnum', y='HN72h', hue='AvalDay',
    #                 palette='coolwarm', s=60, edgecolor='w', alpha=0.5)
    # plt.title('Scatter Plot of Features with Avalanche Day Classification')
    # plt.xlabel('HSnum')
    # plt.ylabel('HN72h')
    # plt.legend(title='AvalDay', loc='upper right')
    # plt.show()

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

    save_outputfile(results_df, common_path /
                    'under_oversampling_comparison.csv')

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

    classifier_nm = develop_SVM(
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

    res_nm_new = train_and_evaluate_svm(
        X_train_new, y_train_new, X_test_new, y_test_new)

    res_nm_new_list = []

    # Add each result to the list with the sampling method as an identifier
    res_nm_new_list.append(
        {'Run': '1', **res_nm_new})

    classifier_nm_new = develop_SVM(
        X_train_new, y_train_new, X_test_new, y_test_new, res_nm_new)

    # Calculate evaluation metrics
    y_predict = classifier_nm_new.predict(X_test_new)
    accuracy = accuracy_score(y_test_new, y_predict)
    precision = precision_score(y_test_new, y_predict, average='macro')
    recall = recall_score(y_test_new, y_predict, average='macro')
    f1 = f1_score(y_test_new, y_predict, average='macro')

    res_2 = {
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'best_params': {'C': classifier_nm_new.C, 'gamma': classifier_nm_new.gamma}
    }

    res_nm_new_list.append(
        {'Run': '2', **res_2})
    res_nm_new_df = pd.DataFrame(res_nm_new_list)

    save_outputfile(res_nm_new_df, common_path / 'nearmiss_result.csv')

    feature_importance_df = permutation_ranking(
        classifier_nm_new, X_test_new, y_test_new)

    # # ---------------------------------------------------------------
    # # --- b) DEVELOP SVM FOR SMOTE OVERSAMPLING ---
    # # ---------------------------------------------------------------

    # classifier_sm = develop_SVM(
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

    # classifier_sm_new = develop_SVM(
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
        'TempGrad_HS', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
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
        # current_features = base_predictors + [feature]
        current_features = [feature]

        # Evaluate the model with the selected features
        result = test_features_config(mod1, current_features)

        # Store the result in the dictionary
        results[feature] = result

        # Print the evaluated feature and the result
        # print(f"Evaluated Feature: {feature}, Result: {result}")

    # Identify the best-performing feature based on the evaluation metric
    # Assuming higher is better; adjust based on metric
    # Extract the feature with the maximum precision
    best_feature = max(
        results, key=lambda x: results[x][1]['recall'])
    max_value = results[best_feature][1]['recall']

    print(
        f"Best Feature: {best_feature}, Best Result: {max_value}")

    data = []
    for key, (model, metrics) in results.items():
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

    f1 = ['HSnum']
    res1 = test_features_config(mod1, f1)

    f2 = f1 + ['HNnum']
    res2 = test_features_config(mod1, f2)

    f3 = f2 + ['HN_2d']
    res3 = test_features_config(mod1, f3)

    f4 = f3 + ['HN_3d']
    res4 = test_features_config(mod1, f4)

    f5 = f4 + ['HN_5d']
    res5 = test_features_config(mod1, f5)

    f6 = f5 + ['Precip_1d']
    res6 = test_features_config(mod1, f6)

    f7 = f6 + ['Precip_2d']
    res7 = test_features_config(mod1, f7)

    f8 = f7 + ['Precip_3d']
    res8 = test_features_config(mod1, f8)

    f9 = f8 + ['Precip_5d']
    res9 = test_features_config(mod1, f9)

    f10 = f9 + ['FreshSWE']
    res10 = test_features_config(mod1, f10)

    f11 = f10 + ['SeasonalSWE_cum']
    res11 = test_features_config(mod1, f11)

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
    plt.ylabel('Precision Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    df_res = df_res.sort_values(by='Recall', ascending=False)

    save_outputfile(df_res, common_path / 'config_snowload_features.csv')

    # RESULTS: the best configuration based on Recall it the number 9 (or 10)

    # ....... 2. SNOW LOAD DUE WIND DRIFT ...........................

    wd4 = f9 + ['SnowDrift_1d']
    res_wd4 = test_features_config(mod1, wd4)

    wd5 = wd4 + ['SnowDrift_2d']
    res_wd5 = test_features_config(mod1, wd5)

    wd6 = wd5 + ['SnowDrift_3d']
    res_wd6 = test_features_config(mod1, wd6)

    wd7 = wd6 + ['SnowDrift_5d']
    res_wd7 = test_features_config(mod1, wd7)

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
    plt.ylabel('Precision Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_wd, common_path / 'config_snowdrift_features.csv')

    # Evaluate a combination of snow metrics as predictors:
    # - `HN_3d`: Total new snow height accumulated over the last 3 days, indicating recent snowfall
    # - `HSnum`: Current snowpack height, representing the total snow depth at the time
    # - `PR`: Penetration, a measure of snow resistance (indicator of snowpack strength)
    f3 = ['HSnum', 'HN_3d', 'PR']
    res3 = test_features_config(mod1, f3)

    # Evaluate a combination of snow metrics as predictors for model performance:
    # - `HN_3d`: Total new snow height accumulated over the last 3 days, indicating recent snowfall and its potential impact on snowpack stability.
    # - `HSnum`: Current snowpack height, representing the total depth of snow at the time of measurement, which is crucial for understanding the overall snow accumulation in a given area.
    # - `PR`: Penetration, a measure of snow resistance (indicator of snowpack strength)
    # - `HN_5d`: Total new snow height accumulated over the last 5 days, capturing medium-term snow accumulation trends and helping to account for sustained snowfall events.
    f4 = ['HN_3d', 'HSnum', 'PR', 'HN_5d']
    res4 = test_features_config(mod1, f4)

    # Evaluate a combination of snow metrics as predictors for model performance:
    f5 = ['HN_3d', 'HSnum', 'PR', 'HN_5d', 'Precip_5d']
    res5 = test_features_config(mod1, f5)

    # Evaluate a combination of snow metrics as predictors for model performance:
    f6 = ['HN_3d', 'HSnum', 'PR', 'HN_5d', 'Precip_5d', 'Precip_3d']
    res6 = test_features_config(mod1, f6)

    # Evaluate a combination of snow metrics as predictors for model performance:
    f7 = ['HN_3d', 'HSnum', 'PR', 'HN_5d',
          'Precip_5d', 'Precip_3d', 'TH03G']
    res7 = test_features_config(mod1, f7)

    # if __name__ == '__main__':
    #     main()
