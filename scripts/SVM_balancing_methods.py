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


def cross_validate_svm(X, y, param_grid, cv=5, scoring='recall'):
    """
    Performs hyperparameter tuning and cross-validation for an SVM model.

    Parameters:
        X (array-like): Training data features.
        y (array-like): Training data labels.
        param_grid (dict): Dictionary with parameters names (`C` and `gamma`) as keys and lists of parameter settings to try as values.
        cv (int): Number of cross-validation folds (default is 5).
        scoring (str): Scoring metric for GridSearchCV (default is 'f1_macro').

    Returns:
        dict: Contains the best parameters, cross-validation scores, and the best model.
    """
    # Initialize GridSearchCV for hyperparameter tuning
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid,
                        cv=cv, scoring=scoring, verbose=3)
    grid.fit(X, y)

    # Extract the best parameters and model
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    # Perform cross-validation using the best model
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)
    print("Best Parameters:", best_params)
    print("Average Cross-Validation Score:", cv_scores.mean())
    print("Standard Deviation of Scores:", cv_scores.std())

    # Return the best model and evaluation metrics
    return {
        'best_model': best_model,
        'best_params': best_params,
        'cv_mean_score': cv_scores.mean(),
        'cv_std_score': cv_scores.std()
    }


def plot_learning_curve(clf, X, y, cv=5):
    """
    Plots the learning curve for the given classifier.

    Parameters:
    - clf: Trained classifier (e.g., SVM model)
    - X: Feature data
    - y: Target labels
    - cv: Number of cross-validation folds (default is 10)

    Returns:
    - None (displays a plot)
    """
    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(clf, X, y, cv=cv)

    # Calculate mean and standard deviation of training and validation scores
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Learning Curve for SVM")
    plt.show()


def tune_train_evaluate_svm(X, y, X_test, y_test, param_grid, cv=5):
    '''
    Performs hyperparameter tuning, training, and evaluation of an SVM classifier.

    Parameters
    ----------
    X : array-like
        Training data features.
    y : array-like
        Training data labels.
    X_test : array-like
        Test data features.
    y_test : array-like
        Test data labels.
    param_grid : dict
        Grid of 'C' and 'gamma' values for hyperparameter tuning.
    cv : int, optional
        Number of cross-validation folds. Default is 5.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics (accuracy, precision, recall, F1 score)
        and the best hyperparameters (C, gamma) found during tuning.
    '''

    # 1. Hyperparameter Tuning: Cross-validation to find the best C and gamma
    cv_results = cross_validate_svm(X, y, param_grid, cv, scoring='recall')

    # 2. Train the SVM Classifier with Best Hyperparameters
    clf = svm.SVC(
        kernel='rbf', C=cv_results['best_params']['C'], gamma=cv_results['best_params']['gamma'])
    clf.fit(X, y)

    # 3. Evaluate Training Performance with a Learning Curve
    plot_learning_curve(clf, X, y, cv)

    # 4. Evaluate Test Set Performance
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    # Return results as a dictionary
    return {
        'recall': recall,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'best_params': cv_results['best_params']
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


def train_evaluate_final_svm(X_train, y_train, X_test, y_test, best_params):
    '''
    Train and evaluate an SVM model using the best hyperparameters.

    This function takes in training and test datasets along with the optimal 
    hyperparameters (C and gamma) to train an SVM model with an RBF kernel. 
    It performs cross-validation on the training set, evaluates performance 
    on the test set, and computes key performance metrics such as accuracy, 
    precision, recall, and F1 score. It also visualizes the learning curve, 
    confusion matrix, and ROC curve for the trained model.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        The training input samples.

    y_train : array-like, shape (n_samples,)
        The target values for training.

    X_test : array-like, shape (n_samples, n_features)
        The test input samples.

    y_test : array-like, shape (n_samples,)
        The true target values for the test set.

    best_params : dict
        A dictionary containing the best hyperparameters for the SVM model. 
        Expected keys are:
            - 'C': Regularization parameter (float)
            - 'gamma': Kernel coefficient (float)

    Returns
    -------
    model : object
        The trained SVM model (fitted estimator).

    metrics : dict
        A dictionary containing the evaluation metrics:
            - 'accuracy': Test set accuracy (float)
            - 'precision': Test set precision (float)
            - 'recall': Test set recall (float)
            - 'f1': Test set F1 score (float)
            - 'best_params': Best hyperparameters used in the model (dict)
    '''

    # Creating new SVM model with the best parameters
    clf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])

    # Cross-validation on the training set
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print("Average Cross-Validation Score:", scores.mean())
    print("Standard Deviation of Scores:", scores.std())

    # Training the new SVM model
    model = clf.fit(X_train, y_train)

    # Test set evaluation
    test_accuracy = model.score(X_test, y_test)
    print("Test Set Accuracy:", test_accuracy)

    # Evaluate Training Performance with a Learning Curve
    plot_learning_curve(clf, X_train, y_train, cv=10)

    # Predicting on the test data
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    # Compute and plot the ROC curve
    plot_roc_curve(X_test, y_test, clf)

    # Return model and performance metrics
    metrics = {
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'best_params': best_params
    }

    return model, metrics


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


def get_adjacent_values(arr, best_value):
    idx = np.where(np.isclose(arr, best_value))[0][0]
    # Get previous, current, and next values safely
    prev_value = arr[idx - 1] if idx > 0 else arr[idx]
    next_value = arr[idx + 1] if idx < len(arr) - 1 else arr[idx]
    return prev_value, arr[idx], next_value


def evaluate_svm_with_feature_selection(data, feature_list):
    """
    Evaluates an SVM model using iterative cross-validation and hyperparameter tuning
    for a given feature configuration. This function performs three stages of grid search
    to find the best C and gamma parameters, and then trains and evaluates the model.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing features and the target variable 'AvalDay'.
    feature_list : list
        A list of feature column names to be used for training the model.

    Returns
    -------
    feature_list : list
        The input list of features used for training the model.
    classifier : sklearn.svm.SVC
        The trained SVM classifier with the best parameters.
    results : dict
        A dictionary containing evaluation metrics of the trained model including:
        - 'accuracy'
        - 'precision'
        - 'recall'
        - 'f1'
        - 'best_params': Best C and gamma values from the final cross-validation.
    """

    # Add target variable to the feature list
    feature_with_target = feature_list + ['AvalDay']

    # Data preprocessing: filter relevant features and drop missing values
    clean_data = data[feature_with_target].dropna()

    # Extract features and target variable
    X = clean_data[feature_list]
    y = clean_data['AvalDay']

    # Step 1: Apply NearMiss undersampling to balance the dataset
    X_resampled, y_resampled = undersampling_nearmiss(X, y)

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # Step 3: Initial broad search for hyperparameters C and gamma
    initial_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001]
    }
    result_1iter = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, initial_param_grid, cv=5
    )

    # Step 4: Refining the search space based on the best parameters from the first iteration
    best_params = result_1iter['best_params']
    refined_C_range = np.linspace(
        best_params['C'] * 0.1, best_params['C'] * 10, 20)
    refined_gamma_range = np.linspace(
        best_params['gamma'] * 0.1, best_params['gamma'] * 10, 20)

    refined_param_grid = {
        'C': refined_C_range,
        'gamma': refined_gamma_range
    }
    result_2iter = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, refined_param_grid, cv=10
    )

    # Step 5: Fine-tuning around the best parameters found in the second iteration
    best_params2 = result_2iter['best_params']
    C_adj_values = get_adjacent_values(
        refined_param_grid['C'], best_params2['C'])
    gamma_adj_values = get_adjacent_values(
        refined_param_grid['gamma'], best_params2['gamma'])

    final_C_range = np.linspace(C_adj_values[0], C_adj_values[-1], 20)
    final_gamma_range = np.linspace(
        gamma_adj_values[0], gamma_adj_values[-1], 20)

    final_param_grid = {
        'C': final_C_range,
        'gamma': final_gamma_range
    }
    result_3iter = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, final_param_grid, cv=10
    )

    # Step 6: Train the final model with the best hyperparameters and evaluate it
    classifier, evaluation_metrics = train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, result_3iter['best_params']
    )

    return feature_list, classifier, evaluation_metrics


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
    res_rand = tune_train_evaluate_svm(
        X_rand_train, y_rand_train, X_rand_test, y_rand_test)

    # 2. Random undersampling N days before
    X_rand_10d_train, X_rand_10d_test, y_rand_10d_train, y_rand_10d_test = train_test_split(
        X_rand_10d, y_rand_10d, test_size=0.25, random_state=42)
    res_rand_10d = tune_train_evaluate_svm(
        X_rand_10d_train, y_rand_10d_train, X_rand_10d_test, y_rand_10d_test)

    # 3. Nearmiss undersampling
    X_nm_train, X_nm_test, y_nm_train, y_nm_test = train_test_split(
        X_nm, y_nm, test_size=0.25, random_state=42)
    res_nm = tune_train_evaluate_svm(
        X_nm_train, y_nm_train, X_nm_test, y_nm_test)

    # 4. Random oversampling
    res_ros = tune_train_evaluate_svm(X_ros, y_ros, X_test, y_test)

    # 5. SMOTE oversampling
    res_sm = tune_train_evaluate_svm(X_sm, y_sm, X_test, y_test)

    # 6. adasyn oversampling
    res_adas = tune_train_evaluate_svm(X_adas, y_adas, X_test, y_test)

    # 7. SVMSMOTE oversampling
    res_svmsm = tune_train_evaluate_svm(X_svmsm, y_svmsm, X_test, y_test)

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

    classifier_nm = train_evaluate_final_svm(
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

    res_nm_new = tune_train_evaluate_svm(
        X_train_new, y_train_new, X_test_new, y_test_new)

    res_nm_new_list = []

    # Add each result to the list with the sampling method as an identifier
    res_nm_new_list.append(
        {'Run': '1', **res_nm_new})

    classifier_nm_new = train_evaluate_final_svm(
        X_train_new, y_train_new, X_test_new, y_test_new, res_nm_new)

    # Calculate evaluation metrics
    y_predict = classifier_nm_new.predict(X_test_new)
    accuracy = accuracy_score(y_test_new, y_predict)
    precision = precision_score(y_test_new, y_predict)
    recall = recall_score(y_test_new, y_predict)
    f1 = f1_score(y_test_new, y_predict)

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

    f1 = ['HSnum']
    res1 = evaluate_svm_with_feature_selection(mod1, f1)

    f2 = f1 + ['HNnum']
    res2 = evaluate_svm_with_feature_selection(mod1, f2)

    f3 = f2 + ['HN_2d']
    res3 = evaluate_svm_with_feature_selection(mod1, f3)

    f4 = f3 + ['HN_3d']
    res4 = evaluate_svm_with_feature_selection(mod1, f4)

    f5 = f4 + ['HN_5d']
    res5 = evaluate_svm_with_feature_selection(mod1, f5)

    f6 = f5 + ['Precip_1d']
    res6 = evaluate_svm_with_feature_selection(mod1, f6)

    f7 = f6 + ['Precip_2d']
    res7 = evaluate_svm_with_feature_selection(mod1, f7)

    f8 = f7 + ['Precip_3d']
    res8 = evaluate_svm_with_feature_selection(mod1, f8)

    f9 = f8 + ['Precip_5d']
    res9 = evaluate_svm_with_feature_selection(mod1, f9)

    f10 = f9 + ['FreshSWE']
    res10 = evaluate_svm_with_feature_selection(mod1, f10)

    f11 = f10 + ['SeasonalSWE_cum']
    res11 = evaluate_svm_with_feature_selection(mod1, f11)

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

    # RESULTS: the best configuration based on Recall is:
    # f9: HSnum, HNnum, HN_2d, HN_3d, HN_5d,
    #     Precip_1d, Precip_2d, Precip_3d, Precip_5d

    # ....... 2. SNOW LOAD DUE WIND DRIFT ...........................

    wd4 = f9 + ['SnowDrift_1d']
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
    plt.ylabel('Precision Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_wd, common_path / 'config_snowdrift_features.csv')

    # RESULTS: the wind drift does not provide improvements to the model.

    # ....... 3. PAST AVALANCHE ACTIVITY ...........................

    a10 = f9 + ['AvalDay_2d']
    res_a10 = evaluate_svm_with_feature_selection(mod1, a10)

    a11 = a10 + ['AvalDay_3d']
    res_a11 = evaluate_svm_with_feature_selection(mod1, a11)

    a12 = a11 + ['AvalDay_5d']
    res_a12 = evaluate_svm_with_feature_selection(mod1, a12)

    results_features = [res9, res_a10, res_a11, res_a12]

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
    plt.title('Scores for Snow Drift features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Precision Score')
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

    # ....... 4. SNOW TEMPERATURE AS  ...........................

    ts13 = a12 + ['TH01G']
    res_ts13 = evaluate_svm_with_feature_selection(mod1, ts13)

    ts14 = ts13 + ['Tsnow_delta_1d']
    res_ts14 = evaluate_svm_with_feature_selection(mod1, ts14)

    ts15 = ts14 + ['Tsnow_delta_2d']
    res_ts15 = evaluate_svm_with_feature_selection(mod1, ts15)

    ts16 = ts15 + ['Tsnow_delta_3d']
    res_ts16 = evaluate_svm_with_feature_selection(mod1, ts16)

    ts17 = ts16 + ['Tsnow_delta_5d']
    res_ts17 = evaluate_svm_with_feature_selection(mod1, ts17)

    results_features = [res_a12, res_ts14,
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
    plt.title('Scores for Snow Drift features in different configuration')
    plt.xlabel('Feature Configuration')
    plt.ylabel('Precision Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

    save_outputfile(df_res_ts, common_path /
                    'config_snowtemp_features.csv')

    # if __name__ == '__main__':
    #     main()
