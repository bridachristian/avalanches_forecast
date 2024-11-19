import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import undersampling_random, undersampling_random_timelimited, undersampling_nearmiss
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.utils import get_adjacent_values, save_outputfile


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


def plot_confusion_matrix(y_test, y_pred):
    """
    Plots and returns the confusion matrix for model evaluation.

    This function generates and visualizes a confusion matrix using a heatmap, which helps in evaluating
    the performance of a classification model. It compares the true class labels (`y_test`) with the predicted
    labels (`y_pred`) to show how well the model is performing in terms of both the correct and incorrect classifications.

    Args:
        y_test (array-like): True class labels (ground truth values).
        y_pred (array-like): Predicted class labels from the model.

    Returns:
        np.ndarray: The confusion matrix as a 2D array, where each element [i, j] represents the number of samples
                    with true label `i` and predicted label `j`.

    Notes:
        - The confusion matrix is a square matrix where the rows represent the true labels and the columns represent the predicted labels.
        - The diagonal elements represent the number of correct predictions for each class.
        - The off-diagonal elements represent misclassifications where the true label is different from the predicted label.

    Example:
        cm = plot_confusion_matrix(y_test, y_pred)
    """
    # Step 1: Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Step 2: Visualize the confusion matrix using a heatmap
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
    """
    Plots the Receiver Operating Characteristic (ROC) curve for binary classification models.

    This function computes the ROC curve and calculates the Area Under the Curve (AUC) for a given model. 
    It then plots the ROC curve, which shows the trade-off between the true positive rate (TPR) and the false positive rate (FPR) 
    for different threshold values.

    Args:
        X_test (array-like): The test data features, used for generating the predicted probabilities or decision function scores.
        y_test (array-like): The true binary class labels corresponding to the test data.
        clf (object): A trained classifier with a `decision_function` method (e.g., `SVC`, `LogisticRegression`, etc.).

    Returns:
        None: The function displays a plot of the ROC curve and AUC score but does not return any value.

    Notes:
        - The classifier `clf` must have a `decision_function` method that outputs continuous scores.
        - The function assumes a binary classification task, where the labels in `y_test` are either 0 or 1.
        - The ROC curve is plotted with a blue line, and the diagonal line (representing random chance) is plotted in red.

    Example:
        plot_roc_curve(X_test, y_test, clf)
    """
    # Compute ROC curve and ROC area
    y_score = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    # Diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def permutation_ranking(classifier, X_test, y_test):
    """
    Computes and visualizes the permutation feature importance for a given classifier.

    This function computes feature importance by measuring the decrease in model performance (accuracy) 
    when each feature is randomly permuted. The greater the decrease in performance, the more important the feature is.

    Args:
        classifier (object): A trained classifier that implements the `predict` method (e.g., `SVC`, `RandomForestClassifier`, etc.).
        X_test (pd.DataFrame): The test features used to compute permutation importance. It must be a DataFrame with column names.
        y_test (pd.Series or array-like): The true class labels for the test data.

    Returns:
        pd.DataFrame: A DataFrame containing the features sorted by importance, including their mean and standard deviation of importance scores.

    Notes:
        - The classifier should already be trained before calling this function.
        - The function uses `permutation_importance` from `sklearn`, which is available for classifiers with the `predict` method.
        - The importance scores are based on the mean decrease in accuracy when features are permuted.

    Example:
        feature_importance_df = permutation_ranking(classifier, X_test, y_test)
    """

    # Compute permutation importance
    perm_importance = permutation_importance(
        classifier, X_test, y_test, n_repeats=30, random_state=42)

    # Sort features by mean importance score
    sorted_idx = perm_importance.importances_mean.argsort()

    # Create DataFrame with sorted feature importance
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
        best_params['C'] * 0.1, best_params['C'] * 10, 10)
    refined_gamma_range = np.linspace(
        best_params['gamma'] * 0.1, best_params['gamma'] * 10, 10)

    refined_param_grid = {
        'C': refined_C_range,
        'gamma': refined_gamma_range
    }
    result_2iter = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, refined_param_grid, cv=5
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
