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
from scripts.svm.undersampling_methods import (undersampling_random, undersampling_random_timelimited, undersampling_nearmiss,
                                               undersampling_cnn, undersampling_enn, undersampling_clustercentroids, undersampling_tomeklinks)
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.utils import get_adjacent_values, save_outputfile, remove_correlated_features
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, matthews_corrcoef
from scripts.svm.feature_engineering import transform_features


def plot_learning_curve(clf, X, y, title, cv=5, scoring='f1'):
    """
    Plots the learning curve for the given classifier.

    Parameters:
    - clf: Trained classifier (e.g., SVM model)
    - X: Feature data
    - y: Target labels
    - title: Title of the plot
    - cv: Number of cross-validation folds (default is 5)
    - scoring: Scoring metric (e.g., 'accuracy', 'f1', or a custom scorer)

    Returns:
    - None (displays a plot)
    """
    if scoring == 'mcc':
        scorer = make_scorer(matthews_corrcoef)
    else:
        scorer = scoring

    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, cv=cv, scoring=scorer)

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel(f"{scoring} Score")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.title(f'Learning Curve - {title}')
    plt.grid(True)
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


def permutation_ranking(classifier, X, y):
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
    # perm_importance = permutation_importance(
    #     classifier, X, y, n_repeats=100, random_state=42, scoring='f1_macro', n_jobs=-1)
    perm_importance = permutation_importance(
        classifier, X, y, n_repeats=100, random_state=42, scoring='recall_macro', n_jobs=-1)

    # Sort features by mean importance score
    sorted_idx = perm_importance.importances_mean.argsort()

    # Create DataFrame with sorted feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns[sorted_idx],
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
    plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
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

    feature_plus = feature_list + ['AvalDay']
    data_clean = data[feature_plus]
    data_clean = data_clean.dropna()
    data_transformed = transform_features(data_clean.copy())

    X = data_transformed[feature_list]
    y = data_transformed['AvalDay']

    features_to_remove = remove_correlated_features(X, y)

    X = X.drop(columns=features_to_remove)

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

    X_resampled, y_resampled = undersampling_clustercentroids(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42)

    # scaler = MinMaxScaler()
    # X_train = pd.DataFrame(scaler.fit_transform(
    #     X_train), columns=X_train.columns, index=X_train.index)
    # X_test = pd.DataFrame(scaler.transform(
    #     X_test), columns=X_test.columns, index=X_test.index)

    result = tune_train_evaluate_svm(
        X_train, y_train, X_test, y_test, param_grid,
        resampling_method='Cluster Centroids')

    # Step 6: Train the final model with the best hyperparameters and evaluate it
    classifier, evaluation_metrics = train_evaluate_final_svm(
        X_train, y_train, X_test, y_test, result['best_params']
    )

    return feature_list, classifier, evaluation_metrics
