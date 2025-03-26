import optuna
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.inspection import permutation_importance
from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import undersampling_random, undersampling_random_timelimited, undersampling_nearmiss
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.utils import plot_decision_boundary, get_adjacent_values, detect_grid_type, plot_threshold_scoring


def randomsearch_cross_validate_svm(X, y, param_distributions, n_iter=50, cv=5, scoring='f1_macro', random_state=42):
    """
    Performs coarse hyperparameter tuning and cross-validation for an SVM model using RandomizedSearchCV.

    Parameters:
        X (array-like): Training data features.
        y (array-like): Training data labels.
        param_distributions (dict): Dictionary with parameter names (`C` and `gamma`) as keys and distributions/ranges of parameter values to sample from.
        n_iter (int): Number of parameter settings sampled (default is 20).
        cv (int): Number of cross-validation folds (default is 5).
        scoring (str): Scoring metric for RandomizedSearchCV (default is 'f1_macro').
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        dict: Contains the best parameters, cross-validation scores, and the best model.
    """
    # Initialize RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=svm.SVC(kernel='rbf'),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=3,
        random_state=random_state
    )
    random_search.fit(X, y)

    # Extract the best parameters and model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

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


def cross_validate_svm(X, y, param_grid, cv=5, title='CV scores', scoring='f1_macro'):
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
                        cv=cv, scoring=scoring, verbose=2, n_jobs=-1)
    grid.fit(X, y)

    # Extract the best parameters and model
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    # Perform cross-validation using the best model
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)
    print("Best Parameters:", best_params)
    print("Average Cross-Validation Score:", cv_scores.mean())
    print("Standard Deviation of Scores:", cv_scores.std())

    # Extract grid search results and create a heatmap
    results = grid.cv_results_
    scores = results['mean_test_score']

    # Reshape scores for heatmap visualization
    c_values = param_grid['C']
    # grid_type_C = detect_grid_type(c_values)

    gamma_values = param_grid['gamma']
    scores_matrix = scores.reshape(len(c_values), len(gamma_values))

    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_matrix, annot=False, fmt=".3f",
                xticklabels=gamma_values, yticklabels=c_values, cmap="viridis")
    plt.title(f'{title} - {scoring}')
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.show()

    # Return the best model and evaluation metrics
    return {
        'best_model': best_model,
        'best_params': best_params,
        'cv_mean_score': cv_scores.mean(),
        'cv_std_score': cv_scores.std()
    }


def tune_svm_with_optuna(X, y, n_trials=50, cv=5, scoring="accuracy"):
    """
    Performs hyperparameter tuning for an SVM with RBF kernel using Optuna.

    Parameters:
    - X: Features (numpy array or DataFrame)
    - y: Target labels (numpy array or Series)
    - n_trials: Number of trials to run (default: 50)
    - cv: Number of cross-validation folds (default: 5)
    - scoring: Metric to optimize (default: "accuracy")

    Returns:
    - best_model: Trained SVM model with best parameters
    - best_params: Dictionary of best hyperparameters
    - best_score: Best cross-validation score
    """

    def objective(trial):
        """Objective function to optimize SVM hyperparameters."""
        # Log-scale search for C and gamma
        C = trial.suggest_loguniform("C", 1e-4, 1e4)
        gamma = trial.suggest_loguniform("gamma", 1e-4, 1e4)

        # Create SVM model
        model = SVC(kernel='rbf', C=C, gamma=gamma)

        # Perform cross-validation
        score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
        return score  # Optuna maximizes this score

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Get best hyperparameters
    best_params = study.best_params
    best_score = study.best_value

    # Train the final model with best parameters
    best_model = SVC(kernel='rbf', **best_params)
    best_model.fit(X, y)  # Fit on entire dataset

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f}")

    return best_model, best_params, best_score


def tune_train_evaluate_svm(X, y, X_test, y_test, param_grid, resampling_method, cv=10):
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
    from scripts.svm.evaluation import plot_learning_curve

    # 1. Hyperparameter Tuning: Cross-validation to find the best C and gamma
    cv_results_coarse = cross_validate_svm(
        X, y, param_grid, cv, title=f'1st run - CV scores for {resampling_method} ', scoring='f1_macro')

    # optuna_results_coarse = tune_svm_with_optuna(
    #     X, y, n_trials=100, cv=10, scoring="f1_macro")

    # Create a finer grid based on adiacent values fo coarse grid

    C_fine = get_adjacent_values(
        param_grid['C'], cv_results_coarse['best_params']['C'])
    gamma_fine = get_adjacent_values(
        param_grid['gamma'], cv_results_coarse['best_params']['gamma'])

    finer_param_grid = {
        # 20 values between the adjacent C values
        'C': np.linspace(C_fine[0], C_fine[-1], 21, dtype=np.float64),
        # 20 values between the adjacent gamma values
        'gamma': np.linspace(gamma_fine[0], gamma_fine[-1], 21, dtype=np.float64)
    }

    cv_results = cross_validate_svm(
        X, y, finer_param_grid, cv, title=f'2nd run - CV scores for {resampling_method} ', scoring='f1_macro')

    # 2. Train the SVM Classifier with Best Hyperparameters
    clf = svm.SVC(
        kernel='rbf', C=cv_results['best_params']['C'], gamma=cv_results['best_params']['gamma'])
    clf.fit(X, y)

    # 3. Evaluate Training Performance with a Learning Curve
    # plot_learning_curve(clf, X, y, title=f'{resampling_method}', cv=cv)
    # plot_threshold_scoring(X, y, X_test, y_test, clf)

    if X.shape[1] == 2:
        plot_decision_boundary(X, y, clf,
                               title=f'{resampling_method}', palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    # 4. Evaluate Test Set Performance
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

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
    from scripts.svm.evaluation import plot_learning_curve, plot_confusion_matrix, plot_confusion_matrix, plot_roc_curve

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
    plot_learning_curve(clf, X_train, y_train, cv=10,
                        title='Learning Curve Final SVM')

    # Predicting on the test data
    y_pred = model.predict(X_test)

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

    # Compute and plot the ROC curve
    # plot_roc_curve(X_test, y_test, clf)

    # Return model and performance metrics
    metrics = {
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
        'best_params': best_params
    }

    return model, metrics
