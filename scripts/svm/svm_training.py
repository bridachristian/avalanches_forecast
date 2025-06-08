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
                             f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import undersampling_random, undersampling_random_timelimited, undersampling_nearmiss
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.utils import (plot_decision_boundary, get_adjacent_values,
                               detect_grid_type, plot_threshold_scoring)
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                                    plot_roc_curve)


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
    print(f"Best Parameters: {best_params}")
    print(f"Average Cross-Validation Score: {cv_scores.mean():.4f}")
    print(f"Standard Deviation of Scores: {cv_scores.std():.4f}")

    # Extract grid search results and create a heatmap
    results = grid.cv_results_
    scores = results['mean_test_score']

    # Reshape scores for heatmap visualization
    c_values = param_grid['C']
    gamma_values = param_grid['gamma']
    scores_matrix = scores.reshape(len(c_values), len(gamma_values))

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_matrix, annot=False, fmt=".3f",
                xticklabels=gamma_values, yticklabels=c_values, cmap="viridis")
    plt.title(f'{title} - {scoring}')
    plt.xlabel("Gamma")
    plt.ylabel("C")

    # Mark the best (C, gamma) combination with a red dot
    best_C = best_params["C"]
    best_gamma = best_params["gamma"]
    best_C_index = np.where(np.array(c_values) == best_C)[0][0]
    best_gamma_index = np.where(np.array(gamma_values) == best_gamma)[0][0]
    plt.scatter(best_gamma_index + 0.5, best_C_index + 0.5,
                color='red', s=100, edgecolors='black', label="Best (C, gamma)")
    plt.legend(loc="upper right")

    # Show the plot
    plt.show()

    # Converti dimensioni in pollici (tesi: 15×12 cm)
    fig_width = 15 / 2.54
    fig_height = 12 / 2.54

    # Meshgrid per i valori di gamma (x) e C (y)
    X, Y = np.meshgrid(gamma_values, c_values)

    # Crea figura
    plt.figure(figsize=(fig_width, fig_height))

    # Contour plot colorato senza linee di contorno
    contour = plt.contourf(X, Y, scores_matrix, levels=20, cmap="cividis")
    cbar = plt.colorbar(contour)
    cbar.set_label("F1-Score", fontsize=11)

    # Punto migliore evidenziato con marker rosso
    best_C = best_params["C"]
    best_gamma = best_params["gamma"]
    plt.plot(best_gamma, best_C, 'o', color='red', markersize=8,
             markeredgecolor='black', label="Best hyperparameters: $C = 2$, $\gamma =  0.5$")

    # Scala logaritmica per entrambi gli assi
    plt.xscale("log")
    plt.yscale("log")

    # Etichette e stile
    plt.xlabel("Gamma (log scale)", fontsize=11)
    plt.ylabel("C (log scale)", fontsize=11)
    plt.title(f'{title} – Grid Search (F1-Score)', fontsize=12, weight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(loc="lower right", fontsize=9)

    # Ottimizza il layout per la stampa
    plt.tight_layout()
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


def tune_train_evaluate_svm(X, y, X_test, y_test, param_grid, resampling_method, cv=5):
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
    from sklearn.model_selection import StratifiedKFold

    # 🔒 Safe Stratified CV Setup
    if isinstance(cv, int):
        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = cv

    print(
        f"[INFO] Resampled label distribution:\n{np.unique(y, return_counts=True)}")

    # 1. Hyperparameter tuning (Stratified)
    cv_results_coarse = cross_validate_svm(
        X, y, param_grid, cv=cv_strategy,
        title=f'1st run - CV scores for {resampling_method}', scoring='f1_macro'
    )

    # # 1. Hyperparameter Tuning: Cross-validation to find the best C and gamma
    # cv_results_coarse = cross_validate_svm(
    #     X, y, param_grid, cv, title=f'1st run - CV scores for {resampling_method} ', scoring='f1_macro')

    # optuna_results_coarse = tune_svm_with_optuna(
    #     X, y, n_trials=100, cv=10, scoring="f1_macro")

    # Create a finer grid based on adiacent values fo coarse grid

    # C_fine = get_adjacent_values(
    #     param_grid['C'], cv_results_coarse['best_params']['C'])
    # gamma_fine = get_adjacent_values(
    #     param_grid['gamma'], cv_results_coarse['best_params']['gamma'])

    # finer_param_grid = {
    #     # 20 values between the adjacent C values
    #     'C': np.linspace(C_fine[0], C_fine[-1], 21, dtype=np.float64),
    #     # 20 values between the adjacent gamma values
    #     'gamma': np.linspace(gamma_fine[0], gamma_fine[-1], 21, dtype=np.float64)
    # }

    # cv_results = cross_validate_svm(
    #     X, y, finer_param_grid, cv, title=f'2nd run - CV scores for {resampling_method} ', scoring='f1_macro')

    # 2. Train the SVM Classifier with Best Hyperparameters
    clf = svm.SVC(
        kernel='rbf', C=cv_results_coarse['best_params']['C'], gamma=cv_results_coarse['best_params']['gamma'])
    clf.fit(X, y)

    # 3. Evaluate Training Performance with a Learning Curve
    plot_learning_curve(clf, X, y, title=f'{resampling_method}', cv=cv)
    plot_threshold_scoring(X, y, X_test, y_test, clf)

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
    mcc = matthews_corrcoef(y_test, y_pred)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')

    # Return results as a dictionary
    return {
        'recall': recall,
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'MCC': mcc,
        'best_params': cv_results_coarse['best_params']
    }


def train_evaluate_final_svm(X_train, y_train, X_test, y_test, best_params, display_plot=True):
    '''
    Train and evaluate an SVM model using the best hyperparameters.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,)
    X_test : array-like, shape (n_samples, n_features)
    y_test : array-like, shape (n_samples,)
    best_params : dict
        Expected keys: 'C', 'gamma'
    display_plot : bool
        If True, display plots

    Returns
    -------
    model : Trained SVM model
    metrics : dict
        Evaluation metrics
    '''
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import warnings

    # Validate best_params
    if not all(k in best_params for k in ['C', 'gamma']):
        raise ValueError("best_params must contain keys 'C' and 'gamma'")

    # Create SVM with best hyperparameters
    clf = svm.SVC(
        kernel='rbf', C=best_params['C'], gamma=best_params['gamma'],
        probability=True, class_weight='balanced')

    # Use StratifiedKFold for balanced CV splits
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Cross-validation on training set
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scores = cross_val_score(clf, X_train, y_train, cv=cv)

    print(f"Average Cross-Validation Score: {scores.mean():.4f}")
    print(f"Standard Deviation of Scores: {scores.std():.4f}")

    # Train model on full training data
    model = clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Test Set Accuracy: {accuracy:.4f}")

    # Plots
    if display_plot:
        plot_learning_curve(clf, X_train, y_train, cv=cv,
                            title='Learning Curve - Final SVM')
        plot_threshold_scoring(X_train, y_train, X_test, y_test, clf)
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(X_test, y_test, clf)

    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'MCC': mcc,
        'best_params': best_params
    }
