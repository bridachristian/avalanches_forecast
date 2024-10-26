# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:33:33 2024

@author: Christian
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import svm
import numpy as np
# from libsvm.svmutil import *
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_train, svm_problem, svm_parameter, svm_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


def load_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    data = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    data['DataRilievo'] = pd.to_datetime(
        data['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index
    data.set_index('DataRilievo', inplace=True)

    return data

# Function to normalize data


def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Function to find best C and gamma values using cross-validation


def cross_validate_svm(y, X, C_values, gamma_values, second_order_factor=0.5):
    problem = svm_problem(y, X)
    best_accuracy, best_C, best_gamma = 0, None, None
    refined_search_needed = True

    # Initial cross-validation loop with range adjustment
    while refined_search_needed:
        refined_search_needed = False  # Reset flag
        best_accuracy, best_C, best_gamma = 0, None, None  # Reset best values for each run

        for C in C_values:
            for gamma in gamma_values:
                param_str = f'-t 2 -c {C} -g {gamma} -v 5'
                param = svm_parameter(param_str)
                accuracy = svm_train(problem, param)

                if accuracy > best_accuracy:
                    best_accuracy, best_C, best_gamma = accuracy, C, gamma

        # Adjust C and gamma ranges if at the bounds, and flag further refinement if needed
        if best_C == max(C_values):
            C_values = np.logspace(np.log10(min(C_values)), np.log10(
                max(C_values)) + 1, len(C_values) + 1)
            refined_search_needed = True
        elif best_C == min(C_values):
            C_values = np.logspace(
                np.log10(min(C_values)) - 1, np.log10(max(C_values)), len(C_values) + 1)
            refined_search_needed = True

        if best_gamma == max(gamma_values):
            gamma_values = np.logspace(np.log10(min(gamma_values)), np.log10(
                max(gamma_values)) + 1, len(gamma_values) + 1)
            refined_search_needed = True
        elif best_gamma == min(gamma_values):
            gamma_values = np.logspace(np.log10(
                min(gamma_values)) - 1, np.log10(max(gamma_values)), len(gamma_values) + 1)
            refined_search_needed = True

    # Second-order refined cross-validation search
    refined_C_values = np.logspace(
        np.log10(best_C) - second_order_factor, np.log10(best_C) + second_order_factor, num=5
    )
    refined_gamma_values = np.logspace(
        np.log10(best_gamma) - second_order_factor, np.log10(best_gamma) + second_order_factor, num=5
    )

    for C in refined_C_values:
        for gamma in refined_gamma_values:
            param_str = f'-t 2 -c {C} -g {gamma} -v 5'
            param = svm_parameter(param_str)
            accuracy = svm_train(problem, param)

            if accuracy > best_accuracy:
                best_accuracy, best_C, best_gamma = accuracy, C, gamma

    return best_C, best_gamma, best_accuracy

# Function to train and test the model


def train_and_evaluate(X_train, y_train, X_test, y_test, best_C, best_gamma):
    problem_train = svm_problem(y_train, X_train)
    param_string = f'-t 2 -c {best_C} -g {best_gamma}'
    param = svm_parameter(param_string)
    model = svm_train(problem_train, param)

    predicted_labels, accuracy, decision_values = svm_predict(
        y_test, X_test, model, options="b")
    return accuracy[0], predicted_labels, decision_values


def main():
    # --- PATHS ---
    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_undersampling.csv'

    # output_filepath = Path(
    #     'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_undersampling.csv')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1_undersampled = load_data(filepath)
    print(mod1_undersampled.dtypes)  # For initial data type inspection

    # --- SPLIT FEATURES AND TARGET ---

    # Specify target
    y = mod1_undersampled['AvalDay'].values

    # Specify features
    # X = mod1_undersampled[['HN72h']].values
    # X = mod1_undersampled.drop(columns=['Stagione', 'AvalDay']).values
    features = ['HN72h']  # Example feature set
    X = mod1_undersampled[features].values

    # Normalize data
    X = normalize_data(X)

    # Define parameter grid
    C_values = np.logspace(-3, 3, 7)
    gamma_values = np.logspace(-3, 3, 7)

    # Cross-validation to find best parameters
    best_C, best_gamma, best_accuracy = cross_validate_svm(
        y, X, C_values, gamma_values)
    print(
        f"Best C: {best_C}, Best gamma: {best_gamma}, Best CV accuracy: {best_accuracy}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Train and evaluate
    test_accuracy, predicted_labels, decision_values = train_and_evaluate(
        X_train, y_train, X_test, y_test, best_C, best_gamma)

    # Output performance
    print("Test accuracy:", test_accuracy)
    print("Predicted labels:", predicted_labels)
    print("Decision values:", decision_values)

    # 1. Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # 2. Print or visualize the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # For a better visualization, plot the confusion matrix as a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[
                'Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f'{features}')
    plt.show()

    # 3. Additional evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels))

    print("Accuracy Score:", accuracy_score(y_test, predicted_labels))

    # --- PLOT ---

    # # Crea una griglia di punti per tracciare il confine di decisione

    # ##############################################################################################

    # # Definire il modello SVM
    # svm_model = SVC(kernel='rbf', random_state=42)

    # # Eseguire la cross-validation
    # cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)

    # # Mostra i risultati
    # print("Cross-Validation Accuracy Scores:", cv_scores)
    # print("Mean CV Accuracy:", cv_scores.mean())

    # # Create a svm Classifier
    # clf = svm.SVC(kernel='rbf')  # Linear Kernel

    # # Train the model using the training sets
    # clf.fit(X_train, y_train)

    # # Predict the response for test dataset
    # y_pred = clf.predict(X_test)

    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # # Model Precision: what percentage of positive tuples are labeled as such?
    # print("Precision:", metrics.precision_score(y_test, y_pred))

    # # Model Recall: what percentage of positive tuples are labelled as such?
    # print("Recall:", metrics.recall_score(y_test, y_pred))
