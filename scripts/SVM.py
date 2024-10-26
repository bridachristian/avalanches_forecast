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


def load_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    data = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    data['DataRilievo'] = pd.to_datetime(
        data['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index
    data.set_index('DataRilievo', inplace=True)

    return data


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

    # X = mod1_undersampled[['HN72h']].values
    X = mod1_undersampled.drop(columns=['Stagione', 'AvalDay']).values
    # X = mod1_undersampled[['HSdiff72h']].values
    y = mod1_undersampled['AvalDay'].values

    # # Normalizzare i dati
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- CROSS VALIDATION ---
    problem = svm_problem(y, X)

    # Definisci la griglia di parametri
    C_values = np.logspace(-3, 3, 7)  # ad esempio, 0.01, 0.1, 1, 10, 100
    gamma_values = np.logspace(-3, 3, 7)  # ad esempio, 0.001, 0.01, 0.1, 1, 10

    best_accuracy = 0
    best_C = None
    best_gamma = None

    # Cerca la combinazione migliore
    for C in C_values:
        for gamma in gamma_values:
            # "-v 5" esegue una 5-fold cross-validation
            param_str = f'-t 2 -c {C} -g {gamma} -v 5'
            param = svm_parameter(param_str)

            # Esegui la cross-validation
            accuracy = svm_train(problem, param)

            # Aggiorna i migliori parametri se l'accuratezza è migliorata
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
                best_gamma = gamma

    print("Best C:", best_C)
    print("Best gamma:", best_gamma)
    print("Best cross-validation accuracy:", best_accuracy)

    # --- SPLIT TRAIN AND TEST DATASET ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # --- MODEL ---
    problem = svm_problem(y, X)

    param_string = f'-t 2 -c {best_C} -g {best_gamma}'

    param = svm_parameter(param_string)

    model = svm_train(problem, param)

    # --- EVALUATION ---
    predicted_labels, accuracy, decision_values = svm_predict(
        y_test, X_test, model)

    print("Predicted labels:", predicted_labels)
    print("Accuracy:", accuracy)
    print("Decision values:", decision_values)

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
