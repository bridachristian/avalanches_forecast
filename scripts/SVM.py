# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:33:33 2024

@author: Christian
"""
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
# from libsvm.svmutil import *
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from libsvm.svmutil import svm_train, svm_problem, svm_parameter


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

    X = mod1_undersampled[['HN72h']].values
    y = mod1_undersampled['AvalDay'].values

    # Normalizzare i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- CROSS VALIDATION ---
    problem = svm_problem(y, X_scaled)

    # Definisci i parametri SVM con cross-validation
    # C è il parametro di penalità e `-v` specifica il numero di fold per la cross-validation
    # `-t 0` indica un kernel lineare, `-v 5` indica 5-fold CV
    param = svm_parameter('-t 3 -c 0.01 -v 5 -h 0')
    # Esegui la cross-validation
    accuracy = svm_train(problem, param)
    print("Cross-validation accuracy:", accuracy)

    # --- SPLIT TRAIN AND TEST DATASET ---

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # --- MODEL ---

    # --- EVALUATION ---

    # Definire il modello SVM
    svm_model = SVC(kernel='rbf', random_state=42)

    # Eseguire la cross-validation
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)

    # Mostra i risultati
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # Create a svm Classifier
    clf = svm.SVC(kernel='rbf')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))
