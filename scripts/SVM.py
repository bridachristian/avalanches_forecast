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
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati')

    filepath = common_path / 'mod1_undersampling.csv'

    # output_filepath = Path(
    #     'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_undersampling.csv')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1_undersampled = load_data(filepath)
    print(mod1_undersampled.dtypes)  # For initial data type inspection

    # --- SPLIT FEATURES AND TARGET---

    X = mod1_undersampled[['HNnum', 'HN72h']]
    y = mod1_undersampled['AvalDay']

    # Normalizzare i dati
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- SPLIT TRAIN AND TEST DATASET---

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42)

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
