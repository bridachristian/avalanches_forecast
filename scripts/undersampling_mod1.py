# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:31:23 2024

@author: Christian
"""
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    # mod1 = mod1.drop(columns=['Stagione'])
    mod1['DataRilievo'] = pd.to_datetime(
        mod1['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index
    mod1.set_index('DataRilievo', inplace=True)

    return mod1


def undersampling_random(mod1):
    # Rimuovi le righe dove 'AvalDay' è NaN
    mod1 = mod1.dropna(subset=['AvalDay'])

    # Lista per tenere traccia degli indici da mantenere
    indices_to_keep = []

    # Trova gli indici dove AvalDay == 1
    aval_days_indices = mod1.index[mod1['AvalDay'] == 1].tolist()

    # Itera su ogni indice dove AvalDay è 1
    for idx in aval_days_indices:
        # Aggiungi gli indici dei 10 giorni precedenti e 5 giorni successivi
        start_idx = idx - pd.DateOffset(days=10)  # Sottrai 10 giorni
        end_idx = idx + pd.DateOffset(days=5)  # Aggiungi 5 giorni

        # Aggiungi gli intervalli di date alla lista
        indices_to_keep.extend(mod1.loc[start_idx:end_idx].index)

    # Rimuovi duplicati
    indices_to_keep = list(set(indices_to_keep))

    # Filtra il dataframe usando gli indici selezionati (dove AvalDay == 1 o vicini)
    mod1_undersampled = mod1.loc[indices_to_keep].sort_index()

    one_class_samples = mod1[mod1['AvalDay'] == 1].dropna()  # NaN excluded

    # Ora aggiungiamo campionamento casuale di 440 righe con AvalDay == 0
    # escludiamo quelle che contengono colonne NaN
    zero_class_samples = mod1[mod1['AvalDay'] == 0].dropna().sample(
        n=len(one_class_samples), random_state=42)

    # Uniamo i dati campionati con AvalDay == 0 ai dati con AvalDay == 1
    final_dataset = pd.concat(
        [one_class_samples, zero_class_samples]).sort_index()

    return final_dataset


def undersampling_nearmiss(mod1):

    # Drop NaNs and prepare X and y with index preserved
    mod1_clean = mod1.dropna()  # Keep the clean DataFrame with the datetime index
    X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    y = mod1_clean['AvalDay']

    # Save the datetime index
    original_index = X.index

    # Perform NearMiss resampling
    nm = NearMiss(version=3)
    X_res, y_res = nm.fit_resample(X, y)

    # Convert X_res and y_res back to DataFrames and Series with the original datetime index
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name='AvalDay')

    # Apply the original datetime index to the resampled data
    X_res.index = original_index[nm.sample_indices_]
    y_res.index = original_index[nm.sample_indices_]

    # Combine X_res and y_res into a single DataFrame
    final_dataset = pd.concat([X_res, y_res], axis=1)

    # final_dataset = pd.concat([X_res, y_res]).sort_index()

    # X = X[['SWEnew', 'TmaxG']]
    # counter = Counter(y)
    # for label, _ in counter.items():
    #     row_ix = np.where(y == label)[0]
    #     # Use .iloc for DataFrame indexing
    #     plt.scatter(X.iloc[row_ix, 1], X.iloc[row_ix, 0], label=str(label))

    # plt.legend()
    # plt.show()

    # counter = Counter(y_res)
    # for label, _ in counter.items():
    #     row_ix = np.where(y_res == label)[0]
    #     # Use .iloc for DataFrame indexing
    #     plt.scatter(X_res.iloc[row_ix, 1],
    #                 X_res.iloc[row_ix, 0], label=str(label))

    # plt.legend()
    # plt.show()

    return final_dataset


def oversampling_SMOTE(mod1):

    # Drop NaNs and prepare X and y with index preserved
    mod1_clean = mod1.dropna()
    X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
    y = mod1_clean['AvalDay']

    # print("Original class distribution:", Counter(y))

    # Apply SMOTE to the dataset
    # smote = SMOTE(sampling_strategy='minority', random_state=42)
    smote_tomek = SMOTETomek(random_state=42)

    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    # print("Resampled class distribution:", Counter(y_resampled))

    # Convert X_res and y_res back to DataFrames and Series with a reset index
    X_res = pd.DataFrame(X_resampled, columns=X.columns)
    y_res = pd.Series(y_resampled, name='AvalDay')

    # Option 1: Keep original indices for non-synthetic data and assign new ones to synthetic samples
    original_index = mod1_clean.index
    # Assign the original index to non-synthetic data rows, and new indices to synthetic rows
    new_index = pd.Index(range(len(X_res)))  # new index as a placeholder
    X_res.index = new_index
    y_res.index = new_index

    # Option 2: Combine `X_res` and `y_res` into a single DataFrame with the new index
    final_dataset = pd.concat([X_res, y_res], axis=1)

    return final_dataset


def save_outputfile(df, output_filepath):
    """Save the mod1_features dataframe to a CSV file."""
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures.csv'

    output_filepath = common_path / 'mod1_undersampling.csv'
    output_filepath2 = common_path / 'mod1_oversampling.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- UNDERSAMPLING FEATURES ---

    # mod1_undersampled = undersampling_random(mod1)
    mod1_undersampled = undersampling_nearmiss(mod1)
    mod1_oversampled = oversampling_SMOTE(mod1)

    # --- SAVE OUTPUT FILE ---

    save_outputfile(mod1_oversampled, output_filepath2)


if __name__ == '__main__':
    main()
