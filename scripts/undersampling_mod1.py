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


def load_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    # mod1 = mod1.drop(columns=['Stagione'])
    mod1['DataRilievo'] = pd.to_datetime(
        mod1['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index
    mod1.set_index('DataRilievo', inplace=True)

    return mod1


def undersampling(mod1):
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


def save_outputfile(df, output_filepath):
    """Save the mod1_features dataframe to a CSV file."""
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    filepath = common_path / 'mod1_newfeatures_selected.csv'

    output_filepath = common_path / 'mod1_undersampling.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- UNDERSAMPLING FEATURES ---

    mod1_undersampled = undersampling(mod1)

    # --- SAVE OUTPUT FILE ---

    save_outputfile(mod1_undersampled, output_filepath)


if __name__ == '__main__':
    main()
