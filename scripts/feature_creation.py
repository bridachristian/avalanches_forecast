# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:14:07 2024

@author: Christian
"""

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


def calculate_snow_height_differences(df):
    """Calculate snow height differences over different periods."""
    df['HSdiff24h'] = df['HSnum'].diff(periods=1)
    df['HSdiff48h'] = df['HSnum'].diff(periods=2)
    df['HSdiff72h'] = df['HSnum'].diff(periods=3)
    df['HSdiff120h'] = df['HSnum'].diff(periods=5)
    return df


def calculate_new_snow(df):
    """Calculate cumulative new snow metrics over different periods."""
    df['HN48h'] = df['HNnum'].rolling(window=2).sum()
    df['HN72h'] = df['HNnum'].rolling(window=3).sum()
    df['HN120h'] = df['HNnum'].rolling(window=5).sum()
    return df


def calculate_temperature(df):
    """Calculate minimum, maximum temperatures and their differences over different periods."""
    df['Tmin48h'] = df['TminG'].rolling(window=2).min()
    df['Tmax48h'] = df['TmaxG'].rolling(window=2).max()
    df['Tmin72h'] = df['TminG'].rolling(window=3).min()
    df['Tmax72h'] = df['TmaxG'].rolling(window=3).max()
    df['Tmin120h'] = df['TminG'].rolling(window=5).min()
    df['Tmax120h'] = df['TmaxG'].rolling(window=5).max()

    # Temperature amplitude
    df['Tdelta24h'] = df['TmaxG'] - df['TminG']
    df['Tdelta48h'] = df['Tmax48h'] - df['Tmin48h']
    df['Tdelta72h'] = df['Tmax72h'] - df['Tmin72h']
    df['Tdelta120h'] = df['Tmax120h'] - df['Tmin120h']
    return df


def calculate_wind_snow_drift(df):
    """Calculate snow drift based on wind strength (VQ1)."""
    df['SnowDrift'] = df['VQ1'].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 0})
    df['SnowDrift48h'] = df['SnowDrift'].rolling(window=2).sum()
    df['SnowDrift72h'] = df['SnowDrift'].rolling(window=3).sum()
    df['SnowDrift120h'] = df['SnowDrift'].rolling(window=5).sum()
    return df


def calculate_swe(df):
    """Calculate snow water equivalent (SWE) and cumulative precipitation sums."""
    df['rho_adjusted'] = np.where(df['HNnum'] < 6, 100, df['rho'])
    df['rho_adjusted'] = np.where(df['HNnum'] == 0, 0, df['rho_adjusted'])

    df['SWEnew'] = df['HNnum'] * df['rho_adjusted'] / 100
    df['SWE_cumulative'] = df.groupby('Stagione')['SWEnew'].cumsum()

    # Precipitation sums over different periods
    df['PSUM24h'] = df['SWEnew']
    df['PSUM48h'] = df['SWEnew'].rolling(window=2).sum()
    df['PSUM72h'] = df['SWEnew'].rolling(window=3).sum()
    df['PSUM120h'] = df['SWEnew'].rolling(window=5).sum()
    return df


def calculate_wet_snow(df):
    """Calculate wet snow presence based on CS (critical snow surface)."""
    df['WetSnow1'] = np.where(df['CS'] >= 20, 1, 0)
    df['WetSnow1'] = np.where(df['CS'].isna(), np.nan, df['WetSnow1'])

    df['WetSnow2'] = np.where(df['TH01G'] >= -1, 1, 0)
    df['WetSnow2'] = np.where(df['TH01G'].isna(), np.nan, df['WetSnow2'])

    return df


def calculate_temperature_gradient(df):
    """Calculate the temperature gradient based on the snow height."""
    df['T_gradient'] = abs(df['TH01G']) / (df['HSnum'] - 10)
    df['T_gradient'] = np.where(
        df['T_gradient'] == np.inf, np.nan, df['T_gradient'])
    return df


def calculate_avalanche_days(df):
    """Calculate avalanche occurrence and moving averages."""
    df['AvalDay'] = np.where(df['L1'] >= 1, 1, df['L1'])
    df['AvalDay48h'] = df['AvalDay'].rolling(window=2).mean()
    df['AvalDay72h'] = df['AvalDay'].rolling(window=3).mean()
    df['AvalDay120h'] = df['AvalDay'].rolling(window=5).mean()
    return df


def calculate_LooseSnow_avalanches(df):
    df['LooseSnowAval'] = np.where(
        (df['L1'] >= 1) & (df['L2'].isin([1, 2])), 1, 0)
    df['LooseSnowAval'] = np.where(
        df['L1'].isna(), np.nan, df['LooseSnowAval'])
    return df


def calculate_MFcrust(df):
    df['MFcrust'] = np.where(df['CS'].isin([12, 13, 22, 23]), 1, 0)
    df['MFcrust'] = np.where(df['CS'].isna(), np.nan, df['MFcrust'])
    return df


def save_mod1_features(df, output_filepath):
    """Save the mod1_features dataframe to a CSV file."""
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation')

    filepath = common_path / 'mod1_gapfillled.csv'

    output_filepath = common_path / 'mod1_newfeatures.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- DATA MANIPULATION ---

    # Add new variables to the dataset
    mod1_features = mod1[['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG',
                          'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1', 'L2']]
    mod1_features = calculate_snow_height_differences(mod1_features)
    mod1_features = calculate_new_snow(mod1_features)
    mod1_features = calculate_temperature(mod1_features)
    mod1_features = calculate_wind_snow_drift(mod1_features)
    mod1_features = calculate_swe(mod1_features)
    mod1_features = calculate_wet_snow(mod1_features)
    mod1_features = calculate_temperature_gradient(mod1_features)
    mod1_features = calculate_avalanche_days(mod1_features)
    mod1_features = calculate_LooseSnow_avalanches(mod1_features)
    mod1_features = calculate_MFcrust(mod1_features)

    # --- DATA SAVING ---

    save_mod1_features(mod1_features, output_filepath)


if __name__ == '__main__':
    main()
