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


def calculate_day_of_season(df, season_start_date='12-01'):
    """
    Calculate the day of the season (starting from a given season start date, e.g., 1st December).

    Parameters:
    - df: DataFrame containing a column with date information (e.g., 'Date' in format YYYY-MM-DD).
    - season_start_date: The start date of the season in 'MM-DD' format (default is '12-01').

    Returns:
    - DataFrame with an additional 'DayOfSeason' column.
    """

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Calculate season start for each year
    season_start_year = df.index.to_series().apply(
        lambda x: x.year if x.month >= 12 else x.year - 1
    )
    season_start = pd.to_datetime(
        season_start_year.astype(str) + '-' + season_start_date
    )

    # Calculate the day of the season
    df['DayOfSeason'] = (df.index - season_start).dt.days + 1

    return df


def calculate_snow_height_differences(df):
    """Calculate snow height differences over different periods."""
    df['HS_delta_1d'] = df['HSnum'].diff(periods=1)
    df['HS_delta_2d'] = df['HSnum'].diff(periods=2)
    df['HS_delta_3d'] = df['HSnum'].diff(periods=3)
    df['HS_delta_5d'] = df['HSnum'].diff(periods=5)
    return df


def calculate_new_snow(df):
    """Calculate cumulative new snow metrics over different periods."""
    df['HN_2d'] = df['HNnum'].rolling(
        window=2).sum()  # Cumulative snowfall over 2 days
    # Cumulative snowfall over 3 days
    df['HN_3d'] = df['HNnum'].rolling(window=3).sum()
    # Cumulative snowfall over 5 days
    df['HN_5d'] = df['HNnum'].rolling(window=5).sum()

    # Calculate days since last snowfall (where HNnum > 1)
    df['DaysSinceLastSnow'] = (df['HNnum'] <= 1).astype(
        int).groupby(df['HNnum'].gt(1).cumsum()).cumsum()

    return df


def calculate_temperature(df):
    """Calculate minimum, maximum temperatures and their differences over different periods."""
    # Minimum/maximum temperatures in the last 2, 3, 5 days
    df['Tmin_2d'] = df['TminG'].rolling(
        window=2).min()  # Min temperature over 2 days
    df['Tmax_2d'] = df['TmaxG'].rolling(
        window=2).max()  # Max temperature over 2 days
    df['Tmin_3d'] = df['TminG'].rolling(
        window=3).min()  # Min temperature over 3 days
    df['Tmax_3d'] = df['TmaxG'].rolling(
        window=3).max()  # Max temperature over 3 days
    df['Tmin_5d'] = df['TminG'].rolling(
        window=5).min()  # Min temperature over 5 days
    df['Tmax_5d'] = df['TmaxG'].rolling(
        window=5).max()  # Max temperature over 5 days

    # Temperature amplitude (difference between max and min temperatures)
    df['TempAmplitude_1d'] = df['TmaxG'] - \
        df['TminG']               # Amplitude for today
    df['TempAmplitude_2d'] = df['Tmax_2d'] - \
        df['Tmin_2d']   # Amplitude over 2 days
    df['TempAmplitude_3d'] = df['Tmax_3d'] - \
        df['Tmin_3d']   # Amplitude over 3 days
    df['TempAmplitude_5d'] = df['Tmax_5d'] - \
        df['Tmin_5d']   # Amplitude over 5 days

    # Difference of TaG between today and previous days
    df['Ta_delta_1d'] = df['TaG'].diff(periods=1)
    df['Ta_delta_2d'] = df['TaG'].diff(periods=2)
    df['Ta_delta_3d'] = df['TaG'].diff(periods=3)
    df['Ta_delta_5d'] = df['TaG'].diff(periods=5)

    # Difference of TminG between today and previous days
    df['Tmin_delta_1d'] = df['TminG'].diff(periods=1)
    df['Tmin_delta_2d'] = df['TminG'].diff(periods=2)
    df['Tmin_delta_3d'] = df['TminG'].diff(periods=3)
    df['Tmin_delta_5d'] = df['TminG'].diff(periods=5)

    # Difference of TmaxG between today and previous days
    df['Tmax_delta_1d'] = df['TmaxG'].diff(periods=1)
    df['Tmax_delta_2d'] = df['TmaxG'].diff(periods=2)
    df['Tmax_delta_3d'] = df['TmaxG'].diff(periods=3)
    df['Tmax_delta_5d'] = df['TmaxG'].diff(periods=5)

    return df


def calculate_degreedays(df):
    # Calculate the daily mean temperature
    df['T_mean'] = (df['TmaxG'] + df['TminG']) / 2

    # Calculate positive degree days (only when T_mean > 0)
    df['DegreeDays_Pos'] = np.where(df['T_mean'] > 0, df['T_mean'], 0)

    # Calculate cumulative degree days over different periods
    df['DegreeDays_cumsum_2d'] = df['DegreeDays_Pos'].rolling(window=2).sum()
    df['DegreeDays_cumsum_3d'] = df['DegreeDays_Pos'].rolling(window=3).sum()
    df['DegreeDays_cumsum_5d'] = df['DegreeDays_Pos'].rolling(window=5).sum()

    return df


def calculate_snow_temperature(df):
    df['TH10_tanh'] = 20*np.tanh(0.2*df['TH01G'])  # new, hyperbolic transform.
    df['TH30_tanh'] = 20*np.tanh(0.2*df['TH03G'])  # new, hyperbolic transform.

    df['Tsnow_delta_1d'] = df['TH01G'].diff(periods=1)
    df['Tsnow_delta_2d'] = df['TH01G'].diff(periods=2)
    df['Tsnow_delta_3d'] = df['TH01G'].diff(periods=3)
    df['Tsnow_delta_5d'] = df['TH01G'].diff(periods=5)

    # Categorize snow types based on temperature
    df['SnowType_Cold'] = np.where(df['TH01G'] < -10, 1, 0)
    df['SnowType_Warm'] = np.where(
        (df['TH01G'] >= -10) & (df['TH01G'] < -2), 1, 0)
    df['SnowType_Wet'] = np.where(
        (df['TH01G'] >= -2) & (df['TH01G'] <= 0), 1, 0)

    # Condensed snow condition index
    df['SnowConditionIndex'] = np.where(
        df['TH01G'] < -10, 0,  # Cold Snow
        np.where((df['TH01G'] >= -10) & (df['TH01G'] < -2), 1,  # Warm Snow
                 np.where((df['TH01G'] >= -2) & (df['TH01G'] <= 0), 2,  # Wet Snow
                          -1))  # Invalid condition (optional)
    )

    # Count consecutive days of wet snow
    df['ConsecWetSnowDays'] = (
        df['SnowType_Wet'].groupby(
            (df['SnowType_Wet'] != df['SnowType_Wet'].shift()).cumsum()).cumsum()
    )

    # Zero out non-wet days in the consecutive count column for clarity
    df['ConsecWetSnowDays'] = np.where(
        df['SnowType_Wet'] == 1, df['ConsecWetSnowDays'], 0)

    # Drop intermediate snow type columns
    df = df.drop(columns=['SnowType_Cold', 'SnowType_Warm', 'SnowType_Wet'])

    return df


def calculate_wind_snow_drift(df):
    """Calculate snow drift based on wind strength (VQ1)."""
    df['SnowDrift_1d'] = df['VQ1'].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 0})
    df['SnowDrift_2d'] = df['SnowDrift_1d'].rolling(window=2).sum()
    df['SnowDrift_3d'] = df['SnowDrift_1d'].rolling(window=3).sum()
    df['SnowDrift_5d'] = df['SnowDrift_1d'].rolling(window=5).sum()

    return df


def calculate_swe(df):
    # Adjusted snow density based on conditions
    df['rho_adj'] = np.where(df['HNnum'] < 6, 100, df['rho'])
    df['rho_adj'] = np.where(df['HNnum'] == 0, 0, df['rho_adj'])

    # Calculate fresh snow water equivalent (FreshSWE)
    df['FreshSWE'] = df['HNnum'] * df['rho_adj'] / 100
    df['SeasonalSWE_cum'] = df.groupby('Stagione')['FreshSWE'].cumsum()

    # Precipitation sums over different periods
    df['Precip_1d'] = df['FreshSWE']  # Instantaneous precipitation (FreshSWE)
    df['Precip_2d'] = df['FreshSWE'].rolling(
        window=2).sum()  # Cumulative for 48h
    df['Precip_3d'] = df['FreshSWE'].rolling(
        window=3).sum()  # Cumulative for 72h
    df['Precip_5d'] = df['FreshSWE'].rolling(
        window=5).sum()  # Cumulative for 120h

    df = df.drop(columns=['rho_adj'])

    return df


def calculate_wet_snow(df):
    # Wet snow presence based on CS (Critical Snow Surface)
    df['WetSnow_CS'] = np.where(df['CS'] >= 20, 1, 0)
    df['WetSnow_CS'] = np.where(df['CS'].isna(), np.nan, df['WetSnow_CS'])

    # Wet snow presence based on temperature (TH01G)
    df['WetSnow_Temperature'] = np.where(df['TH01G'] >= -2, 1, 0)
    df['WetSnow_Temperature'] = np.where(
        df['TH01G'].isna(), np.nan, df['WetSnow_Temperature'])

    return df


def calculate_temperature_gradient(df):
    # Calculate the temperature gradient based on snow height
    df['TempGrad_HS'] = abs(df['TH01G']) / (df['HSnum'] - 10)
    df['TempGrad_HS'] = np.where(
        df['TempGrad_HS'] == np.inf, np.nan, df['TempGrad_HS'])
    return df


def calculate_LooseSnow_avalanches(df):
    df['LooseSnowAval_Type'] = np.where(
        (df['L1'] >= 1) & (df['L2'].isin([3, 4])), 1, 0)
    df['LooseSnowAval_Type'] = np.where(
        df['L1'].isna(), np.nan, df['LooseSnowAval_Type'])
    return df


def calculate_MFcrust(df):
   # Identify MF crust presence
    df['MF_Crust_Present'] = np.where(df['CS'].isin([12, 13, 22, 23]), 1, 0)
    df['MF_Crust_Present'] = np.where(
        df['CS'].isna(), np.nan, df['MF_Crust_Present'])

    # Forward fill to ensure continuity
    df['MF_Crust_Present'] = df['MF_Crust_Present'].ffill()

    # Check for new MF crust (transition from 0 to 1)
    df['New_MF_Crust'] = df['MF_Crust_Present'].diff().fillna(
        0).where(df['MF_Crust_Present'] == 1, 0)

    # Count consecutive days with crust
    df['ConsecCrustDays'] = (df['MF_Crust_Present'].cumsum() -
                             df['MF_Crust_Present'].cumsum().where(df['MF_Crust_Present'] == 0).ffill().fillna(0))

    # Drop the 'CS' column as it's no longer needed
    df = df.drop(columns=['CS'])

    return df


def calculate_penetration(df):
    df['Penetration_ratio'] = df['PR'] / df['HSnum']
    df['Penetration_ratio'] = np.where(
        df['Penetration_ratio'] >= 1, 1, df['Penetration_ratio'])
    return df


def calculate_avalanche_days(df):
    """Calculate avalanche occurrence and moving averages."""
    df['AvalDay'] = np.where(df['L1'] >= 1, 1, df['L1'])
    df['AvalDay'] = np.where((df['AvalDay'] == 1) & (
        df['L2'].isin([1, 2, 5, 6])) & (df['VQ1'] >= 1), 0, df['AvalDay'])

    df['AvalDay_2d'] = df['AvalDay'].shift(1).rolling(window=1).mean()
    df['AvalDay_3d'] = df['AvalDay'].shift(1).rolling(window=2).mean()
    df['AvalDay_5d'] = df['AvalDay'].shift(1).rolling(window=4).mean()
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

    # --- NEW FEATURES CREATION ---

    # Add new variables to the dataset
    mod1 = calculate_day_of_season(mod1, season_start_date='12-01')
    mod1_features = mod1[['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG',
                          'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1', 'L2']]
    mod1_features = calculate_day_of_season(
        mod1_features, season_start_date='12-01')
    mod1_features = calculate_snow_height_differences(mod1_features)
    mod1_features = calculate_new_snow(mod1_features)
    mod1_features = calculate_temperature(mod1_features)
    mod1_features = calculate_degreedays(mod1_features)
    mod1_features = calculate_wind_snow_drift(mod1_features)
    mod1_features = calculate_swe(mod1_features)
    mod1_features = calculate_penetration(mod1_features)
    mod1_features = calculate_wet_snow(mod1_features)
    mod1_features = calculate_temperature_gradient(mod1_features)
    mod1_features = calculate_snow_temperature(mod1_features)
    mod1_features = calculate_MFcrust(mod1_features)
    mod1_features = calculate_avalanche_days(mod1_features)

    mod1_features = mod1_features.drop(columns=['rho', 'B', 'L1', 'L2'])

    # --- DROP NON-SENSE FEATURES ---

    # mod1_features = mod1_features.drop(
    #     columns=['rho', 'B', 'L1', 'L2', 'rho_adjusted'])

    # --- DATA SAVING ---

    save_mod1_features(mod1_features, output_filepath)


if __name__ == '__main__':
    main()
