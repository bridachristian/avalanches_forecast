# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:14:07 2024

@author: Christian
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns


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
    # Initialize the difference columns
    for period in [1, 2, 3, 5]:
        col_name = f'HS_delta_{period}d'
        df[col_name] = df['HSnum'].diff(periods=period)

    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Identify where Stagione changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Reset differences to NaN for each new season
    for col in [f'HS_delta_{period}d' for period in [1, 2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Set HS_delta_2d, HS_delta_3d, and HS_delta_5d to NaN for 2, 3, 5 days following a Stagione change
    for period in [2, 3, 5]:
        for idx in df.index[stagione_changes]:
            # Add the specified period to the current timestamp using pd.Timedelta
            end_idx = idx + pd.Timedelta(days=period)

            # Set the snow height differences to NaN for the period following the change
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   f'HS_delta_{period}d'] = np.nan

    # Filter by z-score for each HS_delta_Xd column
    for col in [f'HS_delta_{period}d' for period in [1, 2, 3, 5]]:
        # Calculate z-scores, ignoring NaN values
        col_zscore = zscore(df[col], nan_policy='omit')
        # Set values with z-scores exceeding ±3 to NaN
        df[col] = np.where(np.abs(col_zscore) > 3, np.nan, df[col])

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
    """
    Calculate minimum, maximum temperatures, their differences, and temperature amplitudes over different periods.
    Resets calculations at the start of a new season ('Stagione').
    """
    # Define periods for rolling calculations and differences
    periods = [2, 3, 5]

    # Rolling min, max, and reset for each new season
    for period in periods:
        df[f'Tmin_{period}d'] = df['TminG'].rolling(window=period).min()
        df[f'Tmax_{period}d'] = df['TmaxG'].rolling(window=period).max()

    # Detect season changes and reset rolling calculations
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)
    for col in [f'Tmin_{period}d' for period in periods] + [f'Tmax_{period}d' for period in periods]:
        df.loc[stagione_changes, col] = np.nan

    # Calculate temperature amplitude
    df['TempAmplitude_1d'] = df['TmaxG'] - df['TminG']
    for period in periods:
        df[f'TempAmplitude_{period}d'] = df[f'Tmax_{period}d'] - \
            df[f'Tmin_{period}d']

    # Calculate differences for TaG, TminG, TmaxG
    for temp_col in ['TaG', 'TminG', 'TmaxG']:
        for period in [1] + periods:
            col_name = f'{temp_col}_delta_{period}d'
            df[col_name] = df[temp_col].diff(periods=period)
            df.loc[stagione_changes, col_name] = np.nan

    # Set NaN for 2, 3, 5 days after a season change for each relevant column
    for period in [2, 3, 5]:
        # Set NaN for the next 'period' days after the season change
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            # Ensure the range includes the next 'period' days
            df.loc[idx + pd.Timedelta(days=1): end_idx, [
                f'Tmin_{period}d', f'Tmax_{period}d', f'TempAmplitude_{period}d']] = np.nan
            for temp_col in ['TaG', 'TminG', 'TmaxG']:
                df.loc[idx + pd.Timedelta(days=1): end_idx,
                       f'{temp_col}_delta_{period}d'] = np.nan

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

    # Detect season changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Reset cumulative degree days to NaN for each new season
    for col in [f'DegreeDays_cumsum_{period}d' for period in [2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Set NaN for 1, 2, 3, 5 days after a season change for cumulative degree days
    for period in [2, 3, 5]:
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            # Set NaN for the following 'period' days for cumulative degree days
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'DegreeDays_cumsum_{p}d' for p in [2, 3, 5]]] = np.nan

    return df


def calculate_snow_temperature(df):
    """
    Calculate snow-related temperature features and categorize snow types based on temperature.
    Resets calculations when the 'Stagione' column changes.
    """
    # Hyperbolic transformations
    df['TH10_tanh'] = 20 * np.tanh(0.2 * df['TH01G'])
    df['TH30_tanh'] = 20 * np.tanh(0.2 * df['TH03G'])

    # Snow temperature differences
    for period in [1, 2, 3, 5]:
        df[f'Tsnow_delta_{period}d'] = df['TH01G'].diff(periods=period)

    # Categorize snow types based on temperature
    df['SnowConditionIndex'] = np.select(
        [df['TH01G'] < -10, (df['TH01G'] >= -10) & (df['TH01G'] < -2),
         (df['TH01G'] >= -2) & (df['TH01G'] <= 0)],
        [0, 1, 2],  # 0: Cold Snow, 1: Warm Snow, 2: Wet Snow
        default=np.nan  # Invalid condition (optional)
    )

    # Count consecutive days of wet snow
    df['ConsecWetSnowDays'] = (
        df['SnowConditionIndex'].eq(2).groupby(
            (df['SnowConditionIndex'].ne(2)).cumsum()
        ).cumsum()
    )

    # Zero out non-wet days in the consecutive count column
    df['ConsecWetSnowDays'] = np.where(
        df['SnowConditionIndex'] == 2, df['ConsecWetSnowDays'], 0)

    # Handle resets when 'Stagione' changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Set NaN for snow-related temperature features for 1, 2, 3, and 5 days after a season change
    for period in [1, 2, 3, 5]:
        # Set NaN for the following 'period' days for snow temperature differences
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'Tsnow_delta_{p}d' for p in [1, 2, 3, 5]]] = np.nan
            df.loc[idx + pd.Timedelta(days=1): end_idx, ['SnowConditionIndex',
                                                         'ConsecWetSnowDays', 'TH10_tanh', 'TH30_tanh']] = np.nan

    # Set NaN for snow-related temperature features at the start of each new season
    for col in [f'Tsnow_delta_{period}d' for period in [1, 2, 3, 5]] + [
            'SnowConditionIndex', 'ConsecWetSnowDays', 'TH10_tanh', 'TH30_tanh']:
        df.loc[stagione_changes, col] = np.nan

    return df


def calculate_wind_snow_drift(df):
    """
    Calculate snow drift indices based on wind strength (VQ1).
    Resets calculations when the 'Stagione' column changes.

    Parameters:
    - df: DataFrame containing wind strength column 'VQ1' and season column 'Stagione'.

    Returns:
    - DataFrame with added snow drift columns.
    """
    # Map wind strength to drift index
    df['SnowDrift_1d'] = df['VQ1'].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 0})

    # Calculate rolling sums for 2, 3, and 5 days
    for period in [2, 3, 5]:
        df[f'SnowDrift_{period}d'] = df['SnowDrift_1d'].rolling(
            window=period).sum()

    # Identify season changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Reset rolling sums to NaN when the season changes
    for col in [f'SnowDrift_{period}d' for period in [2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Set NaN for 1, 2, 3, and 5 days after a season change for snow drift indices
    for period in [1, 2, 3, 5]:
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'SnowDrift_{p}d' for p in [2, 3, 5]]] = np.nan

    return df


def calculate_swe(df):
    """
    Calculate Snow Water Equivalent (SWE) and related precipitation metrics.

    Parameters:
    - df: DataFrame containing columns 'HNnum', 'rho', and 'Stagione'.

    Returns:
    - DataFrame with additional SWE and precipitation metrics.
    """
    # Adjust snow density based on snowfall conditions
    df['rho_adj'] = np.where(df['HNnum'] < 6, 100, df['rho'])
    df['rho_adj'] = np.where(df['HNnum'] == 0, 0, df['rho_adj'])

    # Calculate fresh snow water equivalent (FreshSWE)
    df['FreshSWE'] = df['HNnum'] * df['rho_adj'] / 100

    # Cumulative Seasonal SWE
    df['SeasonalSWE_cum'] = df.groupby('Stagione')['FreshSWE'].cumsum()

    # Rolling precipitation sums for different periods
    for period in [1, 2, 3, 5]:
        col_name = f'Precip_{period}d'
        df[col_name] = df['FreshSWE'].rolling(window=period).sum()

    # Identify season changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Set NaN for 1, 2, 3, and 5 days after a season change for precipitation sums
    for period in [1, 2, 3, 5]:
        for idx in df.index[stagione_changes]:
            end_idx = idx + pd.Timedelta(days=period)
            df.loc[idx + pd.Timedelta(days=1): end_idx,
                   [f'Precip_{p}d' for p in [1, 2, 3, 5]]] = np.nan

    # Reset rolling precipitation sums to NaN when the season changes
    for col in [f'Precip_{period}d' for period in [1, 2, 3, 5]]:
        df.loc[stagione_changes, col] = np.nan

    # Drop intermediate adjustment column
    df.drop(columns=['rho_adj'], inplace=True)

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
    """
    Calculate Melt-Freeze (MF) crust presence, new crust events, 
    and consecutive days with crust.

    Parameters:
    - df: DataFrame with columns 'CS' and 'Stagione'.

    Returns:
    - DataFrame with additional columns related to MF crust analysis.
    """
    # Identify MF crust presence
    df['MF_Crust_Present'] = np.where(df['CS'].isin([12, 13, 22, 23]), 1, 0)
    df['MF_Crust_Present'] = np.where(
        df['CS'].isna(), np.nan, df['MF_Crust_Present'])

    # Forward fill to ensure continuity
    df['MF_Crust_Present'] = df['MF_Crust_Present'].ffill()

    # Identify new MF crust formations (transition from 0 to 1)
    df['New_MF_Crust'] = (
        (df['MF_Crust_Present'] == 1) &
        (df['MF_Crust_Present'].shift(1) == 0)
    ).astype(int)

    # Reset counts and handle season changes
    stagione_changes = df['Stagione'] != df['Stagione'].shift(1)

    # Count consecutive days with crust
    df['ConsecCrustDays'] = df.groupby(
        (df['MF_Crust_Present'] != df['MF_Crust_Present'].shift()).cumsum()
    ).cumcount() + 1

    # Zero out consecutive crust days where crust is not present
    df['ConsecCrustDays'] = np.where(
        df['MF_Crust_Present'] == 1, df['ConsecCrustDays'], 0)

    # Reset consecutive count when seasons change
    df.loc[stagione_changes, 'ConsecCrustDays'] = np.nan

    # Drop the 'CS' column as it's no longer needed
    df.drop(columns=['CS'], inplace=True)

    return df


def calculate_penetration(df):
    df['Penetration_ratio'] = df['PR'] / df['HSnum']
    df['Penetration_ratio'] = np.where(
        df['Penetration_ratio'] >= 1, 1, df['Penetration_ratio'])
    return df


def calculate_avalanche_days(df):
    """
    Calculate avalanche occurrence and moving averages over specified periods.

    Parameters:
    - df: DataFrame with columns 'L1', 'L2', and 'VQ1'.

    Returns:
    - DataFrame with added avalanche-related columns.
    """
    # Define conditions for avalanche occurrence
    conditions = [
        df['L1'].isin([2, 3, 4]),  # L1 is 2, 3, or 4
        df['L1'] == 0,             # L1 is 0
        df['L1'] == 1              # L1 is 1
    ]
    # Corresponding values for 'AvalDay'
    values = [1, 0, np.nan]

    # Calculate initial avalanche occurrence
    df['AvalDay'] = np.select(conditions, values, default=np.nan)

    # Additional condition to reset 'AvalDay' based on 'L2' and 'VQ1'
    df['AvalDay'] = np.where(
        (df['AvalDay'] == 1) & (df['L2'].isin([1, 2, 5, 6])) & (df['VQ1'] >= 1),
        0,
        df['AvalDay']
    )

    # Calculate moving averages of avalanche occurrence
    df['AvalDay_2d'] = df['AvalDay'].shift(1).rolling(window=2).mean()
    df['AvalDay_3d'] = df['AvalDay'].shift(1).rolling(window=3).mean()
    df['AvalDay_5d'] = df['AvalDay'].shift(1).rolling(window=5).mean()

    # Reset moving averages when seasons change
    if 'Stagione' in df.columns:
        stagione_changes = df['Stagione'] != df['Stagione'].shift(1)
        for col in ['AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d']:
            df.loc[stagione_changes, col] = np.nan

        # Set NaN for 1, 2, 3, and 5 days after a season change for avalanche-related columns
        for period in [1, 2, 3, 5]:
            for idx in df.index[stagione_changes]:
                end_idx = idx + pd.Timedelta(days=period)
                df.loc[idx + pd.Timedelta(days=1): end_idx,
                       ['AvalDay', 'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d']] = np.nan

    return df


def save_mod1_features(df, output_filepath):
    """Save the mod1_features dataframe to a CSV file."""
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


def plot_boxplot(df, title, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.violinplot(data=df)
    plt.title("Boxplot of DataFrame Features", fontsize=14)
    plt.ylabel("Values", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()


def plot_new_columns(df_before, df_after, title, figsize=(10, 6)):
    """Plot boxplots of only the new columns added to the DataFrame."""
    new_columns = list(set(df_after.columns) - set(df_before.columns))
    if not new_columns:
        print(f"No new columns added for {title}.")
        return

    plt.figure(figsize=figsize)
    sns.violinplot(data=df_after[new_columns])
    plt.title(title, fontsize=14)
    plt.ylabel("Values", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation')

    filepath = common_path / 'mod1_gapfillled.csv'

    output_filepath = common_path / 'mod1_newfeatures_NEW.csv'
    summary_filepath = common_path / 'mod1_newfeatures_NEW_summary.csv'

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # --- NEW FEATURES CREATION ---

    # Initial measured values
    mod1_features = mod1[['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG',
                          'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1', 'L2']]

    plot_boxplot(mod1_features, title="Boxplot of Measured Values")

    # Step-by-step feature calculations with appropriate titles
    df_before = mod1_features.copy()
    mod1_features = calculate_day_of_season(
        mod1_features, season_start_date='12-01')
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Day of Season Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_snow_height_differences(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Snow Height Differences")

    df_before = mod1_features.copy()
    mod1_features = calculate_new_snow(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of New Snow Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_temperature(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Temperature Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_degreedays(mod1_features)
    plot_new_columns(df_before, mod1_features, title="Boxplot of Degree Days")

    df_before = mod1_features.copy()
    mod1_features = calculate_wind_snow_drift(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Wind and Snow Drift Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_swe(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of SWE (Snow Water Equivalent) Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_penetration(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Snow Penetration Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_wet_snow(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Wet Snow Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_temperature_gradient(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Temperature Gradient Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_snow_temperature(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Snow Temperature Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_MFcrust(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Melt-Freeze Crust Features")

    df_before = mod1_features.copy()
    mod1_features = calculate_avalanche_days(mod1_features)
    plot_new_columns(df_before, mod1_features,
                     title="Boxplot of Avalanche Days Features")

    mod1_features = mod1_features.drop(columns=['rho', 'B', 'L1', 'L2'])

    # --- BASIC STATISTICS OF FEATURES ---
    summary_stats = mod1_features.describe().transpose()

    # --- DATA SAVING ---

    save_mod1_features(mod1_features, output_filepath)
    save_mod1_features(summary_stats, summary_filepath)


if __name__ == '__main__':
    main()
