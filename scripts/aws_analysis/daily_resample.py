# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:03:34 2024

@author: Christian
"""

import pandas as pd
from pathlib import Path
import numpy as np


def read_csv(file):

    try:
        df = pd.read_csv(file, sep=';', na_values='-999')

    except UnicodeDecodeError as e:
        print(f"Error decoding the file: {e}")
    # df = df.iloc[:, :-1]

    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M')

    return df


def resample(df, freq, output_freq):
    # Set the sampling frequency
    sampling_frequency = freq

    # Calculate the expected number of samples per hour
    samples_per_hour = pd.Timedelta(
        output_freq) // pd.Timedelta(sampling_frequency)

    # Set the threshold to 75% of valid data
    threshold = 0.50

    def filter_and_resample(group, agg_func):
        non_nan_count = group.count()
        if non_nan_count < threshold * samples_per_hour:
            return pd.Series([np.nan], index=[variable])
        else:
            return pd.Series([agg_func(group)], index=[variable])

    def resample_wind_direction(series):
        # Extract wind direction values, ensuring it is a NumPy array
        wd_values = series.dropna().values

        non_nan_count = len(wd_values)
        if non_nan_count < threshold * samples_per_hour:
            mean_wd_deg = np.nan
        else:

            if len(wd_values) == 0:
                return np.nan

            # Convert wind direction to radians
            wd_rad = np.deg2rad(wd_values)

            # Calculate vector components
            u = np.sin(wd_rad)
            v = np.cos(wd_rad)

            # Average the components
            u_mean = np.mean(u)
            v_mean = np.mean(v)

            # Calculate the average wind direction
            mean_wd_rad = np.arctan2(u_mean, v_mean)
            # Ensure the result is within [0, 360) range
            mean_wd_deg = np.rad2deg(mean_wd_rad) % 360

        return mean_wd_deg

    # Dictionary to store the resampled data
    resampled_data = {}

    for variable in df.columns:
        resampler = df.resample(output_freq)
        resampled_df = resampler.mean()
        time_index = resampled_df.index

        groups = df.resample(output_freq)[variable]

        if variable == 'PSUM':
            # resampled_data[variable] = groups.sum()
            resampled_data[variable] = groups.apply(
                lambda group: filter_and_resample(group, np.sum))
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index
        elif variable in ['TA', 'RH', 'ISWR', 'HS', 'TGS']:
            resampled_data[variable] = groups.apply(
                lambda group: filter_and_resample(group, np.mean))
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index

        elif variable == 'DW':
            resampled_data[variable] = groups.apply(
                resample_wind_direction)
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index
        elif variable in ['VW', 'VW_MAX']:
            hourly_resampled_mean = groups.apply(
                lambda group: filter_and_resample(group, np.mean))
            hourly_resampled_gust = groups.apply(
                lambda group: filter_and_resample(group, np.max))
            resampled_data['VW'] = hourly_resampled_mean
            resampled_data['VW_MAX'] = hourly_resampled_gust
            resampled_data['VW'].name = 'VW'
            resampled_data['VW_MAX'].name = 'VW_MAX'
            resampled_data['VW'].index = time_index
            resampled_data['VW_MAX'].index = time_index

    # Combine all resampled data into a single DataFrame
    resampled_df = pd.concat(resampled_data.values(), axis=1)
    resampled_df.columns = resampled_data.keys()

    # Convert index to DateTimeIndex for consistency
    resampled_df.index = pd.to_datetime(resampled_df.index)

    return resampled_df


def count_nan(df, freq):

    # Check for duplicate indices
    if df.index.duplicated().any():
        print("Duplicate indices found. Aggregating data...")
        # Aggregating duplicates (e.g., taking the mean of duplicated rows)
        df = df.groupby(df.index).mean()

    full_date_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=freq)

    df_regularized = df.reindex(full_date_range)

    # Count missing value
    missing_values = df_regularized.isna().sum()

    missing_value_percent = 100 * missing_values / len(full_date_range)
    total_samples = len(full_date_range)

    # Convert missing values to DataFrame
    missing_data_df = pd.DataFrame(missing_values)

    # Add total_samples as a new row
    missing_data_df.loc['TotalSamples'] = [total_samples]

    return missing_data_df


data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Dati_meteo\\07_Daily_resample\\meteomont3h\\")

# List all files in the folder, excluding '.ini' files
files = [file for file in data_folder.iterdir() if file.is_file()]

freq = '3h'
output_freq = '1d'

# Print the list of files
for file in files:
    station = file.stem
    print(station)

    df = read_csv(file)

    df.index = df.index - pd.DateOffset(hours=9)

    df_daily = resample(df, freq, output_freq)

    # Count missing values for each station
    missing_values_before = count_nan(df, freq)

    missing_values_after = count_nan(df_daily, output_freq)

    # Write Outputs
    df_daily.to_csv(
        data_folder / f'Results\{station}_daily.csv',
        index=True,
        index_label='timestamp',
        sep=';',
        na_rep='NaN',
        date_format='%Y-%m-%d %H:%M')

    missing_values_before.to_csv(
        data_folder / f'Results\{station}_{freq}_NanStatistics.csv',
        index=True)

    missing_values_after.to_csv(
        data_folder / f'Results\{station}_{output_freq}_NanStatistics.csv',
        index=True)
