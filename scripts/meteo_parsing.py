# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:15:49 2024

@author: Christian
"""

import pandas as pd
from pathlib import Path
import numpy as np


def read_csv(file, variable):

    try:
        df = pd.read_csv(file, sep=';')

    except UnicodeDecodeError as e:
        print(f"Error decoding the file: {e}")
    df = df.iloc[:, :-1]

    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')

    df.columns.values[0] = variable

    return df


def hourly_resample(df, variable, freq):
    # Set the sampling frequency
    sampling_frequency = freq

    # Calculate the expected number of samples per hour
    samples_per_hour = pd.Timedelta('1h') // pd.Timedelta(sampling_frequency)

    # Set the threshold to 75% of valid data
    threshold = 0.75

    def filter_and_resample(group, agg_func):
        non_nan_count = group[variable].count()
        if non_nan_count < threshold * samples_per_hour:
            return pd.Series([np.nan], index=[variable])
        else:
            return pd.Series([agg_func(group[variable])], index=[variable])

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

    # Group by hours and apply the function
    hourly_groups = df.resample('h')

    if variable == 'P':
        hourly_resampled = hourly_groups.sum()
        # hourly_resampled = hourly_groups.apply(
        #     lambda group: filter_and_resample(group, np.sum))
    elif variable in ['Ta', 'RH', 'SR']:
        hourly_resampled = hourly_groups.apply(
            lambda group: filter_and_resample(group, np.mean))
    elif variable == 'WD':
        hourly_resampled = hourly_groups[variable].apply(
            resample_wind_direction)
    elif variable == 'WV':
        hourly_resampled_mean = hourly_groups.apply(
            lambda group: filter_and_resample(group, np.mean))
        hourly_resampled_gust = hourly_groups.apply(
            lambda group: filter_and_resample(group, np.max))
        hourly_resampled = pd.concat(
            [hourly_resampled_mean, hourly_resampled_gust], axis=1)
        hourly_resampled.columns = ['WV_avg', 'WV_gust']

    # Convert index to DateTimeIndex for consistency
    hourly_resampled.index = pd.to_datetime(hourly_resampled.index)

    return hourly_resampled


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
    return missing_values, missing_value_percent, total_samples


def main_meteotrentino():
    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Dati_meteo\\")

    files = [file for file in data_folder.iterdir() if file.is_file()]

    # Extract file names
    file_names = [file_path.name for file_path in files]
    station_names = [file_name.split('_')[0] for file_name in file_names]
    unique_station_names = list(set(station_names))

    for first_station_name in unique_station_names:
        print('*************')
        print(first_station_name)

        # Filter files that start with the first unique station name
        files_for_first_station = [
            file for file in files if file.name.startswith(first_station_name + '_')]

        # Print the filtered file paths
        combined_df = pd.DataFrame()

        # Dictionary to store missing data info
        missing_data_info = {}

        for file in files_for_first_station:
            print('--------------')
            station = file.name.split('_')[0]
            variable = file.name.split('_')[1]
            freq = file.name.split('_')[2][0:-4]
            if freq == '5m':
                freq = '5min'
            elif freq == '10m':
                freq = '10min'
            elif freq == '15m':
                freq = '15min'
            elif freq == '30m':
                freq = '30min'

            print(f'Analyzing: {file.name}')
            print(f'Variable: {variable}. Sampling frequency: {freq}')
            df = read_csv(file, variable)
            df_hourly = hourly_resample(df, variable, freq)

            # Count missing values for each station
            missing_values, missing_value_percent, total_samples = count_nan(
                df, freq)
            # Store missing data info
            missing_data_info[file.name] = {
                'Variable': variable,
                'MissingValues': missing_values[variable],
                'TotalSamples': total_samples
            }

            combined_df = pd.concat([combined_df, df_hourly], axis=1)

            duplicates = combined_df.columns[combined_df.columns.duplicated(
            )].unique()
            for dup in duplicates:
                tmp = combined_df[dup]
                tmp['comb'] = tmp.iloc[:, 0].combine_first(tmp.iloc[:, 1])
                tmp = tmp.drop(columns=dup)
                tmp.columns = [dup]
                combined_df = combined_df.drop(columns=dup)
                combined_df = pd.concat([combined_df, tmp], axis=1)

        missing_data_info_df = pd.DataFrame(missing_data_info)

        missing_data_aggreg = missing_data_info_df.T.groupby(
            'Variable', as_index=False).sum()
        missing_data_aggreg['Percentage'] = 100 * \
            missing_data_aggreg['MissingValues'] / \
            missing_data_aggreg['TotalSamples']

        # Write Outputs
        combined_df.to_csv(
            data_folder / f'Results/{station}.csv',
            index=True,
            index_label='Date',
            sep='\t',
            na_rep='-999',
            date_format='%Y-%m-%dT%H:%M')

        missing_data_aggreg.to_csv(
            data_folder / f'Results/{station}_NanStatistics.csv',
            index=False)


if __name__ == '__main__':
    main_meteotrentino()
