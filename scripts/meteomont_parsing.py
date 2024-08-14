# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:09:39 2024

@author: Christian
"""
import pandas as pd
from pathlib import Path
import numpy as np


def read_csv(file):

    try:
        df = pd.read_csv(file, sep=';', decimal='.', na_values='NaN')

    except UnicodeDecodeError as e:
        print(f"Error decoding the file: {e}")

    df.set_index('Date time', inplace=True)
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')

    return df


def hourly_resample(df, freq):
    # Set the sampling frequency
    sampling_frequency = freq

    # Calculate the expected number of samples per hour
    samples_per_hour = pd.Timedelta('3h') // pd.Timedelta(sampling_frequency)

    # Set the threshold to 75% of valid data
    threshold = 0.75

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
        resampler = df.resample('h')
        resampled_df = resampler.mean()
        time_index = resampled_df.index

        hourly_groups = df.resample('h')[variable]

        if variable == 'P':
            resampled_data[variable] = hourly_groups.sum()
            # resampled_data[variable] = hourly_groups.apply(
            #     lambda group: filter_and_resample(group, np.sum))
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index
        elif variable in ['Ta', 'RH', 'SR', 'HS', 'Ts']:
            resampled_data[variable] = hourly_groups.apply(
                lambda group: filter_and_resample(group, np.mean))
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index

        elif variable == 'WD':
            resampled_data[variable] = hourly_groups.apply(
                resample_wind_direction)
            resampled_data[variable].name = variable
            resampled_data[variable].index = time_index
        elif variable == 'WV':
            hourly_resampled_mean = hourly_groups.apply(
                lambda group: filter_and_resample(group, np.mean))
            hourly_resampled_gust = hourly_groups.apply(
                lambda group: filter_and_resample(group, np.max))
            resampled_data['WV_avg'] = hourly_resampled_mean
            resampled_data['WV_gust'] = hourly_resampled_gust
            resampled_data['WV_avg'].name = 'WV_avg'
            resampled_data['WV_gust'].name = 'WV_gust'
            resampled_data['WV_avg'].index = time_index
            resampled_data['WV_gust'].index = time_index

    # Combine all resampled data into a single DataFrame
    resampled_df = pd.concat(resampled_data.values(), axis=1)
    resampled_df.columns = resampled_data.keys()

    # Convert index to DateTimeIndex for consistency
    resampled_df.index = pd.to_datetime(resampled_df.index)

    return resampled_df


def main_meteomont():
    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Dati_meteo\\Meteomont\\")

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
        for file in files_for_first_station:
            print('--------------')
            station = file.name.split('_')[0]
            freq = file.name.split('_')[1][0:-4]
            print(f'Analyzing: {file.name}')
            print(f'Sampling frequency: {freq}')
            df = read_csv(file)
            df_hourly = hourly_resample(df, freq)
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

            combined_df.to_csv(
                data_folder / f'Results/{station}.csv',
                index_label='Date',
                sep='\t',
                na_rep='-999',
                date_format='%Y-%m-%dT%H:%M')


if __name__ == '__main__':
    main_meteomont()
