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

    # Group by hours and apply the function
    hourly_groups = df.resample('h')

    if variable == 'P':
        hourly_resampled = hourly_groups.apply(
            lambda group: filter_and_resample(group, np.sum))
    elif variable in ['Ta', 'RH', 'SR']:
        hourly_resampled = hourly_groups.apply(
            lambda group: filter_and_resample(group, np.mean))
    elif variable == 'WD':
        hourly_resampled = np.nan  # Placeholder for actual wind direction resampling logic
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


data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Dati_meteo\\")


files = [file for file in data_folder.iterdir() if file.is_file()]

file = files[9]

print(file.name)
station = file.name.split('_')[0]
variable = file.name.split('_')[1]
freq = file.name.split('_')[2][0:-4]

df = read_csv(file, variable)


df_hourly = hourly_resample(df, variable, freq)


# df_hourly.to_csv(data_folder / 'test.csv',index=True, index_label='Date', sep=';')
