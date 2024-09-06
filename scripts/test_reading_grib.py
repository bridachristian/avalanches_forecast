# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:02:19 2024

@author: Christian
"""

import cfgrib
from pathlib import Path
import pandas as pd

data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\GapFillingERA5\\")

# file = data_folder / f'era5_Ta_2023.grib'
file = data_folder / f'era5_VW_max_2023.grib'

# Open the GRIB file
ds = cfgrib.open_dataset(file)

# Display the dataset
# print(ds)

# List all variables
print(ds.variables)

# Access a specific variable (e.g., 'temperature')
# temperature = ds['t2m']
gust = ds['i10fg']

# Plot the variable (requires matplotlib)
# temperature.plot()
# df_temp = temperature.to_dataframe()
df_gust = gust.to_dataframe()

# df_temp = df_temp['t2m']
df_gust = df_gust['i10fg']

# df_temp.to_csv('TA_era5.csv', index=True)
df_gust.to_csv('VWmax_era5.csv', index=True)

# Reset index to convert MultiIndex to columns
# df_temp = df_temp.reset_index()
df_gust = df_gust.reset_index()

# Set the 'time' column as the index
# df_temp = df_temp.set_index('time')
df_gust = df_gust.set_index('time')

# Ensure the index is a DatetimeIndex
# df_temp.index = pd.to_datetime(df_temp.index)
df_gust.index = pd.to_datetime(df_gust.index)

# df_daily = df_temp.resample('D').mean()
df_daily = df_gust.resample('D').mean()

# df_daily.to_csv(data_folder/f'TA_daily_era5.csv', index=True)
df_daily.to_csv(data_folder/f'VWmax_daily_era5.csv', index=True)


# measured data resampling
data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\GapFillingERA5\\")

filedata = data_folder / f'T0366.csv'


data = pd.read_csv(filedata, sep=';')

data = data.set_index('timestamp')
data.index = pd.to_datetime(data.index)

data_daily = data.resample('D').max()
data_daily.to_csv(data_folder/f'Daily_measured.csv', index=True)
