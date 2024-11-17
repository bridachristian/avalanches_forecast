# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:00:55 2024

@author: Christian
"""
import pandas as pd
from pathlib import Path


data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Mod1_Tarlenta\\")
files = [file for file in data_folder.iterdir() if file.is_file()
         and "mod1_tarlenta_merged.csv" not in file.name]

# Initialize an empty dataframe to store the combined data
dfs = []

file = files[0]  # to debug

for file in files:
    df = pd.read_csv(file, sep=';')
    # Convert 'Date' column to datetime
    df['DataRilievo'] = pd.to_datetime(
        df['DataRilievo'], format="%d/%m/%Y %H:%M")

    # Set 'Date' as index
    df.set_index('DataRilievo', inplace=True)

    # Define the start and end dates
    start_date = df.index.min()
    end_date = df.index.max()

    # Create a DataFrame with a regular time series
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    regular_df = pd.DataFrame(index=date_index)

    # Fill missing dates with NaN
    merged_df = regular_df.merge(
        df, how='left', left_index=True, right_index=True)

    dfs.append(merged_df)


combined_df = pd.concat(dfs, ignore_index=False)

# print(combined_df)

# combined_df.to_csv(data_folder / 'mod1_tarlenta_merged.csv',
#                    index=True, index_label='DataRilievo', sep=';')
