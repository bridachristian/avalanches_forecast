# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:08:21 2024

@author: Christian
"""
import pandas as pd
from pathlib import Path
import numpy as np


def read_csv(file):

    try:
        df = pd.read_csv(file, sep=';')

    except UnicodeDecodeError as e:
        print(f"Error decoding the file: {e}")
    # df = df.iloc[:, :-1]

    df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M')

    return df


data_folder = Path(
    "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Dati_meteo\\06_DQC_level6_meteoIOinterpolation\\csv")

# List all files in the folder, excluding '.ini' files
files = [file for file in data_folder.iterdir() if file.is_file()]


# Print the list of files
for file in files:
    print(file)
    station = file.stem
    df = read_csv(file)
    hourly_groups = df.resample('h').mean()

    # Write Outputs
    hourly_groups.to_csv(
        data_folder / f'{station}_resampled.csv',
        index=True,
        index_label='timestamp',
        sep='\t',
        na_rep='-999',
        date_format='%Y-%m-%dT%H:%M')
