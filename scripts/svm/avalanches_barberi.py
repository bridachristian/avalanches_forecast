# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 15:27:32 2025

@author: Christian
"""

import pandas as pd
from pathlib import Path


# --- PATHS ---

# Filepath and plot folder paths
common_path = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\AvalanchesEvents_Barberi\\')

# filepath = common_path / 'mod1_newfeatures_NEW.csv'
filepath = common_path / 'valanghepeio.csv'
results_path = common_path

# Carica i dati (sostituisci con il tuo DataFrame)
df = pd.read_csv(filepath,  sep=';')  # o pd.read_excel(...)

# Conversione della colonna data
df['avalancheEventA1_DATA'] = pd.to_datetime(
    df['avalancheEventA1_DATA'], format="%d/%m/%Y")

# Raggruppa per data evento e aggrega le zone interessate
grouped = df.groupby('avalancheEventA1_DATA').agg({
    'Zona': lambda x: ', '.join(sorted(set(x))),
}).reset_index()

# Ordina per data
grouped = grouped.sort_values('avalancheEventA1_DATA')

# Lista delle zone da cercare
target_zones = ["CV", "DG", "SX", "V", "VT"]

# Crea una colonna flag: True se almeno una zona è contenuta, altrimenti False
grouped['flag'] = grouped['Zona'].apply(
    lambda z: any(tz in z.split(", ") for tz in target_zones))

# Visualizza il risultato
print(grouped.head())

grouped.to_csv(results_path / "valanghe_per_data2.csv", index=False)
