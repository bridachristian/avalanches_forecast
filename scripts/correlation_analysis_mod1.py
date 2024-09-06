# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:38:13 2024

@author: Christian
"""

import scipy.stats
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


filepath = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_tarlenta.csv')

mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

mod1['TmaxG'] = np.where(mod1['Tmax'] < 81, np.where(
    mod1['Tmax'] >= 50, -(mod1['Tmax'] - 50), mod1['Tmax']), np.nan)


mod1['TH01G'] = np.where(mod1['TH010'] < 81, np.where(
    mod1['TH010'] >= 50, -(mod1['TH010'] - 50), mod1['TH010']), np.nan)

mod1['TH03G'] = np.where(mod1['TH030'] < 81, np.where(
    mod1['TH030'] >= 50, -(mod1['TH030'] - 50), mod1['TH030']), np.nan)

print(mod1.dtypes)

x = mod1['TaG'].dropna()
y = mod1['TminG'].dropna()

scipy.stats.pearsonr(x, y)    # Pearson's r
scipy.stats.spearmanr(x, y)   # Spearman's rho
scipy.stats.kendalltau(x, y)  # Kendall's tau


# Create scatter plot between the two time series
plt.scatter(mod1['TaG'], mod1['TminG'], alpha=0.1)
plt.scatter(mod1['TaG'], mod1['TmaxG'], alpha=0.1)
plt.scatter(mod1['TaG'], mod1['HS'], alpha=0.1)
# Add labels and title
plt.xlabel('Series1')
plt.ylabel('Series2')
plt.title('Scatter Plot between Series1 and Series2')

# Show the plot
plt.show()
