# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:38:13 2024

@author: Christian
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


filepath = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_tarlenta_01dec_15apr.csv')

mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

mod1['TmaxG'] = np.where(mod1['Tmax'] < 81, np.where(
    mod1['Tmax'] >= 50, -(mod1['Tmax'] - 50), mod1['Tmax']), np.nan)


mod1['TH01G'] = np.where(mod1['TH010'] < 81, np.where(
    mod1['TH010'] >= 50, -(mod1['TH010'] - 50), mod1['TH010']), np.nan)

mod1['TH03G'] = np.where(mod1['TH030'] < 81, np.where(
    mod1['TH030'] >= 50, -(mod1['TH030'] - 50), mod1['TH030']), np.nan)

mod1['HS'] = np.where(mod1['HS'] == 999, np.nan, mod1['HS'])
mod1['HN'] = np.where(mod1['HN'] >= 800, np.nan, mod1['HN'])

mod1['DataRilievo'] = pd.to_datetime(
    mod1['DataRilievo'], format='%d/%m/%Y')

mod1.set_index('DataRilievo', inplace=True)

print(mod1.dtypes)

mod1_subset = mod1[['N', 'V',
                   'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HS', 'HN', 'rho', 'TH01G', 'TH03G', 'B']]
statistics = mod1_subset.describe()


window_size = 3  # Define the window size for the moving average
mod1_subset['HSma3'] = mod1_subset['HS'].rolling(window=window_size).mean()
mod1_subset['HSdiff'] = mod1_subset['HS'].diff()

mod1_subset['HNma3'] = mod1_subset['HN'].rolling(window=window_size).mean()
mod1_subset['HN3gg'] = mod1_subset['HN'].rolling(window=window_size).sum()

mod1_subset['TAma3'] = mod1_subset['TaG'].rolling(window=window_size).mean()
mod1_subset['Tmin3gg'] = mod1_subset['TminG'].rolling(window=window_size).min()
mod1_subset['Tmax3gg'] = mod1_subset['TmaxG'].rolling(window=window_size).max()
mod1_subset['Tdelta3gg'] = mod1_subset['Tmax3gg'] - mod1_subset['Tmin3gg']


# x = mod1['TaG'].dropna()
# y = mod1['TminG'].dropna()

# scipy.stats.pearsonr(x, y)    # Pearson's r
# scipy.stats.spearmanr(x, y)   # Spearman's rho
# scipy.stats.kendalltau(x, y)  # Kendall's tau


# # Create scatter plot between the two time series
# plt.scatter(mod1['TaG'], mod1['TminG'], alpha=0.1)
# plt.scatter(mod1['TaG'], mod1['TmaxG'], alpha=0.1)
# plt.scatter(mod1['TaG'], mod1['HS'], alpha=0.1)
# # Add labels and title
# plt.xlabel('Series1')
# plt.ylabel('Series2')
# plt.title('Scatter Plot between Series1 and Series2')

# # Show the plot
# plt.show()


# ----------------------------------------------------------

# Example DataFrame (you can use mod1_subset or any other dataset)
# Assuming mod1_subset is already created
data = mod1_subset.drop(mod1_subset.columns[0], axis=1)

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.dropna())  # Dropping NaN if present

# Step 2: Perform PCA (reduce to 2 principal components for 2D plotting)
pca = PCA(n_components=12)
pca_components = pca.fit_transform(data_scaled)

# Step 3: Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=pca_components, columns=[
                      'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])

# Step 4: Plot the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2 Principal Components')
plt.grid(True)
plt.show()

explained_variance = pca.explained_variance_ratio_

# Step 4: Plot the explained variance ratio of the 12 components
plt.figure(figsize=(10, 6))
plt.bar(range(1, 13), explained_variance,
        alpha=0.7, align='center', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components (1-12)')
plt.xticks(np.arange(1, 13))
plt.grid(True)
plt.show()

print(pca.explained_variance_ratio_)

# Step 3: Get the loadings for the first principal component
loadings = pca.components_[0]  # Loadings for PC1

# Create a DataFrame for easier interpretation
loadings_df = pd.DataFrame(
    loadings, index=data.columns, columns=['PC1_Loading'])

# Sort the loadings by absolute value to see the top contributors
sorted_loadings = loadings_df.reindex(
    loadings_df['PC1_Loading'].abs().sort_values(ascending=False).index)

print(sorted_loadings)


# Step 4: Get the loadings for the first two principal components
loadings = pca.components_.T  # Transpose to align with features
loadings_df = pd.DataFrame(loadings, index=data.columns,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])

# Step 5: Plot PC1 vs PC2
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.1,
            c='blue',  s=50)

# Add arrows and labels for each variable
for i, variable in enumerate(loadings_df.index):
    plt.text(loadings_df['PC1'][i]*10, loadings_df['PC2'][i]*10,
             variable, color='black', ha='center', va='center')
    plt.arrow(0, 0, loadings_df['PC1'][i]*10, loadings_df['PC2'][i]*10,
              head_width=0.5, head_length=0.5, fc='red', ec='red')


plt.xlabel(f'Principal Component 1 - ({explained_variance[0]:.3f})')
plt.ylabel(f'Principal Component 2- ({explained_variance[1]:.3f})')
plt.title('PCA: PC1 vs PC2 with Variable Contributions')
plt.grid(True)
plt.show()
