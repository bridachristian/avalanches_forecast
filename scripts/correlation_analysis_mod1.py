# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:38:13 2024

@author: Christian
"""

import seaborn as sns
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

mod1_subset = mod1[['Stagione', 'N', 'V',
                   'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']]
statistics = mod1_subset.describe()
print(mod1_subset.dtypes)


# ------ Data Manipulation ------
# Add snowdrift index based on VQ1
mod1_subset['SnowDrift'] = mod1_subset['VQ1'].map(
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 0})

# Map wind direction on VQ2
# mod1_subset['WD'] = mod1_subset['VQ2'].map(
#     {0: 'None', 1: 'N', 2: 'E', 3: 'S', 4: 'W', 5: 'All'})

# Add Snow Water Equivalent of fresh snow.
mod1_subset['rho_adjusted'] = np.where(
    mod1_subset['HNnum'] < 6, 100, mod1_subset['rho'])  # rho = 100 for HN < 6
mod1_subset['SWEnew'] = mod1_subset['HNnum']*mod1_subset['rho_adjusted']/100

mod1_subset['SWE_cumulative'] = mod1_subset.groupby('Stagione')[
    'SWEnew'].cumsum()

# Add avalanche day based on L1
mod1_subset['AvalDay'] = np.where(mod1_subset['L1'] >= 1, 1, mod1_subset['L1'])

# ------ Correlation Matrix------

mod1_final = mod1_subset[['N', 'V',
                          'SnowDrift', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'SWEnew', 'SWE_cumulative']]

corr_matrix = mod1_final.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)

# Add title
plt.title('Correlation Matrix ', size=16)

# Show the plot
plt.show()

# ------ Histograms ------

# 1. Descriptive statistics
descriptive_stats = mod1_final.describe()
print(descriptive_stats)

# 2. Plot histograms for each column
mod1_final.hist(bins=10, figsize=(15, 12), edgecolor='black', color='skyblue')

axes = mod1_final.hist(bins=20, figsize=(
    15, 12), edgecolor='black', color='skyblue')

# Set title for the overall figure
plt.suptitle('Histograms of observations', size=16)

# Adjust y-label for each plot and layout
for ax in axes.flatten():
    ax.set_ylabel('n.obs')  # Set y-axis label to 'n.obs'

# Adjust layout to ensure proper spacing for the title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


# ------ Principal Component Analysis------

# Example DataFrame (you can use mod1_subset or any other dataset)
# Assuming mod1_subset is already created
data = mod1_subset[['N', 'V',
                    'SnowDrift', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'SWEnew', 'SWE_cumulative']]

# ************* 2D PCA ***********

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.dropna())  # Dropping NaN if present

# Step 2: Perform PCA (reduce to 2 principal components for 2D plotting)
pca = PCA(n_components=16)
pca_components = pca.fit_transform(data_scaled)

# Step 3: Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=pca_components, columns=[
                      'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16'])

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
plt.bar(range(1, 17), explained_variance,
        alpha=0.7, align='center', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components (1-16)')
plt.xticks(np.arange(1, 17))
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
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16'])

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


# ************* 3D PCA ***********
# Step 2: Perform PCA (reduce to 3 principal components for 3D plotting)
pca = PCA(n_components=3)
pca_components = pca.fit_transform(data_scaled)

# Step 3: Create a DataFrame for the first 3 principal components
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2', 'PC3'])

# Step 4: Plot the PCA result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the points
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
           c='blue', edgecolor='k', s=50)

# Set labels
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')

# Set plot title
ax.set_title('PCA - First 3 Principal Components')

# Show plot
plt.show()

# Step 4: Get the loadings for the first two principal components
loadings = pca.components_.T  # Transpose to align with features
loadings_df = pd.DataFrame(loadings, index=data.columns,
                           columns=['PC1', 'PC2', 'PC3'])

# Step 5: Plot PC1 vs PC2 vs PC3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(pca_df['PC1'], pca_df['PC2'],
           pca_df['PC3'], alpha=0.1, c='blue', s=50)

# Add arrows and labels for each variable
for i, variable in enumerate(loadings_df.index):
    ax.text(loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i]*10, loadings_df['PC3'].iloc[i]*10,
            variable, color='black', ha='center', va='center')

    ax.quiver(0, 0, 0,
              loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i] *
              10, loadings_df['PC3'].iloc[i]*10,
              arrow_length_ratio=0.1, color='red')

# Set labels and title with label padding
ax.set_xlabel(
    f'Principal Component 1 - ({explained_variance[0]:.3f})')
ax.set_ylabel(
    f'Principal Component 2 - ({explained_variance[1]:.3f})')
ax.set_zlabel(
    f'Principal Component 3 - ({explained_variance[2]:.3f})')
ax.set_title('PCA: PC1 vs PC2 vs PC3 with Variable Contributions')

plt.grid(True)
plt.show()


# ------ New variable creation ------
mod1_features = mod1_subset[['N', 'V',
                             'SnowDrift', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'SWEnew']]


# .... Snow Height ....

# Difference in snow height in 1 day, 2 days, 3 days, 5 days.
mod1_features['HSdiff24h'] = mod1_features['HSnum'].diff(periods=1)
mod1_features['HSdiff48h'] = mod1_features['HSnum'].diff(periods=2)
mod1_features['HSdiff72h'] = mod1_features['HSnum'].diff(periods=3)
mod1_features['HSdiff120h'] = mod1_features['HSnum'].diff(periods=5)

# .... New Snow ....

# New snow 2 days, 3 days, 5 days.
mod1_features['HN48h'] = mod1_features['HNnum'].rolling(window=2).sum()
mod1_features['HN72h'] = mod1_features['HNnum'].rolling(window=3).sum()
mod1_features['HN120h'] = mod1_features['HNnum'].rolling(window=5).sum()

# .... Air temperaure ....

# Air temperature min/max 2 days, 3 days, 5 days.
mod1_features['Tmin48h'] = mod1_features['TminG'].rolling(window=2).min()
mod1_features['Tmax48h'] = mod1_features['TmaxG'].rolling(window=2).max()

mod1_features['Tmin72h'] = mod1_features['TminG'].rolling(window=3).min()
mod1_features['Tmax72h'] = mod1_features['TmaxG'].rolling(window=3).max()

mod1_features['Tmin120h'] = mod1_features['TminG'].rolling(window=5).min()
mod1_features['Tmax120h'] = mod1_features['TmaxG'].rolling(window=5).max()

# Air temperature amplitude 1 day, 2 days, 3 days, 5 days.

mod1_features['Tdelta24h'] = mod1_features['TmaxG'] - mod1_features['TminG']
mod1_features['Tdelta48h'] = mod1_features['Tmax48h'] - \
    mod1_features['Tmin48h']
mod1_features['Tdelta72h'] = mod1_features['Tmax72h'] - \
    mod1_features['Tmin72h']
mod1_features['Tdelta120h'] = mod1_features['Tmax120h'] - \
    mod1_features['Tmin120h']

# .... Precipitation ....

# Precipitation (mm) from SWE 2 days, 3 days, 5 days.
mod1_features['PSUM24h'] = mod1_features['SWEnew']
mod1_features['PSUM48h'] = mod1_features['SWEnew'].rolling(window=2).sum()
mod1_features['PSUM72h'] = mod1_features['SWEnew'].rolling(window=3).sum()
mod1_features['PSUM120h'] = mod1_features['SWEnew'].rolling(window=5).sum()

# # Numbers of days since last snow
# mod1_features['DaysSinceLastSnow'] = np.nan
# last_nonzero_index = None

# # Replace NaNs with 0 in 'SWEnew' for processing
# mod1_features['SWEnew_modif'] = mod1_features['SWEnew'].fillna(0)

# # Calculate the number of days since the last non-zero SWEnew
# for i, row in mod1_features.iterrows():
#     if row['SWEnew_modif'] != 0:
#         if last_nonzero_index is not None:
#             # Calculate the difference in days since the last non-zero SWEnew
#             mod1_features.at[i, 'DaysSinceLastSnow'] = i - last_nonzero_index
#         # Update the last non-zero SWEnew index
#         last_nonzero_index = i

# # Set values in 'DaysSinceLastSnow' to NaN if greater than 200 days
# mod1_features['DaysSinceLastSnow'] = np.where(
#     mod1_features['DaysSinceLastSnow'] > 200, np.nan, mod1_features['DaysSinceLastSnow'])


# .... Wet Snow ....

mod1_features['WetSnow'] = np.where(mod1_features['CS'] >= 20, 1, 0)
mod1_features['WetSnow'] = np.where(
    mod1_features['CS'].isna(), np.nan, mod1_features['WetSnow'])

# .... Temperature Gradient ....

mod1_features['Gradient'] = abs(mod1_features['TH01G']) / \
    (mod1_features['HSnum'] - 10)
