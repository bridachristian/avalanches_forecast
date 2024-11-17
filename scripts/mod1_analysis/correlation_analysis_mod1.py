# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:38:13 2024

@author: Christian
"""

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


filepath = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_tarlenta_01dec_15apr.csv')

plot_folder = Path(
    'C:/Users/Christian/OneDrive/Desktop/Family/Christian/MasterMeteoUnitn/Corsi/4_Tesi/05_Plots/03_Correlation_mod1')

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
                   'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1']]
statistics = mod1_subset.describe()
print(mod1_subset.dtypes)


# Add avalanche day based on L1
mod1_subset['AvalDay'] = np.where(mod1_subset['L1'] >= 1, 1, mod1_subset['L1'])

# ------ Correlation Matrix------

mod1_final = mod1_subset.drop(columns=['Stagione', 'L1', 'AvalDay'])
corr_matrix = mod1_final.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(8, 8))

# Draw the heatmap with the mask
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)

# Add title
plt.title('Correlation Matrix ', size=16)

# Show the plot
# plt.show()

# # Use seaborn's clustermap to include dendrograms
# sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1,
#                vmax=1, fmt='.2f', linewidths=0.5, figsize=(10, 10))

# # Add title (matplotlib's clustermap doesn't directly support titles, so using plt.title is required)
# plt.title('Correlation Matrix with Dendrogram',
#           size=16, y=1.05)  # Adjust the title position

# # Show the plot
# plt.show()

# Correcting the path without extra quotes
outpath = plot_folder / 'correlation_matrix.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

# ------ Histograms ------

# 1. Descriptive statistics
descriptive_stats = mod1_final.describe()

print(descriptive_stats)

# 2. Plot histograms for each column
# Plotting histograms
mod1_final.hist(bins=10, figsize=(8, 8), edgecolor='black', color='skyblue')

axes = mod1_final.hist(bins=20, figsize=(
    8, 8), edgecolor='black', color='skyblue')

# Set title for the overall figure
plt.suptitle('Histograms of observations', size=16)

# Adjust y-label for each plot and layout
for ax in axes.flatten():
    ax.set_ylabel('n.obs')  # Set y-axis label to 'n.obs'

# Adjust layout to ensure proper spacing for the title
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
# plt.show()

# Correcting the path without extra quotes
outpath = plot_folder / 'histogram_observation.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

# ------ Principal Component Analysis------

# Example DataFrame (you can use mod1_subset or any other dataset)
# Assuming mod1_subset is already created
data = mod1_subset[['N', 'V',
                    'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B']]

# ************* 2D PCA ***********

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.dropna())  # Dropping NaN if present

# Step 2: Perform PCA (reduce to 2 principal components for 2D plotting)
pca = PCA(n_components=15)
pca_components = pca.fit_transform(data_scaled)

# Step 3: Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=pca_components, columns=[
                      'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15'])

# Step 4: Plot the PCA result
plt.figure(figsize=(8, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2 Principal Components')
plt.grid(True)
plt.show()

explained_variance = pca.explained_variance_ratio_

# Step 4: Plot the explained variance ratio of the 12 components
plt.figure(figsize=(8, 8))
plt.bar(range(1, 16), explained_variance,
        alpha=0.7, align='center', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components (1-15)')
plt.xticks(np.arange(1, 16))
plt.grid(True)
# plt.show()

# Correcting the path without extra quotes
outpath = plot_folder / 'screeplot_PCA.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

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


loadings_dict = {}

# Loop through each component and store the loadings
for i in range(pca.n_components_):
    # Extract loadings for the ith principal component
    loadings = pca.components_[i]

    # Create a DataFrame for the loadings and add it to the dictionary
    loadings_df = pd.DataFrame(
        loadings, index=data.columns, columns=[f'PC{i+1}_Loading'])

    # Sort the loadings by absolute value to see the top contributors
    sorted_loadings_df = loadings_df.reindex(
        loadings_df[f'PC{i+1}_Loading'].abs().sort_values(ascending=False).index)

    # Store the sorted loadings in the dictionary
    loadings_dict[f'PC{i+1}'] = sorted_loadings_df

# Concatenate all loadings into a single DataFrame
all_loadings_df = pd.concat(loadings_dict, axis=1)

# Print the resulting DataFrame with loadings for all principal components
print(all_loadings_df)

# Step 4: Get the loadings for the first two principal components
loadings = pca.components_.T  # Transpose to align with features
loadings_df = pd.DataFrame(loadings, index=data.columns,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15'])

# Step 5: Plot PC1 vs PC2
plt.figure(figsize=(8, 8))
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
# plt.show()
outpath = plot_folder / 'piplot_PCA.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

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
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(pca_df['PC1'], pca_df['PC2'],
           pca_df['PC3'], alpha=0.1, c='blue', s=50)

# Add arrows and labels for each variable
for i, variable in enumerate(loadings_df.index):
    ax.text(loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i]*10, loadings_df['PC3'].iloc[i]*10,
            variable, color='black', ha='center', va='center')

    ax.quiver(0, 0, 0,
              loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i]*10,
              loadings_df['PC3'].iloc[i]*10,
              arrow_length_ratio=0.1, color='red')

# Set labels and title with label padding
# X-axis label padding
ax.set_xlabel(
    f'Principal Component 1 - ({explained_variance[0]:.3f})', labelpad=0)
# Y-axis label padding
ax.set_ylabel(
    f'Principal Component 2 - ({explained_variance[1]:.3f})', labelpad=0)
# Z-axis label padding
ax.set_zlabel(
    f'Principal Component 3 - ({explained_variance[2]:.3f})', labelpad=-10)

ax.set_title('PCA: PC1 vs PC2 vs PC3 with Variable Contributions')

# Manually adjust the layout to increase right padding
plt.subplots_adjust(left=0.1, right=0.85, top=0.9,
                    bottom=0.1)  # Increase right padding

# Optionally adjust the view to improve visibility of the labels
# ax.view_init(elev=30, azim=45)  # Adjust viewing angle if necessary

plt.grid(True)
# plt.show()
outpath = plot_folder / 'piplot_3d_PCA.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)


# ------ New variable creation ------
mod1_features = mod1[['Stagione', 'N', 'V',
                      'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1']]


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

# .... Wind ....
# Snow Drift based on VQ1
mod1_features['SnowDrift'] = mod1_features['VQ1'].map(
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 0})
mod1_features['SnowDrift48h'] = mod1_features['SnowDrift'].rolling(
    window=2).sum()
mod1_features['SnowDrift72h'] = mod1_features['SnowDrift'].rolling(
    window=3).sum()
mod1_features['SnowDrift120h'] = mod1_features['SnowDrift'].rolling(
    window=5).sum()


# Location of Wind Slab
mod1_features['WindSlab'] = mod1_features['VQ2'].map(
    {0: 'None', 1: 'N', 2: 'E', 3: 'S', 4: 'W', 5: 'All'})

# .... Precipitation ....

# Snow Water Equivalent from fresh snow
mod1_features['rho_adjusted'] = np.where(
    mod1_features['HNnum'] < 6, 100, mod1_features['rho'])  # rho = 100 for HN < 6
mod1_features['SWEnew'] = mod1_features['HNnum'] * \
    mod1_features['rho_adjusted']/100

mod1_features['SWE_cumulative'] = mod1_features.groupby('Stagione')[
    'SWEnew'].cumsum()

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

mod1_features['T_gradient'] = abs(mod1_features['TH01G']) / \
    (mod1_features['HSnum'] - 10)

mod1_features['T_gradient'] = np.where(
    mod1_features['T_gradient'] == np.inf, np.nan, mod1_features['T_gradient'])


# .... Surface Hoar ....

mod1_features['SH'] = mod1_features['B'].map({0: 0, 1: 1, 2: 1, 3: 1})

# .... Avalanche Observations ....

# Avalanche day based on L1
mod1_features['AvalDay'] = np.where(
    mod1_features['L1'] >= 1, 1, mod1_features['L1'])

mod1_features['AvalDay48h'] = mod1_features['AvalDay'].rolling(window=2).mean()
mod1_features['AvalDay72h'] = mod1_features['AvalDay'].rolling(window=3).mean()
mod1_features['AvalDay120h'] = mod1_features['AvalDay'].rolling(
    window=5).mean()

# ------ Correlation Matrix------

mod1_features_final = mod1_features.drop(columns=['Stagione', 'WindSlab'])
corr_matrix_features = mod1_features_final.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix_features, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

sns.heatmap(corr_matrix_features, mask=mask, annot=False, cmap='coolwarm',
            vmin=-1, vmax=1, fmt='.2f', linewidths=0.1,
            cbar_kws={'shrink': 0.8},  # Shrink the color bar slightly
            xticklabels=corr_matrix_features.columns, yticklabels=corr_matrix_features.columns)

# Decrease the size of x and y labels
plt.xticks(fontsize=8)  # Decrease x-label font size
plt.yticks(fontsize=8)  # Decrease y-label font size


# Add title
plt.title('Correlation Matrix', size=16)

# Show the plot
# plt.show()
outpath = plot_folder / 'correlation_newvar.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)
# ------ Principal Component Analysis------

# Example DataFrame (you can use mod1_subset or any other dataset)
# Assuming mod1_subset is already created
data = mod1_features.drop(columns=['Stagione', 'WindSlab'])
# ************* 2D PCA ***********

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.dropna())  # Dropping NaN if present

# Step 2: Perform PCA (reduce to 2 principal components for 2D plotting)
pca = PCA(n_components=10)
pca_components = pca.fit_transform(data_scaled)

# Step 3: Create a DataFrame for the principal components
pca_df = pd.DataFrame(data=pca_components, columns=[
                      'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

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
plt.bar(range(1, 11), explained_variance,
        alpha=0.7, align='center', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components (1-16)')
plt.xticks(np.arange(1, 11))
plt.grid(True)
# plt.show()
# Show the plot
# plt.show()
outpath = plot_folder / 'screeplot_PCA_newvar.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

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


loadings_dict = {}

# Loop through each component and store the loadings
for i in range(pca.n_components_):
    # Extract loadings for the ith principal component
    loadings = pca.components_[i]

    # Create a DataFrame for the loadings and add it to the dictionary
    loadings_df = pd.DataFrame(
        loadings, index=data.columns, columns=[f'PC{i+1}_Loading'])

    # Sort the loadings by absolute value to see the top contributors
    sorted_loadings_df = loadings_df.reindex(
        loadings_df[f'PC{i+1}_Loading'].abs().sort_values(ascending=False).index)

    # Store the sorted loadings in the dictionary
    loadings_dict[f'PC{i+1}'] = sorted_loadings_df

# Concatenate all loadings into a single DataFrame
all_loadings_df = pd.concat(loadings_dict, axis=1)

# Print the resulting DataFrame with loadings for all principal components
print(all_loadings_df)

# Step 4: Get the loadings for the first two principal components
loadings = pca.components_.T  # Transpose to align with features
loadings_df = pd.DataFrame(loadings, index=data.columns,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])

# Step 5: Plot PC1 vs PC2
plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.1,
            c='blue',  s=50)

# Add arrows and labels for each variable
for i, variable in enumerate(loadings_df.index):
    plt.text(loadings_df['PC1'][i]*10, loadings_df['PC2'][i]*30,
             variable, color='black', ha='center', va='center')
    plt.arrow(0, 0, loadings_df['PC1'][i]*10, loadings_df['PC2'][i]*30,
              head_width=0.5, head_length=0.5, fc='red', ec='red')


plt.xlabel(f'Principal Component 1 - ({explained_variance[0]:.3f})')
plt.ylabel(f'Principal Component 2- ({explained_variance[1]:.3f})')
plt.title('PCA: PC1 vs PC2 with Variable Contributions')
plt.grid(True)
# plt.show()
outpath = plot_folder / 'piplot_PCA_newvar.png'

# Save the plot with high resolution
plt.savefig(outpath, dpi=300)

# ---- Make plot for avalanche days or not -------

mod1_compare = mod1_subset.drop(columns=['Stagione', 'L1'])
# mod1_compare.hist(bins=20, figsize=(8, 8), edgecolor='black', color='skyblue')

# Separate data based on AvalDays
data_aval_0 = mod1_compare[mod1_compare['AvalDay'] == 0]
data_aval_1 = mod1_compare[mod1_compare['AvalDay'] == 1]

data_aval_0 = data_aval_0.drop(columns=['AvalDay'])
data_aval_1 = data_aval_1.drop(columns=['AvalDay'])

# density plot

# Set up the figure for 4 rows and 4 columns layout
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 8))
# Adjust right side for legend space
fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

# Flatten axes for easy iteration
axes = axes.flatten()

# Loop through each column to plot its distribution
for i, col in enumerate(data_aval_0.columns):
    sns.kdeplot(data_aval_0[col], ax=axes[i], color='blue',
                label='AvalDays = 0', fill=True, alpha=0.5)
    sns.kdeplot(data_aval_1[col], ax=axes[i], color='red',
                label='AvalDays = 1', fill=True, alpha=0.5)
    # Remove the y-axis labels for the central and right columns
    if i % 3 != 0:  # Central and right columns
        axes[i].set_ylabel('')

    # axes[i].set_title(f'Density Plot of {col}', fontsize=10)

# Remove empty subplots if mod1_compare has fewer than 16 columns
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Create a single legend for all plots, on the right side
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left',
           bbox_to_anchor=(0.9, 0.5), borderaxespad=0, ncol=1)

# Add overall title
plt.suptitle('Density Plots for AvalDays = 0 and AvalDays = 1', fontsize=16)

# Display the plot
plt.show()

# histogram
# Set up the figure for 5 rows and 3 columns layout
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 8))
# Adjust right side for legend space
fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

# Flatten axes for easy iteration
axes = axes.flatten()

# Loop through each column to plot its histogram based on count
for i, col in enumerate(data_aval_0.columns):
    sns.histplot(data_aval_0[col], ax=axes[i], color='blue',
                 label='AvalDays = 0', bins=20, alpha=0.5, stat='count')
    sns.histplot(data_aval_1[col], ax=axes[i], color='red',
                 label='AvalDays = 1', bins=20, alpha=0.5, stat='count')

    # Remove the y-axis labels for the central and right columns
    if i % 3 != 0:  # Central and right columns
        axes[i].set_ylabel('')

# Remove empty subplots if there are fewer than 15 columns
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Create a single legend for all plots, on the right side
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left',
           bbox_to_anchor=(0.9, 0.5), borderaxespad=0, ncol=1)

# Add overall title
plt.suptitle('Histogram Plots for AvalDays = 0 and AvalDays = 1', fontsize=16)

# Display the plot
plt.show()
