# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:02:20 2024

@author: Christian
"""

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean_data(filepath):
    """Load and clean the dataset from the given CSV file."""
    mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    # Applying transformations to the dataset
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

    # Set 'DataRilievo' as the index
    mod1.set_index('DataRilievo', inplace=True)

    return mod1


def create_subset(mod1):
    """Create a subset of relevant columns and add the 'AvalDay' column."""
    mod1_subset = mod1[['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG',
                        'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1']]

    # Add 'AvalDay' column based on L1
    mod1_subset['AvalDay'] = np.where(
        mod1_subset['L1'] >= 1, 1, mod1_subset['L1'])

    return mod1_subset


def plot_correlation_matrix(mod1_subset, plot_folder):
    """Plot and save the correlation matrix heatmap."""
    # Drop categorical or unnecessary columns for correlation
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

    # Save the plot with high resolution
    outpath = plot_folder / 'correlation_matrix.png'
    plt.savefig(outpath, dpi=300)
    plt.close()  # Close the plot to prevent display during batch processing


def plot_correlation_matrix_dendogram(mod1_subset, plot_folder):
    """Plot and save the correlation matrix heatmap with a dendrogram."""
    # Drop categorical or unnecessary columns for correlation
    mod1_final = mod1_subset.drop(columns=['Stagione', 'L1', 'AvalDay'])
    corr_matrix = mod1_final.corr()

    # Set up the matplotlib figure and size
    plt.figure(figsize=(10, 10))

    # Draw the clustermap with the correlation matrix
    sns.clustermap(corr_matrix, annot=True, cmap='coolwarm',
                   vmin=-1, vmax=1, fmt='.2f', linewidths=0.5,
                   figsize=(10, 10), method='average')

    # Add title
    plt.title('Correlation Matrix with Dendrogram', size=16)

    # Save the plot with high resolution
    outpath = plot_folder / 'correlation_matrix_dendrogram.png'
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_histograms(mod1_final, plot_folder):
    """Plot and save histograms for each column in mod1_final."""
    # Plotting histograms
    axes = mod1_final.hist(bins=20, figsize=(
        8, 8), edgecolor='black', color='skyblue')

    # Set title for the overall figure
    plt.suptitle('Histograms of Observations', size=16)

    # Adjust y-label for each plot and layout
    for ax in axes.flatten():
        ax.set_ylabel('n.obs')  # Set y-axis label to 'n.obs'

    # Adjust layout to ensure proper spacing for the title
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot with high resolution
    outpath = plot_folder / 'histogram_observation.png'
    plt.savefig(outpath, dpi=300)
    plt.close()  # Close the plot


def standardize_data(data):
    """Standardizes the data using StandardScaler."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.dropna())  # Drop NaNs if present
    return data_scaled


def perform_pca(data_scaled, n_components=15):
    """Performs PCA on standardized data."""
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_
    return pca, pca_components, explained_variance


def plot_pca_2d(pca_df, explained_variance, plot_folder):
    """Plots PCA results for the first two principal components."""
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c='blue', edgecolor='k', s=50)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.3f})')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.3f})')
    plt.title('PCA - 2 Principal Components')
    plt.grid(True)

    # Save plot
    outpath = plot_folder / 'pca_2d_plot.png'
    plt.savefig(outpath, dpi=300)
    plt.close()  # Close the plot to prevent display during batch processing


def plot_explained_variance(explained_variance, plot_folder):
    """Plots the explained variance ratio for each principal component."""
    plt.figure(figsize=(8, 8))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance,
            alpha=0.7, align='center', color='blue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(np.arange(1, len(explained_variance) + 1))
    plt.grid(True)

    # Save plot
    outpath = plot_folder / 'screeplot_PCA.png'
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_loadings(pca, data, explained_variance, plot_folder):
    """Plots the PCA loadings for PC1 vs PC2 with arrows."""
    loadings = pca.components_.T  # Transpose to align with features
    loadings_df = pd.DataFrame(loadings, index=data.columns,
                               columns=[f'PC{i+1}' for i in range(loadings.shape[1])])

    # Create a 2D scatter plot for PC1 vs PC2 with variable contributions
    plt.figure(figsize=(8, 8))
    plt.scatter(loadings_df['PC1'], loadings_df['PC2'],
                alpha=0.1, c='blue', s=50)

    # Add arrows and labels for each variable
    for i, variable in enumerate(loadings_df.index):
        plt.text(loadings_df['PC1'][i]*10, loadings_df['PC2'][i]
                 * 10, variable, color='black', ha='center', va='center')
        plt.arrow(0, 0, loadings_df['PC1'][i]*10, loadings_df['PC2'][i]*10,
                  head_width=0.5, head_length=0.5, fc='red', ec='red')

    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.3f})')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.3f})')
    plt.title('PCA: PC1 vs PC2 with Variable Contributions')
    plt.grid(True)

    # Save plot
    outpath = plot_folder / 'piplot_PCA.png'
    plt.savefig(outpath, dpi=300)
    plt.close()


def calculate_loadings(pca, data):
    """Calculates and prints PCA loadings for all components."""
    loadings_dict = {}

    # Loop through each component and store the loadings
    for i in range(pca.n_components_):
        loadings = pca.components_[i]
        loadings_df = pd.DataFrame(
            loadings, index=data.columns, columns=[f'PC{i+1}_Loading'])
        sorted_loadings_df = loadings_df.reindex(
            loadings_df[f'PC{i+1}_Loading'].abs().sort_values(ascending=False).index)
        loadings_dict[f'PC{i+1}'] = sorted_loadings_df

    # Concatenate all loadings into a single DataFrame
    all_loadings_df = pd.concat(loadings_dict, axis=1)
    print(all_loadings_df)  # Display loadings for all principal components
    return all_loadings_df


def perform_pca_3d(data_scaled):
    """Performs PCA and reduces to 3 principal components for 3D plotting."""
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_

    return pca, pca_components, explained_variance


def plot_pca_3d(pca_df, explained_variance, plot_folder):
    """Plots the PCA results in 3D space for the first three principal components."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the PCA components
    ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
               c='blue', edgecolor='k', s=50)

    # Set axis labels
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%} variance)')

    # Set plot title
    ax.set_title('PCA - First 3 Principal Components')

    # Save the plot
    outpath = plot_folder / 'pca_3d_plot.png'
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_3d_loadings(pca, pca_df, data, explained_variance, plot_folder):
    """Plots 3D PCA loadings with variable contributions as arrows."""
    # Transpose to align features with principal components
    loadings_df = pd.DataFrame(
        pca.components_.T, index=data.columns, columns=['PC1', 'PC2', 'PC3'])

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the principal components
    ax.scatter(pca_df['PC1'], pca_df['PC2'],
               pca_df['PC3'], alpha=0.1, c='blue', s=50)

    # Plot arrows and variable names
    for i, variable in enumerate(loadings_df.index):
        ax.text(loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i]*10, loadings_df['PC3'].iloc[i]*10,
                variable, color='black', ha='center', va='center')

        ax.quiver(0, 0, 0,
                  loadings_df['PC1'].iloc[i]*10, loadings_df['PC2'].iloc[i] *
                  10, loadings_df['PC3'].iloc[i]*10,
                  arrow_length_ratio=0.1, color='red')

    # Set axis labels with explained variance
    ax.set_xlabel(
        f'Principal Component 1 - ({explained_variance[0]:.3f})', labelpad=10)
    ax.set_ylabel(
        f'Principal Component 2 - ({explained_variance[1]:.3f})', labelpad=10)
    ax.set_zlabel(
        f'Principal Component 3 - ({explained_variance[2]:.3f})', labelpad=10)

    # Set plot title
    ax.set_title('PCA: PC1 vs PC2 vs PC3 with Variable Contributions')

    # Adjust plot layout for better label visibility
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)

    # Optional: Adjust viewing angle for better visibility (uncomment if needed)
    # ax.view_init(elev=30, azim=45)

    # Save plot
    outpath = plot_folder / 'piplot_3d_PCA.png'
    plt.savefig(outpath, dpi=300)
    plt.close()

# Function to calculate snow height differences


def calculate_snow_height_differences(df):
    """Calculate snow height differences over different periods."""
    df['HSdiff24h'] = df['HSnum'].diff(periods=1)
    df['HSdiff48h'] = df['HSnum'].diff(periods=2)
    df['HSdiff72h'] = df['HSnum'].diff(periods=3)
    df['HSdiff120h'] = df['HSnum'].diff(periods=5)
    return df

# Function to calculate new snow metrics


def calculate_new_snow(df):
    """Calculate cumulative new snow metrics over different periods."""
    df['HN48h'] = df['HNnum'].rolling(window=2).sum()
    df['HN72h'] = df['HNnum'].rolling(window=3).sum()
    df['HN120h'] = df['HNnum'].rolling(window=5).sum()
    return df

# Function to calculate temperature statistics


def calculate_temperature(df):
    """Calculate minimum, maximum temperatures and their differences over different periods."""
    df['Tmin48h'] = df['TminG'].rolling(window=2).min()
    df['Tmax48h'] = df['TmaxG'].rolling(window=2).max()
    df['Tmin72h'] = df['TminG'].rolling(window=3).min()
    df['Tmax72h'] = df['TmaxG'].rolling(window=3).max()
    df['Tmin120h'] = df['TminG'].rolling(window=5).min()
    df['Tmax120h'] = df['TmaxG'].rolling(window=5).max()

    # Temperature amplitude
    df['Tdelta24h'] = df['TmaxG'] - df['TminG']
    df['Tdelta48h'] = df['Tmax48h'] - df['Tmin48h']
    df['Tdelta72h'] = df['Tmax72h'] - df['Tmin72h']
    df['Tdelta120h'] = df['Tmax120h'] - df['Tmin120h']
    return df

# Function to calculate snow drift based on wind


def calculate_wind_snow_drift(df):
    """Calculate snow drift based on wind strength (VQ1)."""
    df['SnowDrift'] = df['VQ1'].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 0})
    df['SnowDrift48h'] = df['SnowDrift'].rolling(window=2).sum()
    df['SnowDrift72h'] = df['SnowDrift'].rolling(window=3).sum()
    df['SnowDrift120h'] = df['SnowDrift'].rolling(window=5).sum()
    return df

# Function to calculate snow water equivalent (SWE)


def calculate_swe(df):
    """Calculate snow water equivalent (SWE) and cumulative precipitation sums."""
    df['rho_adjusted'] = np.where(df['HNnum'] < 6, 100, df['rho'])
    df['SWEnew'] = df['HNnum'] * df['rho_adjusted'] / 100
    df['SWE_cumulative'] = df.groupby('Stagione')['SWEnew'].cumsum()

    # Precipitation sums over different periods
    df['PSUM24h'] = df['SWEnew']
    df['PSUM48h'] = df['SWEnew'].rolling(window=2).sum()
    df['PSUM72h'] = df['SWEnew'].rolling(window=3).sum()
    df['PSUM120h'] = df['SWEnew'].rolling(window=5).sum()
    return df

# Function to calculate wet snow presence


def calculate_wet_snow(df):
    """Calculate wet snow presence based on CS (critical snow surface)."""
    df['WetSnow'] = np.where(df['CS'] >= 20, 1, 0)
    df['WetSnow'] = np.where(df['CS'].isna(), np.nan, df['WetSnow'])
    return df

# Function to calculate temperature gradient


def calculate_temperature_gradient(df):
    """Calculate the temperature gradient based on the snow height."""
    df['T_gradient'] = abs(df['TH01G']) / (df['HSnum'] - 10)
    df['T_gradient'] = np.where(
        df['T_gradient'] == np.inf, np.nan, df['T_gradient'])
    return df

# Function to calculate avalanche day statistics


def calculate_avalanche_days(df):
    """Calculate avalanche occurrence and moving averages."""
    df['AvalDay'] = np.where(df['L1'] >= 1, 1, df['L1'])
    df['AvalDay48h'] = df['AvalDay'].rolling(window=2).mean()
    df['AvalDay72h'] = df['AvalDay'].rolling(window=3).mean()
    df['AvalDay120h'] = df['AvalDay'].rolling(window=5).mean()
    return df

# Function to plot the correlation matrix


def plot_correlation_matrix_new(df, plot_folder):
    """Plot and save the correlation matrix heatmap."""
    corr_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.1, cbar_kws={'shrink': 0.8})
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Correlation Matrix', size=16)

    outpath = plot_folder / 'correlation_newvar.png'
    plt.savefig(outpath, dpi=300)


def plot_correlation_matrix_new_dendogram(df, plot_folder):
    """Plot and save the correlation matrix heatmap with a dendrogram."""
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Set up the clustermap with the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.clustermap(corr_matrix, annot=False, cmap='coolwarm',
                   vmin=-1, vmax=1, linewidths=0.1, cbar_kws={'shrink': 0.8},
                   figsize=(10, 10), method='average')

    # Add title
    plt.title('Correlation Matrix with Dendrogram', size=16)

    # Save the plot with high resolution
    outpath = plot_folder / 'correlation_newvar_dendrogram.png'
    plt.savefig(outpath, dpi=300)
    plt.close()

# Function for PCA analysis and plotting


def perform_pca_new(df, plot_folder):
    """Perform PCA and plot the results."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.dropna())  # Standardize and drop NaN

    # PCA with 10 components
    pca = PCA(n_components=10)
    pca_components = pca.fit_transform(data_scaled)

    explained_variance = pca.explained_variance_ratio_

    # Scree plot of explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), explained_variance, alpha=0.7, color='blue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)

    outpath = plot_folder / 'screeplot_PCA_newvar.png'
    plt.savefig(outpath, dpi=300)

    # PC1 vs PC2 scatter plot with variable loadings
    loadings = pd.DataFrame(pca.components_.T, index=df.columns, columns=[
                            f'PC{i+1}' for i in range(10)])

    plt.figure(figsize=(10, 8))
    plt.scatter(pca_components[:, 0],
                pca_components[:, 1], alpha=0.1, color='blue')

    for i, variable in enumerate(loadings.index):
        plt.text(loadings['PC1'][i]*10, loadings['PC2'][i]*30,
                 variable, color='black', ha='center', va='center')
        plt.arrow(0, 0, loadings['PC1'][i]*10, loadings['PC2']
                  [i]*30, head_width=0.5, fc='red', ec='red')

    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f})')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f})')
    plt.title('PCA: PC1 vs PC2 with Variable Contributions')
    plt.grid(True)

    outpath = plot_folder / 'piplot_PCA_newvar.png'
    plt.savefig(outpath, dpi=300)


def plot_density(data_aval_0, data_aval_1, plot_folder):
    """Plot density distributions for two groups based on Avalanche Days."""
    # Set up the figure for a 5x3 grid layout
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through each column to plot its density distribution
    for i, col in enumerate(data_aval_0.columns):
        sns.kdeplot(data_aval_0[col], ax=axes[i], color='blue',
                    label='AvalDays = 0', fill=True, alpha=0.5)
        sns.kdeplot(data_aval_1[col], ax=axes[i], color='red',
                    label='AvalDays = 1', fill=True, alpha=0.5)

        # Remove the y-axis labels for the central and right columns
        if i % 3 != 0:  # Central and right columns
            axes[i].set_ylabel('')

    # Remove empty subplots if mod1_compare has fewer than 15 columns
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Create a single legend for all plots on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.9, 0.5), borderaxespad=0, ncol=1)

    # Add overall title
    plt.suptitle(
        'Density Plots for AvalDays = 0 and AvalDays = 1', fontsize=16)

    # Save the figure
    outpath_density = plot_folder / 'density_plots_avaldays.png'
    plt.savefig(outpath_density, dpi=300)
    plt.close(fig)  # Close the figure after saving


def plot_histogram(data_aval_0, data_aval_1, plot_folder):
    """Plot histograms for two groups based on Avalanche Days."""
    # Set up the figure for a 5x3 grid layout
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.85)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through each column to plot its histogram
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

    # Create a single legend for all plots on the right side
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(0.9, 0.5), borderaxespad=0, ncol=1)

    # Add overall title
    plt.suptitle(
        'Histogram Plots for AvalDays = 0 and AvalDays = 1', fontsize=16)

    # Save the figure
    outpath_histogram = plot_folder / 'histogram_plots_avaldays.png'
    plt.savefig(outpath_histogram, dpi=300)
    plt.close(fig)  # Close the figure after saving


def main():
    # --- PATHS ---

    # Filepath and plot folder paths
    filepath = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\mod1_tarlenta_01dec_15apr.csv')

    plot_folder = Path(
        'C:/Users/Christian/OneDrive/Desktop/Family/Christian/MasterMeteoUnitn/Corsi/4_Tesi/05_Plots/03_Correlation_mod1')

    # --- DATA MANIPULATION ---

    # Load and clean data
    mod1 = load_and_clean_data(filepath)
    print(mod1.dtypes)  # For initial data type inspection

    # Create the subset of interest
    mod1_subset = create_subset(mod1)
    print(mod1_subset.describe())  # Display descriptive statistics

    # --- CORRELATION MATRIX ---
    # Plot and save correlation matrix
    plot_correlation_matrix(mod1_subset, plot_folder)
    plot_correlation_matrix_dendogram(mod1_subset, plot_folder)

    # Drop columns that aren't suitable for histograms (like seasonality, etc.)
    mod1_final = mod1_subset.drop(columns=['Stagione', 'L1', 'AvalDay'])

    # Plot and save histograms
    plot_histograms(mod1_final, plot_folder)

    # --- PRINCIPAL COMPONENT ANALYSIS ---

    data = mod1[['N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG',
                 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B']]

    # Step 1: Standardize the data
    data_scaled = standardize_data(data)

    # Step 2: Perform PCA (15 components)
    pca, pca_components, explained_variance = perform_pca(
        data_scaled, n_components=15)

    # Step 3: Create a DataFrame for the principal components
    pca_df = pd.DataFrame(pca_components, columns=[
                          f'PC{i+1}' for i in range(pca.n_components_)])

    # Step 4: Plot PCA results
    plot_pca_2d(pca_df, explained_variance, plot_folder)
    plot_explained_variance(explained_variance, plot_folder)

    # Step 5: Plot loadings and print loadings
    plot_loadings(pca, data, explained_variance, plot_folder)
    loadings_df = calculate_loadings(pca, data)

    # Step 5: Perform PCA (3 components)
    pca, pca_components, explained_variance = perform_pca_3d(data_scaled)

    # Step 6: Create a DataFrame for the first 3 principal components
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2', 'PC3'])

    # Step 7: Plot PCA results in 3D
    plot_pca_3d(pca_df, explained_variance, plot_folder)

    # Step 8: Plot variable contributions (loadings) in 3D
    plot_3d_loadings(pca, pca_df, data, explained_variance, plot_folder)

    # --- NEW FEATURES CREATION ---

    # Add new variables to the dataset
    mod1_features = mod1[['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG',
                          'TmaxG', 'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B', 'L1']]
    mod1_features = calculate_snow_height_differences(mod1_features)
    mod1_features = calculate_new_snow(mod1_features)
    mod1_features = calculate_temperature(mod1_features)
    mod1_features = calculate_wind_snow_drift(mod1_features)
    mod1_features = calculate_swe(mod1_features)
    mod1_features = calculate_wet_snow(mod1_features)
    mod1_features = calculate_temperature_gradient(mod1_features)
    mod1_features = calculate_avalanche_days(mod1_features)

    # --- CORRELATION MATRIX NEW FEATURES ---

    # Plot the correlation matrix
    plot_correlation_matrix_new(mod1_features.drop(
        columns=['Stagione']), plot_folder)
    plot_correlation_matrix_new_dendogram(mod1_features.drop(
        columns=['Stagione']), plot_folder)

    # --- PRINCIPAL COMPONENT ANALYSIS NEW FEATURES ---

    # Perform PCA analysis and plot
    perform_pca_new(mod1_features.drop(
        columns=['Stagione']), plot_folder)

    # --- DISTRIBUTION OF ORIGINAL VARIABLES IN AvDays and NotAvDays  ---

    # Assuming mod1_subset and plot_folder are already defined.
    mod1_compare = mod1_subset.drop(columns=['Stagione', 'L1'])

    # Separate data based on AvalDays
    data_aval_0 = mod1_compare[mod1_compare['AvalDay'] == 0].drop(columns=[
                                                                  'AvalDay'])
    data_aval_1 = mod1_compare[mod1_compare['AvalDay'] == 1].drop(columns=[
                                                                  'AvalDay'])

    # Generate plots
    plot_density(data_aval_0, data_aval_1, plot_folder)
    plot_histogram(data_aval_0, data_aval_1, plot_folder)


if __name__ == '__main__':
    main()
