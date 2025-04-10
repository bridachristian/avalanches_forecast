# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:23:36 2025

@author: Christian
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn import svm
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
from imblearn.under_sampling import NearMiss

from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import (undersampling_random, undersampling_random_timelimited,
                                               undersampling_nearmiss, undersampling_cnn,
                                               undersampling_enn, undersampling_clustercentroids,
                                               undersampling_tomeklinks)
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                                    plot_roc_curve, permutation_ranking, evaluate_svm_with_feature_selection)
from scripts.svm.utils import (save_outputfile, get_adjacent_values, PermutationImportanceWrapper,
                               remove_correlated_features, remove_low_variance, select_k_best)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


if __name__ == '__main__':

    # .........................................................................
    #  TEST 1: HSnum vs HN_3d, full avalanche dataset
    # .........................................................................

    # Filepath and plot folder paths
    common_path = Path(
        'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

    # filepath = common_path / 'mod1_newfeatures_NEW.csv'
    # filepath = common_path / 'mod1_newfeatures.csv'
    filepath = common_path / 'mod1_certified.csv'
    # results_path = Path(
    #     'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

    # --- DATA IMPORT ---

    # Load and clean data
    mod1 = load_data(filepath)
    print(mod1.dtypes)  # For initial data type

    # --- EXPLORATORY DATA ANALYSIS ---

    # COUNTS
    mod1['AvalDay'].value_counts().plot(kind='bar')
    plt.title("Class distribution: Avalanche vs No Avalanche")
    plt.xticks([0, 1], ['No Avalanche', 'Avalanche'], rotation=0)
    plt.show()

    # UNIVARIATE ANALYSIS
    for col in mod1.columns[1:]:
        plt.figure(figsize=(6, 3))
        sns.kdeplot(data=mod1, x=col, hue='AvalDay',
                    fill=True, common_norm=False, alpha=0.5)
        plt.title(f'Distribution of {col} by Avalanche Occurrence')
        plt.show()

    # 3. Boxplots for Class Comparison

    for col in mod1.columns[1:]:
        sns.boxplot(x='AvalDay', y=col, data=mod1)
        plt.title(f'{col} vs Avalanche Occurrence')
        plt.show()

    # 4. Statistical Tests
    from scipy.stats import ttest_ind
    import pandas as pd

    # Create a list to collect results
    ttest_results = []

    # Loop through each column (skipping the first if it's an ID or timestamp)
    for col in mod1.columns[1:]:
        group0 = mod1[mod1['AvalDay'] == 0][col].dropna()
        group1 = mod1[mod1['AvalDay'] > 0][col].dropna()

        if len(group0) > 1 and len(group1) > 1:
            stat, p = ttest_ind(group0, group1, equal_var=False)
            # Interpret p-value
            if p < 0.01:
                interpretation = "Very strong difference"
            elif p < 0.05:
                interpretation = "Significant difference"
            elif p < 0.1:
                interpretation = "Possible difference"
            else:
                interpretation = "No significant difference"

            ttest_results.append({
                'Feature': col,
                'p-value': round(p, 4),
                'Interpretation': interpretation
            })
        else:
            ttest_results.append({
                'Feature': col,
                'p-value': None,
                'Interpretation': 'Not enough data'
            })

    # Convert list to DataFrame
    ttest_df = pd.DataFrame(ttest_results)

    # Optional: sort by p-value
    ttest_df_sorted = ttest_df.sort_values('p-value', na_position='last')

    # Show the results
    print(ttest_df_sorted)

    # 6. Correlation Matrix (with Target)

    import matplotlib.pyplot as plt

    # Create a copy of the DataFrame, drop 'Stagione' column, and compute correlations
    df_corr = mod1.copy()
    df_corr = df_corr.drop(columns='Stagione')

    # Calculate correlations with the target column 'AvalDay'
    correlations = df_corr.corr()['AvalDay'].drop(
        'AvalDay').sort_values(ascending=False)

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    correlations.plot(kind='bar', color='skyblue',
                      title='Correlation with Avalanche Occurrence')

    # Add gridlines for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    plt.tight_layout()  # Adjust layout to make sure everything fits well
    plt.show()

    # 7. PCA for Visualization

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Step 1: Preprocessing
    df_corr = mod1.copy()
    df_corr = df_corr.dropna()
    df_corr = df_corr.drop(columns='Stagione')  # Drop the non-numeric column
    X = df_corr.drop(columns='AvalDay')  # Features
    y = df_corr['AvalDay']  # Target (assuming this is categorical)

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Plot the PCA result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='AvalDay')
    plt.title('PCA - Avalanche Occurrence Class Separation')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # 8. t-SNE for Visualization

    from sklearn.manifold import TSNE

    # Step 1: Preprocessing
    df_corr = mod1.copy()
    df_corr = df_corr.dropna()
    df_corr = df_corr.drop(columns='Stagione')  # Drop the non-numeric column
    X = df_corr.drop(columns='AvalDay')  # Features
    y = df_corr['AvalDay']  # Target (assuming this is categorical)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Step 3: Plot the t-SNE result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                cmap='viridis', edgecolor='k', alpha=0.7)
    plt.colorbar(label='AvalDay')
    plt.title('t-SNE - Avalanche Occurrence Class Separation')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    # 9. Multivariate Regression

    from sklearn.linear_model import LinearRegression

    df_corr = mod1.copy()
    df_corr = df_corr.dropna()
    df_corr = df_corr.drop(columns='Stagione')  # Drop the non-numeric column
    X = df_corr.drop(columns='AvalDay')  # Features
    y = df_corr['AvalDay']  # Target (assuming this is categorical)

    # Train a linear regression model
    regressor = LinearRegression()
    regressor.fit(X_scaled, y)

    # Predict values
    y_pred = regressor.predict(X_scaled)

    # Evaluate the model
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')

    from sklearn.linear_model import LogisticRegression

    # Train a logistic regression model
    classifier = LogisticRegression()
    classifier.fit(X_scaled, y)

    # Predict classes
    y_pred = classifier.predict(X_scaled)

    # Evaluate the model
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}')

    # KMEANS clustering

    # Prepare your data
X = df_corr.drop(columns='AvalDay')  # or whatever your features are
X = X.dropna()
X_scaled = StandardScaler().fit_transform(X)  # Standardize features

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to your DataFrame (optional)
df_corr['Cluster'] = kmeans.labels_

# Plot clusters (only works if X has 2D/3D after PCA or t-SNE)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=kmeans.labels_, cmap='viridis', edgecolor='k')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
