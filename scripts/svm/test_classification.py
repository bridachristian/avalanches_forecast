# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:43:38 2024

@author: Christian
"""

from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import undersampling_random, undersampling_random_timelimited, undersampling_nearmiss
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

# Generate a toy dataset
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
# Filepath and plot folder paths
common_path = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

filepath = common_path / 'mod1_newfeatures.csv'
results_path = Path(
    'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

# output_filepath = common_path / 'mod1_undersampling.csv'
# output_filepath2 = common_path / 'mod1_oversampling.csv'

# --- DATA IMPORT ---

# Load and clean data
mod1 = load_data(filepath)
print(mod1.dtypes)  # For initial data type inspection

# --- FEATURES SELECTION ---
feature = ['HN_3d', 'HSnum']
feature_plus = feature + ['AvalDay']
mod1_clean = mod1[feature_plus]
mod1_clean = mod1_clean.dropna()

# X = mod1_clean.drop(columns=['Stagione', 'AvalDay'])
X = mod1_clean[feature]
y = mod1_clean['AvalDay']

columns = X.columns

# # Initialize the scaler
# scaler = StandardScaler()

# # Fit the scaler and transform the features
# X_scaled = scaler.fit_transform(X)

# # Convert the scaled array back to a DataFrame with the original column names
# X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# # Now, X_scaled_df is the scaled version of the original DataFrame, preserving the structure
# print(X_scaled_df.head())  # Print the first few rows to verify


X_nm, y_nm = undersampling_nearmiss(X, y, version=1, n_neighbors=483)


X_array = X_nm.values.astype(float)
y_array = y_nm.values.astype(float)


param_distributions = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 10000],
    'kernel': ['rbf']
}

random_search = RandomizedSearchCV(
    estimator=SVC(), param_distributions=param_distributions, n_iter=20, cv=5)
random_search.fit(X_array, y_array)
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)


best_C = 1
best_gamma = 0.001
print(f'Run: C = {best_C}, gamma = {best_gamma}')
model = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
model.fit(X_array, y_array)

xx, yy = np.meshgrid(np.linspace(X_array[:, 0].min(), X_array[:, 0].max(), 100),
                     np.linspace(X_array[:, 1].min(), X_array[:, 1].max(), 100))

# Use decision_function after fitting the model
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot in 2D
plt.figure()
plt.title(f"SVM Decision Boundary with C={best_C} and gamma={best_gamma}")
plt.scatter(X_array[:, 0], X_array[:, 1],
            c=y_array, cmap='coolwarm', alpha=0.5)

# Add x and y axis labels
plt.xlabel(columns[0])
plt.ylabel(columns[1])

ax = plt.gca()
ax.contour(xx, yy, Z, levels=[-1, 0, 1],
           linestyles=['--', '-', '--'], colors='k')

plt.show()

# HEATMAP
# Define custom colormap
cmap = LinearSegmentedColormap.from_list(
    'custom_colormap', ['violet', 'white', 'green'], N=256)

vmax = max(abs(Z.min()), abs(Z.max()))

# Create a heatmap of Z values
plt.figure(figsize=(8, 6))
plt.title(
    f"SVM Decision Boundary Heatmap with C={best_C} and gamma={best_gamma}")

# Plot heatmap using contourf, centering the color scale around 0
heatmap = plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm',
                       alpha=0.8)

plt.scatter(X_array[:, 0], X_array[:, 1],
            c=y_array, cmap='coolwarm', alpha=0.5)

# Add contour lines for decision boundary at levels [-1, 0, 1]
ax = plt.gca()  # Get current axes
ax.contour(xx, yy, Z, levels=[-1, 0, 1],
           linestyles=['--', '-', '--'], colors='k')

# Add color bar for the heatmap
cbar = plt.colorbar(heatmap)
cbar.set_label('Decision Function Value (Z)')

# Add axis labels
plt.xlabel(columns[0])
plt.ylabel(columns[1])

# Show the heatmap
plt.show()


# Plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(xx, yy, Z, cmap=cmap, alpha=0.8)
ax.view_init(elev=90, azim=0)  # Adjust elevation and azimuth


# # Plot the scatter points for the data
# scatter = ax.scatter(X_array[:, 0], X_array[:, 1], Z.ravel(),
#                       c=y_array, cmap='coolwarm', edgecolor='k', s=50)

ax.view_init(elev=30, azim=60)  # Adjust elevation and azimuth

# Add labels and a title
ax.set_xlabel(columns[0])
ax.set_ylabel(columns[1])
ax.set_zlabel("Decision Function")
ax.set_title("SVM Decision Boundary in 3D")

# Add color bar for the surface
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
