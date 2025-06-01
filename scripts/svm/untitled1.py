# -*- coding: utf-8 -*-
"""
Created on Thu May 29 09:30:40 2025

@author: Christian
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np

# Dati
methods = [
    "Random Undersampling", "Random Undersampling 10d", "NearMiss v3 (k=1)", "NearMiss v3 (k=3)",
    "NearMiss v3 (k=5)", "NearMiss v3 (k=10)", "NearMiss v3 (k=25)", "NearMiss v3 (k=50)",
    "CNN Undersampling", "ENN Undersampling", "Cluster Centroids Und.", "TomekLinks Und.",
    "Random Oversampling", "SMOTE Oversampling", "ADASYN Oversampling", "SVMSMOTE Oversampling"
]

cases = ["Case 1", "Case 2", "Case 3", "Case 4"]

# F1-scores
f1_scores = [
    [0.649, 0.734, 0.621, 0.744],
    [0.593, 0.687, 0.585, 0.616],
    [0.621, 0.646, 0.634, 0.670],
    [0.674, 0.634, 0.739, 0.610],
    [0.663, 0.621, 0.757, 0.692],
    [0.684, 0.672, 0.757, 0.616],
    [0.684, 0.711, 0.740, 0.638],
    [0.663, 0.683, 0.760, 0.681],
    [0.610, 0.685, 0.523, 0.567],
    [0.690, 0.740, 0.693, 0.663],
    [0.674, 0.687, 0.608, 0.666],
    [0.595, 0.672, 0.588, 0.642],
    [0.626, 0.442, 0.543, 0.477],
    [0.574, 0.491, 0.570, 0.577],
    [0.575, 0.499, 0.516, 0.533],
    [0.620, 0.491, 0.580, 0.579]
]

# MCC
mcc_scores = [
    [0.318, 0.469, 0.289, 0.487],
    [0.195, 0.375, 0.183, 0.234],
    [0.256, 0.292, 0.268, 0.347],
    [0.369, 0.270, 0.492, 0.228],
    [0.360, 0.242, 0.528, 0.396],
    [0.427, 0.344, 0.528, 0.232],
    [0.402, 0.424, 0.486, 0.277],
    [0.360, 0.366, 0.574, 0.371],
    [0.298, 0.370, 0.083, 0.145],
    [0.450, 0.489, 0.484, 0.351],
    [0.348, 0.378, 0.313, 0.350],
    [0.262, 0.367, 0.279, 0.306],
    [0.260, 0.000, 0.114, 0.000],
    [0.173, 0.110, 0.198, 0.193],
    [0.191, 0.096, 0.108, 0.129],
    [0.242, 0.110, 0.163, 0.204]
]

# DataFrame in formato "long" per Seaborn
df_f1 = pd.DataFrame(f1_scores, index=methods, columns=cases).reset_index().melt(
    id_vars="index", var_name="Case", value_name="F1-score")
df_f1.rename(columns={"index": "Method"}, inplace=True)

df_mcc = pd.DataFrame(mcc_scores, index=methods, columns=cases).reset_index(
).melt(id_vars="index", var_name="Case", value_name="MCC")
df_mcc.rename(columns={"index": "Method"}, inplace=True)

# plt.figure(figsize=(14, 6))
# sns.barplot(data=df_f1, x="Method", y="F1-score", hue="Case")
# plt.xticks(rotation=90)
# plt.title("F1-score per metodo e caso")
# plt.tight_layout()
# plt.legend(title="Case")
# plt.grid(True, axis='y', linestyle='--', alpha=0.5)
# plt.show()

# plt.figure(figsize=(14, 6))
# sns.barplot(data=df_mcc, x="Method", y="MCC", hue="Case")
# plt.xticks(rotation=90)
# plt.title("MCC per metodo e caso")
# plt.tight_layout()
# plt.legend(title="Case")
# plt.grid(True, axis='y', linestyle='--', alpha=0.5)
# plt.show()


# f1_df = pd.DataFrame(f1_scores, index=methods, columns=cases)

# plt.figure(figsize=(12, 8))
# sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="YlGnBu",
#             cbar_kws={'label': 'F1-score'})
# plt.title("F1-score ")
# plt.ylabel("Methods")
# plt.tight_layout()
# plt.show()

# # Dati MCC
# mcc_df = pd.DataFrame(mcc_scores, index=methods, columns=cases)

# plt.figure(figsize=(12, 8))
# sns.heatmap(mcc_df, annot=True, fmt=".3f",
#             cmap="YlGnBu", cbar_kws={'label': 'MCC'})
# plt.title("MCC")
# plt.ylabel("Metodo")
# plt.tight_layout()
# plt.show()


# DataFrames già definiti
f1_df = pd.DataFrame(f1_scores, index=methods, columns=cases)
mcc_df = pd.DataFrame(mcc_scores, index=methods, columns=cases)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# Heatmap F1-score: palette "viridis" con annotazioni e colori
sns.heatmap(f1_df, annot=True, fmt=".3f", cmap="viridis",
            cbar_kws={'label': 'F1-score'}, ax=axes[0])

# Sposta le etichette colonne in alto e ruotale per leggibilità
axes[0].xaxis.set_ticks_position('top')
axes[0].xaxis.set_label_position('top')
# axes[0].set_xlabel('Cases', fontsize=12, fontweight='bold', labelpad=10)
axes[0].set_ylabel('Methods', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x')
axes[0].set_title("F1-score", fontsize=16, fontweight='bold', pad=20)

# Heatmap MCC: palette "RdYlGn"
sns.heatmap(mcc_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
            cbar_kws={'label': 'MCC'}, ax=axes[1])

# Etichette in alto e stile titolo per MCC
axes[1].xaxis.set_ticks_position('top')
axes[1].xaxis.set_label_position('top')
# axes[1].set_xlabel('Cases', fontsize=12, fontweight='bold', labelpad=10)
# axes[1].set_ylabel('Methods', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x')
axes[1].set_title("MCC", fontsize=16, fontweight='bold', pad=20)

# Rimuovi etichette y (metodi) sul grafico MCC
axes[1].set_yticklabels([])

plt.tight_layout()
plt.show()
