from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import ClusterCentroids
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path
from scripts.svm.data_loading import load_data
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   StandardScaler, MinMaxScaler)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to numeric features safely.
    - Removes binary features
    - Applies appropriate transformations based on feature type
    - Scales all numeric features at the end, except target and explicitly ignored columns
    - Applies MinMaxScaler only to freshsnow_features
    Returns:
        Transformed DataFrame
    """
    # --- Feature Groups ---
    to_exclude = ['WetSnow_CS', 'WetSnow_Temperature',
                  'MF_Crust_Present', 'New_MF_Crust',
                  'ConsecCrustDays', 'ConsecWetSnowDays',
                  'DegreeDays_Pos', 'DegreeDays_cumsum_2d',
                  'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d']

    sqrt_features = ['DaysSinceLastSnow', 'Penetration_ratio',
                     'TempGrad_HS']
    gaussian_features = ['HSnum', 'DayOfSeason', 'TempAmplitude_1d',
                         'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d']
    negative_features = ['TH01G', 'TH03G']

    precip_features = ['HNnum', 'HN_2d', 'HN_3d', 'HN_5d',
                       'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d']

    # <- Ignored from all transformations
    to_ignore = ['PR', 'Penetration_ratio']
    target = ['AvalDay']

    # --- Clean Up: Remove Binary Features ---
    df = df.drop(
        columns=[col for col in to_exclude if col in df.columns], errors='ignore')

    # --- Transformation helpers ---
    def safe_transform(cols, func):
        """Apply a transformation only if the column exists and not in to_ignore or freshsnow_features."""
        for col in cols:
            if col in df.columns and col not in to_ignore and col not in precip_features:
                df[col] = df[col].apply(func)

    # Apply transformations
    safe_transform(sqrt_features, np.sqrt)

    # Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    yeo_features = list(set(sqrt_features) &
                        set(df.columns) - set(to_ignore) - set(precip_features))
    if yeo_features:
        df[yeo_features] = pt.fit_transform(df[yeo_features])

    # Yeo-Johnson for negative features
    pt_neg = PowerTransformer(method='yeo-johnson', standardize=False)
    neg_feats = list(set(negative_features) & set(df.columns))
    if neg_feats:
        df[neg_feats] = pt_neg.fit_transform(df[neg_feats])

    # QuantileTransformer for Gaussian-like features
    n_samples = df.shape[0]
    n_quantiles = min(1000, n_samples)
    qt = QuantileTransformer(output_distribution='normal',
                             n_quantiles=n_quantiles, random_state=42)
    gauss_feats = list(set(gaussian_features) & set(
        df.columns) - set(to_ignore) - set(precip_features))
    if gauss_feats:
        df[gauss_feats] = qt.fit_transform(df[gauss_feats])

    # --- Create binary features for fresh snow presence ---
    prec_feats = [col for col in precip_features if col in df.columns]
    for col in prec_feats:
        bin_col = f"{col}_bin"
        df[bin_col] = (df[col] > 0).astype(int)

    # --- MinMax scaling for fresh snow features ---
    if prec_feats:
        prec_scaler = MinMaxScaler()
        df[prec_feats] = prec_scaler.fit_transform(df[prec_feats])

    # # --- MinMax scaling for fresh snow features ---
    # prec_feats = [col for col in precip_features if col in df.columns]
    # if prec_feats:
    #     fs_scaler = MinMaxScaler()
    #     df[prec_feats] = prec_scaler.fit_transform(df[prec_feats])

    # --- Standard scaling for the rest (excluding target, to_ignore, freshsnow, and *_bin columns) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [
        col for col in numeric_cols
        if col not in target + to_ignore + precip_features and not col.endswith('_bin')
    ]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # --- Log ---
    logging.info(
        f"Transformations applied. Ignored: {to_ignore}. MinMax only: {precip_features}")
    return df


def transform_penetration_ratio(pr_series: pd.Series) -> pd.DataFrame:
    """
    Trasforma la variabile PR (Penetration Ratio) in modo robusto:
    - Clipping degli outlier oltre il 99° percentile
    - Applicazione di log1p (log(x + 1)), adatto per skew positivo
    - Standardizzazione finale per PR_std
    - Normalizzazione (MinMax) per PR_norm
    Ritorna un DataFrame con due colonne: PR_std e PR_norm
    """
    # Clipping per rimuovere outlier estremi (es. > 99° percentile)
    upper_clip = pr_series.quantile(0.99)
    pr_clipped = pr_series.clip(upper=upper_clip)

    # log1p per skew positivo
    pr_log = np.log1p(pr_clipped)

    # Standardizzazione (z-score)
    std_scaler = StandardScaler()
    pr_std = std_scaler.fit_transform(pr_log.values.reshape(-1, 1)).flatten()

    # Normalizzazione (min-max)
    minmax_scaler = MinMaxScaler()
    pr_norm = minmax_scaler.fit_transform(
        pr_series.values.reshape(-1, 1)).flatten()

    # Use the original column name to create the new column names dynamically
    feature_name = pr_series.name  # Get the name of the input series

    # Return a DataFrame with dynamically named columns
    return pd.DataFrame({
        f'{feature_name}_std': pr_std,
        f'{feature_name}_norm': pr_norm
    }, index=pr_series.index)


def plot_feature_distributions(df, features=None, bins=15):
    """
    Plots histograms for each feature in the DataFrame.

    Parameters:
    - df: DataFrame with features
    - features: List of feature names to plot. If None, all numeric features are plotted.
    - bins: Number of bins in the histogram
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 3))

    for idx, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.hist(df[feature], bins=bins, color='skyblue', edgecolor='black')
        plt.title(feature)
        plt.tight_layout()

    plt.show()


def plot_feature_kde(df, features=None):
    """
    Plots KDE (density) plots for each numeric feature using Seaborn.

    Parameters:
    - df: DataFrame with features
    - features: Optional list of features to include
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 3))

    for idx, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, idx + 1)
        sns.kdeplot(df[feature], fill=True)
        plt.title(f"Distribution: {feature}")
        plt.tight_layout()

    plt.show()


def plot_feature_comparison_side_by_side(raw_df, transformed_df, feature, bins=15):
    """
    Plots KDE, histogram, and data points (rugplot) for a feature before and after transformation.

    Parameters:
    - raw_df: DataFrame with raw/original features
    - transformed_df: DataFrame with transformed features
    - feature: Name of the feature to plot
    - bins: Number of bins for the histogram
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # Original (left)
    sns.histplot(data=raw_df, x=feature, bins=bins, stat='probability', kde=True,
                 color='skyblue', ax=axes[0], edgecolor='white')
    sns.rugplot(data=raw_df, x=feature, ax=axes[0], color='black', height=0.05)
    axes[0].set_title(f'Original: {feature}')
    axes[0].set_xlabel('Value')

    # Transformed (right)
    sns.histplot(data=transformed_df, x=feature, bins=bins, stat='probability', kde=True,
                 color='salmon', ax=axes[1], edgecolor='white')
    sns.rugplot(data=transformed_df, x=feature,
                ax=axes[1], color='black', height=0.05)
    axes[1].set_title(f'Transformed: {feature}')
    axes[1].set_xlabel('Value')

    plt.tight_layout()
    plt.show()


# if __name__ == '__main__':
#     # --- PATHS ---

#     # Filepath and plot folder paths
#     common_path = Path(
#         'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\')

#     filepath = common_path / 'mod1_newfeatures_NEW.csv'
#     # filepath = common_path / 'mod1_certified.csv'
#     results_path = Path(
#         'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\SVM_results\\')

#     # --- DATA IMPORT ---

#     # Load and clean data
#     mod1 = load_data(filepath)
#     print(mod1.dtypes)  # For initial data type inspection

#     # DEFINE INITIAL FEATURE SET
#     feature_set = [
#         'TaG', 'TminG', 'TmaxG', 'HSnum',
#                'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
#                'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
#                'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
#                'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
#                'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
#                'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
#                'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
#                'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean',
#                # 'DegreeDays_Pos',
#                # 'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
#                'Precip_1d', 'Precip_2d', 'Precip_3d',
#                'Precip_5d', 'Penetration_ratio',
#                'ConsecWetSnowDays', 'ConsecCrustDays',
#                'TempGrad_HS', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
#                'Tsnow_delta_5d']

#     # ---------------------------------------------------------------
#     # --- a) FEATURE SELECTION BASED ON PERMUTATION RANKING       ---
#     # ---------------------------------------------------------------

#     feature_plus = feature_set + ['AvalDay']
#     mod1_clean = mod1[feature_plus].dropna()

#     mod1_transformed = transform_features(mod1_clean.copy())

#     # Trasforma PR e aggiunge entrambe le versioni
#     PR_transformed = transform_penetration_ratio(mod1_clean['PR'])
#     Penetration_ratio_transformed = transform_penetration_ratio(
#         mod1_clean['Penetration_ratio'])

#     mod1_transformed = pd.concat([mod1_transformed, PR_transformed], axis=1)
#     mod1_transformed = pd.concat(
#         [mod1_transformed, Penetration_ratio_transformed], axis=1)

#     # Salva PR_norm nel dataset originale (solo per reinserirlo poi)
#     mod1_clean['PR_norm'] = PR_transformed['PR_norm']
#     mod1_clean['Penetration_ratio_norm'] = Penetration_ratio_transformed['Penetration_ratio_norm']

#     # ----------------------------
#     # SPLIT X, y
#     # ----------------------------
#     X = mod1_transformed.drop(columns=['AvalDay'])

#     # Mantieni solo PR_std per il clustering, rimuovi PR_norm
#     X = X.drop(columns=['PR_norm', 'Penetration_ratio_norm'])

#     y = mod1_transformed['AvalDay']

#     # First, check if the columns exist in X before attempting to drop them
#     print(X.columns)  # This will show the columns in X

#     # Conditionally drop columns only if they exist in the DataFrame
#     columns_to_drop = ['PR_norm', 'Penetration_ratio_norm']
#     X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

#     y = mod1_transformed['AvalDay']

#     # ----------------------------
#     # VARIABILI BINARIE E PRECIP
#     # ----------------------------
#     precip_features = ['HNnum', 'HN_2d', 'HN_3d', 'HN_5d',
#                        'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d']

#     bin_features = ['HNnum_bin', 'HN_2d_bin', 'HN_3d_bin', 'HN_5d_bin',
#                     'Precip_1d_bin', 'Precip_2d_bin', 'Precip_3d_bin', 'Precip_5d_bin']

#     # ----------------------------
#     # SEPARAZIONE VARIABILI
#     # ----------------------------
#     X_continuous = X.drop(columns=bin_features + precip_features)
#     X_binary = X[bin_features]
#     X_freshsnow = X[precip_features]

#     # ----------------------------
#     # APPLY CLUSTER CENTROIDS
#     # ----------------------------
#     cc = ClusterCentroids(random_state=42)
#     X_resampled_cont, y_resampled = cc.fit_resample(X_continuous, y)

#     # ----------------------------
#     # RECUPERA INDICI PIÙ VICINI AI CENTROIDI
#     # ----------------------------
#     closest_idxs, _ = pairwise_distances_argmin_min(
#         X_resampled_cont, X_continuous)

#     # Recupera le binarie e le fresh snow originali
#     X_resampled_bin = X_binary.iloc[closest_idxs].reset_index(drop=True)
#     X_resampled_fresh = X_freshsnow.iloc[closest_idxs].reset_index(drop=True)

#     # Recupera anche la PR originale (normale) per interpretabilità
#     PR_original_resampled = mod1_clean['PR_norm'].iloc[closest_idxs].reset_index(
#         drop=True)
#     Penetration_ratio_original_resampled = mod1_clean['Penetration_ratio_norm'].iloc[closest_idxs].reset_index(
#         drop=True)

#     # ----------------------------
#     # REINTEGRA TUTTO NEL DATASET FINALE
#     # ----------------------------
#     X_resampled = pd.concat([X_resampled_cont.reset_index(drop=True),
#                              X_resampled_fresh,
#                              X_resampled_bin], axis=1)

#     # Rimuovi la PR_std usata per il clustering
#     X_resampled = X_resampled.drop(
#         columns=['PR_std', 'Penetration_ratio_std'], errors='ignore')

#     # Inserisci PR_norm (versione interpretabile)
#     X_resampled['PR'] = PR_original_resampled
#     X_resampled['Penetration_ratio'] = Penetration_ratio_original_resampled

#     # # dove df_transformed è l'output di transform_features
#     # plot_feature_distributions(mod1_clean)
#     # # dove df_transformed è l'output di transform_features
#     # plot_feature_distributions(mod1_transformed)

#     for feat in feature_set:
#         if feat in mod1_clean.columns and feat in X_resampled.columns:
#             plot_feature_comparison_side_by_side(
#                 mod1_clean, X_resampled, feat)


#     # ---------------------------------------------------------------
#     # --- a) FEATURE SELECTION BASED ON PERMUTATION RANKING       ---
#     # ---------------------------------------------------------------

#     # feature = [
#     #     'TaG', 'TminG', 'TmaxG', 'HSnum',
#     #     'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason',
#     #     'HS_delta_1d', 'HS_delta_2d', 'HS_delta_3d', 'HS_delta_5d',
#     #     'HN_2d', 'HN_3d', 'HN_5d',
#     #     'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
#     #     'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
#     #     'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
#     #     'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
#     #     'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
#     #     'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
#     #     'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
#     #     'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d',
#     #     'Penetration_ratio',
#     #     'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d',
#     #     'Tsnow_delta_2d', 'Tsnow_delta_3d',
#     #     'Tsnow_delta_5d', 'ConsecWetSnowDays',
#     #     'AvalDay_2d', 'AvalDay_3d', 'AvalDay_5d'
#     # ]

#     feature = [
#         'TaG', 'TminG', 'TmaxG', 'HSnum',
#         'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d',
#         'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
#         'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
#         'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
#         'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
#         'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
#         'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
#         'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
#         'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
#         'Precip_1d', 'Precip_2d', 'Precip_3d',
#         'Precip_5d', 'Penetration_ratio',
#         'WetSnow_CS', 'WetSnow_Temperature',
#         'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
#         'Tsnow_delta_5d', 'ConsecWetSnowDays',
#         'MF_Crust_Present', 'New_MF_Crust', 'ConsecCrustDays'
#     ]

#     feature_plus = feature + ['AvalDay']
#     mod1_clean = mod1[feature_plus]
#     mod1_clean = mod1_clean.dropna()

#     mod1_transformed = transform_features(mod1_clean.copy())

# # Simuliamo un dataset di esempio
# np.random.seed(42)
# n = 1000
# HN_2d = np.concatenate([
#     np.zeros(800),                  # Giorni senza neve
#     np.random.gamma(2, 4, 200)      # Giorni con neve (positivi, skewed)
# ])
# target = np.concatenate([
#     np.zeros(950),  # Classe 0: no valanga
#     np.ones(50)     # Classe 1: valanga
# ])

# df = pd.DataFrame({'HN_2d': HN_2d, 'y': target})

# # Separiamo il feature vector e il target
# X = df[['HN_2d']].copy()
# y = df['y']

# # --- 1. Trasformazione dei valori > 0 ---
# qt = QuantileTransformer(output_distribution='normal', random_state=42)
# X_transformed = X.copy()
# is_positive = X['HN_2d'] > 0

# X_transformed['HN_2d'] = 0  # inizializza a 0
# X_transformed.loc[is_positive, 'HN_2d'] = qt.fit_transform(
#     X.loc[is_positive, ['HN_2d']])

# # Ora HN_2d è più vicino a una gaussiana, con 0 per i giorni senza neve

# # --- 2. Applica ClusterCentroids ---
# cc = ClusterCentroids(random_state=42)
# X_resampled, y_resampled = cc.fit_resample(X_transformed, y)

# # --- 3. Risultati ---
# print("Forma originale:", X.shape, y.value_counts().to_dict())
# print("Forma bilanciata:", X_resampled.shape,
#       dict(pd.Series(y_resampled).value_counts()))

# plot_feature_comparison_side_by_side(
#     X, X_resampled, 'HN_2d')
