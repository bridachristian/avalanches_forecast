from pathlib import Path
from scripts.svm.data_loading import load_data
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   StandardScaler)
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to numeric features safely.
    - Removes binary features
    - Applies appropriate transformations based on feature type
    - Scales all numeric features at the end, except target
    Returns:
        Transformed DataFrame
    """

    # --- Feature Groups ---
    binary_features = ['WetSnow_CS', 'WetSnow_Temperature',
                       'MF_Crust_Present', 'New_MF_Crust']

    log_features = ['HNnum', 'PR', 'HN_2d', 'HN_3d', 'HN_5d']
    sqrt_features = ['DaysSinceLastSnow', 'Penetration_ratio', 'TempGrad_HS',
                     'ConsecWetSnowDays', 'ConsecCrustDays']
    log1p_features = ['DegreeDays_Pos', 'DegreeDays_cumsum_2d',
                      'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
                      'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d']
    gaussian_features = ['HSnum', 'DayOfSeason', 'TempAmplitude_1d',
                         'TempAmplitude_2d', 'TempAmplitude_3d', 'TempAmplitude_5d']
    negative_features = ['TH01G', 'TH03G']

    target = ['AvalDay']

    # --- Clean Up: Remove Binary Features ---
    df = df.drop(
        columns=[col for col in binary_features if col in df.columns], errors='ignore')

    # --- Utility function for log1p(log(x+1)) ---
    def double_log(x): return np.log1p(np.log(x + 1))

    # --- Safe transformations only on existing columns ---
    def safe_transform(cols, func):
        existing = [col for col in cols if col in df.columns]
        if existing:
            df[existing] = df[existing].apply(func)

    # Apply transformations
    safe_transform(log_features, lambda x: x.apply(double_log))
    safe_transform(sqrt_features, np.sqrt)
    safe_transform(log1p_features, np.log1p)

    # Yeo-Johnson for positive + transformed features
    yeo_features = [col for col in (
        log_features + sqrt_features + log1p_features) if col in df.columns]
    if yeo_features:
        pt = PowerTransformer(method='yeo-johnson')
        df[yeo_features] = pt.fit_transform(df[yeo_features])

    # Yeo-Johnson for negative features
    neg_feats = [col for col in negative_features if col in df.columns]
    if neg_feats:
        pt_neg = PowerTransformer(method='yeo-johnson')
        df[neg_feats] = pt_neg.fit_transform(df[neg_feats])

    # QuantileTransformer for Gaussian-like features
    gauss_feats = [col for col in gaussian_features if col in df.columns]
    if gauss_feats:
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        df[gauss_feats] = qt.fit_transform(df[gauss_feats])

    # Standard scaling for all numeric features (excluding the target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in target]

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


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
