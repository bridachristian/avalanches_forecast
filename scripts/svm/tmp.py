# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:52:52 2025

@author: Christian
"""
feature_set = [
    'TaG', 'TminG', 'TmaxG', 'HSnum',
           'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason',
           'HS_delta_1d', 'HS_delta_2d',
           'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
           'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
           'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
           'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
           'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
           'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
           'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean',
           'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d',
           'TempGrad_HS', 'Tsnow_delta_1d', 'Tsnow_delta_2d', 'Tsnow_delta_3d',
           'Tsnow_delta_5d']

PERMUTATION_RANKING = ['HSnum', 'HS_delta_5d', 'Tsnow_delta_3d', 'HS_delta_2d',
                       'PR', 'HS_delta_1d', 'TminG_delta_3d', 'TmaxG_delta_5d',
                       'DayOfSeason', 'TmaxG_delta_3d', 'TmaxG_delta_2d',
                       'TH01G', 'Precip_3d']  # 13 features

ANOVA = ['HSnum', 'TH01G', 'TH03G', 'HS_delta_1d', 'TminG_delta_5d',
         'TmaxG_delta_5d', 'Precip_2d', 'Precip_3d',
         'Precip_5d', 'TempGrad_HS']  # 10 features


SHAP = ['HSnum', 'TH01G', 'PR', 'DayOfSeason', 'TmaxG_delta_5d',
        'HS_delta_5d', 'TH03G', 'HS_delta_1d', 'TmaxG_delta_3d',
        'Precip_3d', 'TempGrad_HS', 'HS_delta_2d', 'TmaxG_delta_2d',
        'TminG_delta_5d', 'TminG_delta_3d', 'Tsnow_delta_3d',
        'TaG_delta_5d', 'Tsnow_delta_1d',
        'TmaxG_delta_1d', 'Precip_2d']  # 20 features


BFE = ['HSnum', 'TH03G', 'HS_delta_1d', 'HS_delta_5d', 'TempAmplitude_2d',
       'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
       'Precip_3d', 'Tsnow_delta_3d']  # 11 features

# FFS = ['HSnum', 'HS_delta_2d', 'HS_delta_3d', 'TempGrad_HS']
FFS = ['TaG', 'HSnum', 'HS_delta_2d', 'HS_delta_3d', 'DaysSinceLastSnow',
       'Tmin_2d', 'TempAmplitude_1d', 'TempAmplitude_2d', 'TempAmplitude_3d',
       'TaG_delta_2d', 'TaG_delta_3d', 'TminG_delta_1d', 'TminG_delta_2d',
       'TminG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'Precip_1d', 'Precip_2d',
       'Precip_3d', 'TempGrad_HS',
       'Tsnow_delta_2d', 'Tsnow_delta_5d']  # 22 features

RFECV = ['HSnum', 'TH01G', 'PR', 'HS_delta_1d', 'HS_delta_5d', 'TminG_delta_3d'
         'TminG_delta_5d', 'TmaxG_delta_3d', 'Precip_3d']  # 8 features

mod1_col = ['Stagione', 'N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG', 'HSnum',
            'HNnum', 'TH01G', 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d',
            'HS_delta_2d', 'HS_delta_3d', 'HS_delta_5d', 'HN_2d', 'HN_3d', 'HN_5d',
            'DaysSinceLastSnow', 'Tmin_2d', 'Tmax_2d', 'Tmin_3d', 'Tmax_3d',
            'Tmin_5d', 'Tmax_5d', 'TempAmplitude_1d', 'TempAmplitude_2d',
            'TempAmplitude_3d', 'TempAmplitude_5d', 'TaG_delta_1d', 'TaG_delta_2d',
            'TaG_delta_3d', 'TaG_delta_5d', 'TminG_delta_1d', 'TminG_delta_2d',
            'TminG_delta_3d', 'TminG_delta_5d', 'TmaxG_delta_1d', 'TmaxG_delta_2d',
            'TmaxG_delta_3d', 'TmaxG_delta_5d', 'T_mean', 'DegreeDays_Pos',
            'DegreeDays_cumsum_2d', 'DegreeDays_cumsum_3d', 'DegreeDays_cumsum_5d',
            'SnowDrift_1d', 'SnowDrift_2d', 'SnowDrift_3d', 'SnowDrift_5d',
            'FreshSWE', 'SeasonalSWE_cum', 'Precip_1d', 'Precip_2d', 'Precip_3d',
            'Precip_5d', 'Penetration_ratio', 'WetSnow_CS', 'WetSnow_Temperature',
            'TempGrad_HS', 'TH10_tanh', 'TH30_tanh', 'Tsnow_delta_1d',
            'Tsnow_delta_2d', 'Tsnow_delta_3d', 'Tsnow_delta_5d',
            'SnowConditionIndex', 'ConsecWetSnowDays', 'MF_Crust_Present',
            'New_MF_Crust', 'ConsecCrustDays', 'AvalDay', 'AvalDay_2d',
            'AvalDay_3d', 'AvalDay_5d']


missing_features = [col for col in mod1_col if col not in feature_set]
print(missing_features)
