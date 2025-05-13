# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:52:52 2025

@author: Christian
"""

PERMUTATION_RANKING = ['TH01G',
                       'TH10_tanh',
                       'TempAmplitude_5d',
                       'HSnum',
                       'HS_delta_3d',
                       'TH30_tanh',
                       'ConsecCrustDays',
                       'HS_delta_1d',
                       'TmaxG_delta_3d',
                       'MF_Crust_Present',
                       'Tmin_2d',
                       'TminG_delta_3d',
                       'DaysSinceLastSnow',
                       'TaG_delta_5d',
                       'TmaxG_delta_2d',
                       'TminG_delta_5d',
                       'Tsnow_delta_3d',
                       'TmaxG_delta_5d',
                       'TaG',
                       'TaG_delta_1d',
                       'DayOfSeason',
                       'T_mean',
                       'Tsnow_delta_2d',
                       'HS_delta_2d',
                       'Tsnow_delta_1d',
                       'TminG_delta_2d',
                       'TmaxG_delta_1d',
                       'Tsnow_delta_5d',
                       'Precip_1d',
                       'PR',
                       'TminG_delta_1d',
                       'TempAmplitude_3d',
                       'DegreeDays_Pos',
                       'TaG_delta_2d',
                       'Precip_2d']  # 35 features

ANOVA = ['HSnum',
         'TH01G',
         'TH03G',
         'HS_delta_1d',
         'Tmin_2d',
         'TminG_delta_5d',
         'TmaxG_delta_5d',
         'Precip_2d',
         'Precip_3d',
         'Precip_5d',
         'TempGrad_HS']  # 11 features

# SHAP = ['DayOfSeason', 'HSnum', 'HS_delta_5d', 'PR', 'HS_delta_3d', 'HS_delta_2d',
#         'Precip_5d', 'Precip_3d', 'TmaxG_delta_3d', 'TmaxG_delta_5d',
#         'TempAmplitude_2d', 'T_mean', 'TaG', 'HS_delta_1d', 'DaysSinceLastSnow',
#         'TmaxG_delta_2d', 'Precip_2d', 'Tmin_2d', 'TminG_delta_3d',
#         'TempAmplitude_3d', 'TminG_delta_5d', 'Tsnow_delta_3d', 'TmaxG_delta_1d',
#         'TminG_delta_2d', 'TH01G', 'TaG_delta_5d']

SHAP = ['HSnum' 'DayOfSeason' 'DaysSinceLastSnow' 'TH03G' 'TmaxG_delta_2d'
        'TH01G' 'Tsnow_delta_2d' 'PR' 'HS_delta_1d' 'TminG_delta_5d' 'Precip_3d'
        'HS_delta_5d' 'Tsnow_delta_3d' 'TaG_delta_5d' 'TmaxG_delta_5d'
        'TmaxG_delta_3d' 'Precip_5d' 'HS_delta_2d' 'TminG_delta_3d'
        'TmaxG_delta_1d' 'TaG_delta_1d' 'T_mean' 'HS_delta_3d' 'Precip_2d'
        'HNnum' 'Tmin_2d']


BFE = ['DayOfSeason', 'HS_delta_1d', 'HS_delta_5d', 'TempAmplitude_1d',
       'TmaxG_delta_3d', 'Precip_3d']

RFECV = ['HSnum' 'TH03G', 'PR', 'DayOfSeason', 'HS_delta_1d', 'HS_delta_2d'
         'HS_delta_5d', 'TempAmplitude_1d', 'TaG_delta_1d', 'TmaxG_delta_2d'
         'TmaxG_delta_5d', 'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d']
