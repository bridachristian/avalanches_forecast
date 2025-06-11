# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 20:59:07 2025

@author: Christian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
import seaborn as sns
import matplotlib.gridspec as gridspec


def eda_nivometeo(df):
    df = df.copy()
    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'])

    # Variabili da analizzare
    cols = ['N', 'V', 'VQ1', 'VQ2', 'TaG', 'TminG', 'TmaxG',
            'HSnum', 'HNnum', 'rho', 'TH01G', 'TH03G', 'PR', 'CS', 'B']

    # 1. Serie temporali
    df_plot = df[['DataRilievo'] + cols].dropna(subset=['DataRilievo'])
    df_plot.set_index('DataRilievo')[cols].plot(subplots=True, figsize=(14, 16),
                                                title='Serie Temporali delle variabili nivometeorologiche', lw=1)
    plt.tight_layout()
    plt.show()

    # 2. Istogrammi delle distribuzioni
    df[cols].hist(bins=30, figsize=(14, 10), color='darkcyan')
    plt.suptitle('Distribuzioni delle Variabili', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 3. Boxplot mensili per alcune variabili chiave
    df['Mese'] = pd.to_datetime(df['DataRilievo']).dt.month
    key_vars = ['TaG', 'TminG', 'TmaxG', 'HSnum', 'HNnum', 'rho']
    for var in key_vars:
        if var in df.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x='Mese', y=var, palette='viridis')
            plt.title(f'{var}: Boxplot Mensile')
            plt.grid(True)
            plt.show()

    # Seleziona le colonne rilevanti e rimuove i NaN
    corr_df = df[cols].dropna()

    if not corr_df.empty:
        # Converti cm in pollici
        cm_to_inch = 1 / 2.54
        fig_width = 15 * cm_to_inch
        fig_height = 12 * cm_to_inch

        # Calcola matrice di correlazione
        corr = corr_df.corr()

        # Crea maschera triangolo superiore
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Heatmap
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.2f',
            square=True,
            annot_kws={"size": 6},
            linewidths=0,          # 🔹 Disattiva le linee tra celle
            linecolor='none',      # 🔹 Nessun colore per linee
            cbar=False,  # Disabilita colorbar automatica
            ax=ax
        )

        # Colorbar manuale a tutta altezza
        norm = plt.Normalize(vmin=corr.values.min(), vmax=corr.values.max())
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])  # necessario per versioni recenti di Matplotlib

        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.0)
        cbar.ax.tick_params(labelsize=8)

        # Titolo e assi
        ax.set_title('Matrice di Correlazione delle Variabili',
                     fontsize=10, fontweight='bold', pad=14)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10,
                           rotation=90, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
        ax.grid(True, which='major', color='lightgray',
                linewidth=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\")
    file = 'mod1_filterd.csv'
    data = 'C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\MOD1_manipulation\\mod1_newfeatures_NEW.csv'

    plot_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\01_Mod1_statistics\\")

    data = data_folder/file

    df = pd.read_csv(data, sep=';', na_values=['NaN', '/', 'nan'])

    df['DataRilievo'] = pd.to_datetime(
        df['DataRilievo'], format="%d/%m/%Y")

    # --- Filtra dati da 1 dicembre 15 aprile ---
    # Crea colonne per mese e giorno
    df['mese'] = df['DataRilievo'].dt.month
    df['giorno'] = df['DataRilievo'].dt.day

    # Condizione logica:
    # - Dicembre (mese 12)
    # - Gennaio, Febbraio, Marzo (mesi 1–3)
    # - Aprile (solo fino al giorno 15)
    mask = (
        (df['mese'] == 12) |
        (df['mese'].isin([1, 2, 3])) |
        ((df['mese'] == 4) & (df['giorno'] <= 15))
    )

    # Applica filtro
    df = df[mask].copy()

    # Rimuovi le colonne ausiliarie se non servono
    df.drop(columns=['mese', 'giorno'], inplace=True)

    # --- 1. EDA tabella ---
    print(df.shape)
    print(df.dtypes)
    print(df.describe().T)

    table = df.describe().T

    # Valori mancanti assoluti e percentuali
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing': missing, '%': missing_percent})
    print(missing_summary[missing_summary['Missing'] > 0])

    # --- Aggiunta colonne dei missing alla tabella descrittiva ---
    table['Missing'] = missing
    table['Missing_perc'] = missing_percent.round(2)

    # --- Visualizza tabella finale ---
    print(table)

    eda_nivometeo(df)

    # --- EDA STATITICS PLOT ---
    # Escludi data e stagione
    features = df.columns.drop(['DataRilievo', 'Stagione'])

    # --- Setup variabili ---
    categoriche = ['N', 'V', 'VQ1', 'VQ2', 'CS', 'B']
    numeriche_continue = ['TaG', 'TminG', 'TmaxG']
    numeriche_positive = ['HSnum', 'PR']
    numeriche_negative = ['TH01G', 'TH03G']
    split_cols = ['HNnum', 'rho']

    # Global font size settings
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })

    # --- Plot 1: Prime 3 categoriche ---
    fig1 = plt.figure(figsize=(15, 3))
    for i, col in enumerate(categoriche[:3]):
        ax = fig1.add_subplot(1, 3, i+1)
        sns.countplot(x=df[col].dropna().astype(int).astype('category'), ax=ax)
        ax.set_title(f'Frequency of {col}')
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Seconde 3 categoriche ---
    fig2 = plt.figure(figsize=(15, 3))
    for i, col in enumerate(categoriche[3:]):
        ax = fig2.add_subplot(1, 3, i+1)
        sns.countplot(x=df[col].dropna().astype(int).astype('category'), ax=ax)
        ax.set_title(f'Frequency of {col}')
    plt.tight_layout()
    plt.show()

    # --- Plot 3: Numeriche continue ---
    fig3 = plt.figure(figsize=(15, 3))
    for i, col in enumerate(numeriche_continue):
        ax = fig3.add_subplot(1, 3, i+1)
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        sns.kdeplot(df[col].dropna(), fill=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

    # --- Plot 4: Numeriche positive ---
    fig4 = plt.figure(figsize=(15, 3))
    for i, col in enumerate(numeriche_positive):
        ax = fig4.add_subplot(1, 2, i+1)
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        ax.set_xlim(left=0)
        ax.set_title(f'Distribution of {col} (Positive)')
    plt.tight_layout()
    plt.show()

    # --- Plot 5: Numeriche negative ---
    fig5 = plt.figure(figsize=(15, 3))
    for i, col in enumerate(numeriche_negative):
        ax = fig5.add_subplot(1, 2, i+1)
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_title(f'Distribution of {col} (Negative)')
    plt.tight_layout()
    plt.show()

    # --- Plot 6: HNnum (split_col 1) ---
    fig6 = plt.figure(figsize=(15, 3))
    zero_count = len(df[df['HNnum'] == 0])
    nonzero_count = len(df[df['HNnum'] > 0])
    nonzero_vals = df[df['HNnum'] > 0]['HNnum']

    ax1 = fig6.add_subplot(1, 2, 1)
    ax1.bar(['Zero', 'Non-zero'], [zero_count,
            nonzero_count], color=['gray', 'blue'])
    ax1.set_title('Counts of 0 and Non-zero in HNnum')
    ax1.set_ylabel('Count')  # Aggiunta etichetta Y

    ax2 = fig6.add_subplot(1, 2, 2)
    sns.histplot(nonzero_vals, bins=30, kde=True, color='blue', ax=ax2)
    ax2.set_title(f'HNnum > 0 (count={nonzero_count})')
    plt.tight_layout()
    plt.show()

    # --- Plot 7: rho (split_col 2) ---
    fig7 = plt.figure(figsize=(15, 3))
    zero_count = len(df[df['rho'] == 0])
    nonzero_count = len(df[df['rho'] > 0])
    nonzero_vals = df[df['rho'] > 0]['rho']

    ax1 = fig7.add_subplot(1, 2, 1)
    ax1.bar(['Zero', 'Non-zero'], [zero_count,
            nonzero_count], color=['gray', 'blue'])
    ax1.set_title('Counts of 0 and Non-zero in rho')
    ax1.set_ylabel('Count')  # Aggiunta etichetta Y

    ax2 = fig7.add_subplot(1, 2, 2)
    sns.histplot(nonzero_vals, bins=30, kde=True, color='blue', ax=ax2)
    ax2.set_title(f'rho > 0 (count={nonzero_count})')
    plt.tight_layout()
    plt.show()

    # --- CLIMATOLOGIA ---
    # Variabili numeriche da analizzare
    # Assumendo df già filtrato per dicembre-15 aprile e senza 29 febbraio
    vars_to_plot = ["HSnum", "TaG", "TminG", "TmaxG", "PR", "TH01G", "TH03G"]

    # Apply Seaborn style
    sns.set(style='whitegrid')

    # Global font size settings – slightly increased
    plt.rcParams.update({
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 20
    })

    for var in vars_to_plot:
        df['DOY'] = df['DataRilievo'].dt.strftime('%m-%d')
        df_filtered = df[df['DOY'] != '02-29'].copy()

        valid_days = pd.date_range(
            '2000-12-01', '2001-04-15').strftime('%m-%d').tolist()
        df_filtered = df_filtered[df_filtered['DOY'].isin(valid_days)].copy()

        daily_stats = df_filtered.groupby('DOY')[var].agg([
            'mean', 'median', 'min', 'max',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ])
        daily_stats.columns = ['mean', 'median', 'min', 'max', 'q25', 'q75']

        dec_days = pd.date_range(
            '2000-12-01', '2000-12-31').strftime('%m-%d').tolist()
        jan_apr_days = pd.date_range(
            '2001-01-01', '2001-04-15').strftime('%m-%d').tolist()
        ordered_days = dec_days + jan_apr_days
        daily_stats = daily_stats.reindex(ordered_days)

        plt.figure(figsize=(16, 4))

        # Fill interquartile range
        plt.fill_between(daily_stats.index, daily_stats['q25'], daily_stats['q75'],
                         color='lightblue', alpha=0.5, label='IQR (25–75%)')

        # Main lines
        plt.plot(daily_stats.index,
                 daily_stats['mean'], label='Mean', color='green', linewidth=2)
        # plt.plot(daily_stats.index, daily_stats['median'],
        #          label='Median', color='black', linestyle='--', linewidth=2)

        # Min/Max
        plt.plot(daily_stats.index,
                 daily_stats['min'], label='Min', color='blue', alpha=0.3, linewidth=1)
        plt.plot(daily_stats.index,
                 daily_stats['max'], label='Max', color='red', alpha=0.3, linewidth=1)

        # Zero line
        plt.axhline(0, color='gray', linestyle=':', linewidth=1)

        # Titles and labels
        plt.title(f'Seasonal Daily Distribution – {var}', fontsize=18)
        plt.xlabel('Day (MM-DD)', fontsize=16)
        plt.ylabel(var, fontsize=16)

        # Improve x-axis ticks
        # Weekly ticks (every 7 days)
        xtick_positions = list(range(0, len(ordered_days), 7))
        xtick_labels = [pd.to_datetime(
            ordered_days[i], format='%m-%d').strftime('%b-%d') for i in xtick_positions]

        # Apply improved x-axis ticks
        plt.xticks(ticks=xtick_positions, labels=xtick_labels,
                   rotation=90, ha='right')
        # plt.xticks(
        #     ticks=range(0, len(ordered_days), 7),
        #     labels=[ordered_days[i] for i in range(0, len(ordered_days), 7)],
        #     rotation=45
        # )

        # Legend and grid
        plt.legend(loc='upper left', bbox_to_anchor=(
            1.02, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.grid(True, linestyle=':', linewidth=0.5)

        plt.show()

        # Save (optional)
        # out_path = plot_folder / f"{var}_daily_distribution.png"
        # plt.savefig(out_path, dpi=300, bbox_inches='tight')
        # plt.close()
