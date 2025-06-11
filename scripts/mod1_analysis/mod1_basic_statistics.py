# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:01:43 2024

@author: Christian
"""
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json
import seaborn as sns

# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


def make_autopct(values):
    '''
    The function makes the text in format xxx% (nn) for counting the
    avalanche events and reporting on a pie chart

    Parameters
    ----------
    values : np.array
        array of values to count.

    Returns
    -------
    str
        string to insert in the plot.

    '''
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct


def L1_counts(df, savefig=False):
    '''
    Count and plot pie chart of total avalanche days vs non avalanche days

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L1

    Returns
    -------
    None.

    '''
    global plot_folder

    # Count occurrences of NaN
    nan_count = df['L1'].isna().sum()

    # Count occurrences of values equal to 0
    zero_count = (df['L1'] == 0).sum()

    # Count occurrences of values not equal to 0
    non_zero_count = ((df['L1'] != 0) & (~df['L1'].isna())).sum()

    # Create DataFrame
    counts_df = pd.DataFrame({
        'Count': [nan_count, zero_count, non_zero_count]
    }, index=['NaN', 'No Avalanche', 'Avalanche'])

    plt.figure(figsize=(12, 9))
    plt.pie(counts_df['Count'], labels=counts_df.index,
            autopct=make_autopct(counts_df['Count']), startangle=140)
    plt.title('Distribution of Avalanche data from L1')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L1_distribution_avalanche_days_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L1_classification(df, savefig=False):
    '''
    Classify the size and the number of avalanches following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L1

    Returns
    -------
    None.

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L1']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L1'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of number and size of observed avalanches from L1')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L1_classification_of_magnitude_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L1_classification_bar(df, savefig=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    global plot_folder
    global codice_nivometeo

    # Mappatura delle etichette
    index_mapping = codice_nivometeo['L1']
    index_mapping_numeric = {float(k): v for k, v in index_mapping.items()}

    # Rimuove L1=0 se presente
    class_counts = df['L1'].value_counts().sort_index()
    class_counts = class_counts[class_counts.index != 0]

    # Aggrega valori 6–9 in una nuova categoria
    aggregated_counts = {}
    for idx, count in class_counts.items():
        if idx in [6, 7, 8, 9]:
            aggregated_counts['6–9'] = aggregated_counts.get('6–9', 0) + count
        else:
            aggregated_counts[int(idx)] = count

    # Ordina per chiave (1–5 numeriche, poi "6–9")
    ordered_keys = [1, 2, 3, 4, 5, '6–9']
    grouped = pd.DataFrame({
        'L1': ordered_keys,
        'count': [aggregated_counts.get(k, 0) for k in ordered_keys]
    })

    # Etichette descrittive
    legend_labels = {
        1: 'Small avalanches (sluff)',
        2: 'Medium-size avalanches',
        3: 'Many medium-sized avalanches',
        4: 'Single large avalanches',
        5: 'Several large avalanches',
        '6–9': 'Old classification'
    }
    grouped['label'] = grouped['L1'].map(legend_labels)

    # Plot
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })

    cm_to_inch = 1 / 2.54
    fig_width = 15 * cm_to_inch
    fig_height = 6 * cm_to_inch

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    # --- Main plot ---
    bars = ax.bar(grouped['L1'].astype(str), grouped['count'],
                  color='steelblue', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(grouped['count']) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    ax.set_title('Avalanche Sizes (L1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Avalanche Size Code')
    ax.set_ylabel('N. Avalanches')
    ax.set_ylim(0, 350)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['L1'].astype(str), rotation=0)

    # --- Legend ---
    legend_handles = []
    for code, label in zip(grouped['L1'], grouped['label']):
        patch = mpatches.Patch(color='steelblue', label=f'{code}: {label}')
        legend_handles.append(patch)

    ax.legend(
        handles=legend_handles,
        title='',
        fontsize=8,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


def L2_classification(df, savefig=False):
    '''
    Classify the type of avalanche following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L2

    Returns
    -------
    None.

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L2']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L2'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of types of avalanches from L2')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L2_types_of_avalanches_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L2_classification_bar(df, savefig=False):
    '''
    Classify the type of avalanche following mod.1 AINEVA and
    plot bar chart to summarise L2 types.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from Mod.1 AINEVA with 'L2' column.
    savefig : bool, optional
        Whether to save the figure to disk. Default is False.

    Returns
    -------
    None
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    global plot_folder
    global codice_nivometeo

    # Mapping
    index_mapping = codice_nivometeo['L2']
    index_mapping_numeric = {float(k): v for k, v in index_mapping.items()}

    # Count L2 classes and remove code 0 if present
    class_counts = df['L2'].value_counts().sort_index()
    class_counts = class_counts[class_counts.index != 0]

    # Build labels
    # labels = [index_mapping_numeric.get(x, str(x)) for x in class_counts.index]

    labels = ['Surface slab avalanche',
              'Ground slab avalanches',
              'Surface Loose snow avalanches',
              'Ground Loose snow avalanches',
              'Surface slab and loose snow avalanches',
              'Ground slab and loose snow avalanches']

    # Create DataFrame
    grouped = pd.DataFrame({
        'L2': class_counts.index.astype(int),
        'label': labels,
        'count': class_counts.values
    })

    # Define color by L2 code
    def get_color(code):
        if code in [1, 2]:
            return 'steelblue'
        elif code in [3, 4]:
            return 'darkorange'
        elif code in [5, 6]:
            return 'mediumorchid'
        else:
            return 'gray'

    colors = grouped['L2'].map(get_color)

    # Figure setup
    cm_to_inch = 1 / 2.54
    fig_width = 15 * cm_to_inch
    fig_height = 6 * cm_to_inch

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot bars
    bars = ax.bar(grouped['L2'].astype(str), grouped['count'],
                  color=colors, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(grouped['count']) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # Titles and labels
    ax.set_title('Avalanche Type (L2)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Avalanche Type Code')
    ax.set_ylabel('N. Avalanches')
    ax.set_ylim(0, 470)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['L2'].astype(str), rotation=0)

    # Legend
    legend_handles = []
    seen_codes = set()
    for code, label in zip(grouped['L2'], grouped['label']):
        if code not in seen_codes:
            patch = mpatches.Patch(color=get_color(
                code), label=f'{code}: {label}')
            legend_handles.append(patch)
            seen_codes.add(code)

    ax.legend(
        handles=legend_handles,
        title='',
        fontsize=8,
        title_fontsize=9,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()


def L3_classification(df, savefig=False):
    '''
    Classify the aspect of avalanche releasse following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L3

    Returns
    -------
    None.

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L3']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L3'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of aspect of avalanche release from L3')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L3_aspect_of_release_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L4_classification(df, savefig=False):
    '''
    Classify the elevation of avalanche release following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L4

    Returns
    -------
    None.

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L4']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L4'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of release avalanches altitude from L4')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L4_altitude_of_release_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L5_classification(df, savefig=False):
    '''
    Classify the time period of avalanche release following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L5

    Returns
    -------
    None.

    Note
    ------
    Be careful, the value '/' is treated as NaN instead of
    'period not established'. This function should be modiified!

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L5']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L5'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of period of avalanches release from L5')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L5_timeperiod_of_release.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L6_classification(df, savefig=False):
    '''
    Classify the artifical triggering mechanism following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L6

    Returns
    -------
    None.

    Note
    ------
    Be careful, the value '/' is treated as NaN instead of
    'period not established'. This function should be modiified!

    '''
    global plot_folder
    global codice_nivometeo

    index_mapping = codice_nivometeo['L6']
    index_mapping_numeric = {
        float(key): value for key, value in index_mapping.items()}

    class_distribution = df['L6'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(
        index_mapping_numeric)

    plt.figure(figsize=(12, 9))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of avalanche triggering results from L6')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L6_artifiial_triggering_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L1_timeline_season(df, savefig=False):
    '''
    Count avalanche days vs non avalanche days and plot a timeline
    grouped by month or year

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L1

    Returns
    -------
    None.

    '''
    global plot_folder

    def transform_value(x):
        if pd.isna(x):
            return np.nan
        elif x == 0:
            return 0
        else:
            return 1

    df['AvNoAv'] = df['L1'].apply(transform_value)

    grouped_df = df[['Stagione', 'AvNoAv']].groupby(
        'Stagione').sum(numeric_only=True).reset_index()

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.bar(grouped_df['Stagione'], grouped_df['AvNoAv'])
    plt.title('Number of avalanche days for each winter season')
    plt.xlabel('Season')
    plt.ylabel('Avalanche days')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
    # plt.grid(True)  # Add grid lines

    for i, stagione in enumerate(grouped_df['Stagione']):
        if i % 5 == 0 and i != 0:  # Check if it's the 5th season and not the first one
            plt.axvline(x=stagione, color='gray', linestyle='--',
                        linewidth=0.5)  # Add vertical line

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'avalanche_days_plot.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L1_timeline_season_class(df, savefig=False):
    """
    Plot absolute and percentage number of avalanche observation days per winter season.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'L1' and 'Stagione' columns.
        'L1' should be 1 for avalanche, 0 for no avalanche, and NaN for missing.
    savefig : bool, optional
        If True, saves the plots to 'plot_folder'. Default is False.

    Returns
    -------
    None
    """
    def classify_L1(x):
        if pd.isna(x):
            return np.nan
        elif x == 0:
            return 0
        else:
            return 1

    df['AvNoAv'] = df['L1'].apply(classify_L1)

    grouped_df = df.groupby('Stagione').agg(
        Avalanche_Days=('AvNoAv', lambda x: x.eq(1).sum()),
        No_Avalanche_Days=('AvNoAv', lambda x: x.eq(0).sum()),
        Missing_Days=('AvNoAv', lambda x: x.isna().sum())
    ).reset_index()

    grouped_df['Total_Days'] = grouped_df[['Avalanche_Days',
                                           'No_Avalanche_Days', 'Missing_Days']].sum(axis=1)

    # Determine which labels to show (reduce clutter)
    stagioni = grouped_df['Stagione'].tolist()
    label_step = max(1, len(stagioni) // 10)  # Show approx. 10 labels
    xticks_labels = [s if i % label_step ==
                     0 else '' for i, s in enumerate(stagioni)]

    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    cm_to_inch = 1 / 2.54
    fig_width = 15 * cm_to_inch
    fig_height = 12 * cm_to_inch
    import matplotlib.ticker as ticker

    # Genera xtick positions
    xtick_positions = list(range(len(stagioni)))

    # Etichette ogni 5 stagioni, altrimenti stringa vuota
    # xtick_labels = [label if i % 5 == 0 else '' for i, label in enumerate(xticks_labels)]

    plt.figure(figsize=(fig_width, fig_height))

    # Bar plots
    plt.bar(stagioni, grouped_df['Avalanche_Days'],
            color="#44a5c2", label='Avalanche Days')
    plt.bar(stagioni, grouped_df['No_Avalanche_Days'],
            bottom=grouped_df['Avalanche_Days'], color="#ffae49", label='No Avalanche Days')
    plt.bar(stagioni, grouped_df['Missing_Days'],
            bottom=grouped_df['Avalanche_Days'] +
            grouped_df['No_Avalanche_Days'],
            color="#D3D3D3", label='Missing Data')

    # Titoli e label
    plt.title('Number of Avalanche Days per Winter Season',
              fontsize=14, fontweight='bold')
    plt.xlabel('Winter Season')
    plt.ylabel('Number of Days')

    # Applica solo le etichette ogni 5 stagioni
    plt.xticks(ticks=xtick_positions, rotation=90)

    # Griglia Y e X ogni 5
    ax = plt.gca()
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.3)

    plt.ylim(0, 174)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if savefig:
        (plot_folder / 'avalanche_days_absolute.png').parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_folder / 'avalanche_days_absolute.png', dpi=300)
    else:
        plt.show()

    # --- Percentage Plot ---
    grouped_df['Avalanche_%'] = (
        grouped_df['Avalanche_Days'] / grouped_df['Total_Days']) * 100
    grouped_df['No_Avalanche_%'] = (
        grouped_df['No_Avalanche_Days'] / grouped_df['Total_Days']) * 100
    grouped_df['Missing_%'] = (
        grouped_df['Missing_Days'] / grouped_df['Total_Days']) * 100

    plt.figure(figsize=(12, 7))
    plt.bar(stagioni, grouped_df['Avalanche_%'],
            color="#44a5c2", label='Avalanche Days (%)')
    plt.bar(stagioni, grouped_df['No_Avalanche_%'],
            bottom=grouped_df['Avalanche_%'], color="#ffae49", label='No Avalanche Days (%)')
    plt.bar(stagioni, grouped_df['Missing_%'],
            bottom=grouped_df['Avalanche_%'] + grouped_df['No_Avalanche_%'],
            color="#D3D3D3", label='Missing Data (%)')

    plt.title(
        'Percentage of Avalanche Observation Days per Winter Season', fontsize=14)
    plt.xlabel('Winter Season')
    plt.ylabel('Percentage of Days (%)')
    plt.xticks(ticks=range(len(stagioni)), labels=xticks_labels, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(plot_folder / 'avalanche_days_percentage.png', dpi=300)
    else:
        plt.show()


def L1_period(df, savefig=False):
    '''
    Count avalanche days vs non avalanche days and plot a timeline
    grouped by month or year

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L1

    Returns
    -------
    None.

    '''
    global plot_folder

    def transform_value(x):
        if pd.isna(x):
            return np.nan
        elif x == 0:
            return 0
        else:
            return 1

    df['AvNoAv'] = df['L1'].apply(transform_value)
    df['NoAval'] = df['L1'].apply(transform_value)
    df['NaNDay'] = df['L1'].apply(transform_value)

    df['season_day'] = pd.to_datetime(df[['Giorno', 'Mese']].assign(
        Anno=2024).rename(columns={'Giorno': 'day', 'Mese': 'month', 'Anno': 'year'}))

    # Create a new column 'year' and adjust it based on the month
    df['year'] = 2024
    df.loc[df['Mese'].isin([10, 11, 12]), 'year'] = 2023

    # Create a new column 'season_day' by concatenating 'Giorno', 'Mese', and the adjusted year
    df['season_day'] = pd.to_datetime(df[['Giorno', 'Mese', 'year']].rename(
        columns={'Giorno': 'day', 'Mese': 'month', 'year': 'year'}))

    # Group by 'Stagione' and aggregate columns using different functions
    grouped_df = df.groupby('season_day').agg({
        'AvNoAv': 'sum',             # Sum of avalanche days
        'NoAval': lambda x: x.eq(0).sum(),   # Count of 0 values
        'NaNDay': lambda x: x.isna().sum()   # Count of NaN values
    }).reset_index()

    # Calculate percentage for each column

    grouped_df['total_data'] = (grouped_df['AvNoAv'] + grouped_df['NoAval'])

    grouped_df['cumulative_NaNDay'] = grouped_df['total_data'] + \
        grouped_df['NaNDay']

    # Plotting
    plt.figure(figsize=(12, 9))

    # Plot step lines
    observ_line = plt.step(grouped_df['season_day'], grouped_df['total_data'],
                           where='mid', color="#44a5c2", label='N. observation')
    nan_line = plt.step(grouped_df['season_day'], grouped_df['cumulative_NaNDay'],
                        where='mid', color="#ffae49", label='N. observation + NaN')

    # Fill areas
    plt.fill_between(grouped_df['season_day'], 0,
                     grouped_df['total_data'], step='mid', alpha=0.5, color="#44a5c2")
    plt.fill_between(grouped_df['season_day'], grouped_df['total_data'],
                     grouped_df['cumulative_NaNDay'], step='mid', alpha=0.5, color="#ffae49")

    # Titles and labels
    plt.title('Number of Observation Years per Calendar Day')
    plt.xlabel('Season')
    plt.ylabel('Years of Observation')
    plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
    plt.legend()
    plt.grid(which='both', alpha=0.2)
    plt.gca().xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=None, bymonthday=1, interval=1))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(
        bymonthday=16, interval=1))  # Minor ticks on the 15th of each month

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d - %m'))
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%d - %m'))
    # Customize tick lengths
    plt.gca().tick_params(axis='x', which='major', length=7)
    plt.gca().tick_params(axis='x', which='minor', length=7)

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'observation_years_per_calendarday.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def L3L4_elevation_aspect_plot(df, savefig=False):
    '''
    Classify the elevation of avalanche release following mod.1 AINEVA and
    plot pie chart to summarise

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L4

    Returns
    -------
    None.

    '''
    global plot_folder
    global codice_nivometeo

    aspects_mapping = codice_nivometeo['L3']
    altitude_mapping = codice_nivometeo['L4']

    small_df = df[['L3', 'L4']]

    # classify all L3 values in N, S, W, E
    small_df.loc[small_df['L3'] == 1, 'L3'] = 'N'
    small_df.loc[small_df['L3'] == 2, 'L3'] = 'E'
    small_df.loc[small_df['L3'] == 3, 'L3'] = 'S'
    small_df.loc[small_df['L3'] == 4, 'L3'] = 'W'
    small_df.loc[small_df['L3'] == 5, 'L3'] = 'N'
    small_df.loc[small_df['L3'] == 6, 'L3'] = 'S'
    small_df.loc[small_df['L3'] == 7, 'L3'] = 'all'
    small_df.loc[small_df['L3'] == 8, 'L3'] = 'W'

    # classify all L4 values based on bands <1800, 1800-2300, 2300-2800, > 2800
    #                                         4        3          2          1
    small_df.loc[small_df['L4'] == 1, 'L4'] = 4
    small_df.loc[small_df['L4'] == 2, 'L4'] = 4
    small_df.loc[small_df['L4'] == 3, 'L4'] = 4
    small_df.loc[small_df['L4'] == 4, 'L4'] = 4
    small_df.loc[small_df['L4'] == 5, 'L4'] = 3
    small_df.loc[small_df['L4'] == 6, 'L4'] = 3
    small_df.loc[small_df['L4'] == 7, 'L4'] = 2
    small_df.loc[small_df['L4'] == 8, 'L4'] = 1
    small_df.loc[small_df['L4'] == 9, 'L4'] = 0

    total = small_df.groupby(['L3', 'L4']).size().reset_index(name='count')
    total = total.iloc[1:]  # exclude no avalanche

    total = total[total['L3'] != 'all']
    total = total[total['L4'] != 0]

    # Pivot the DataFrame
    wide_df = total.pivot(index='L3', columns='L4',
                          values='count').reset_index()

    aspects1 = wide_df.iloc[:, [0, 1]]
    aspects2 = wide_df.iloc[:, [0, 2]]
    aspects3 = wide_df.iloc[:, [0, 3]]
    aspects4 = wide_df.iloc[:, [0, 4]]

    # Plotting the data

    directions = aspects1.iloc[:, 0].tolist()
    counts1 = aspects1.iloc[:, 1].tolist()
    counts2 = aspects2.iloc[:, 1].tolist()
    counts3 = aspects3.iloc[:, 1].tolist()
    counts4 = aspects4.iloc[:, 1].tolist()

    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()

    # # Convert wind speeds to colors
    # tot_counts = total['count'].values
    # colors1 = [plt.cm.viridis(i / max(tot_counts)) for i in counts1]
    # colors2 = [plt.cm.viridis(i / max(tot_counts)) for i in counts2]
    # colors3 = [plt.cm.viridis(i / max(tot_counts)) for i in counts3]
    # colors4 = [plt.cm.viridis(i / max(tot_counts)) for i in counts4]
    # Normalizza tra 0 e 50: i valori >50 saranno trattati come 50
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # clip=True tronca i valori > vmax
    norm = Normalize(vmin=0, vmax=50, clip=True)
    cmap = cm.YlOrRd

    colors1 = [cmap(norm(i)) for i in counts1]
    colors2 = [cmap(norm(i)) for i in counts2]
    colors3 = [cmap(norm(i)) for i in counts3]
    colors4 = [cmap(norm(i)) for i in counts4]

    def get_text_color(rgba):
        r, g, b, _ = rgba
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return 'white' if luminance < 0.5 else 'black'

    def add_labels(bars, counts, radius_factor, colors):
        for i, bar in enumerate(bars):
            angle_rad = angles[i]
            text_color = get_text_color(colors[i])
            ax.text(angle_rad, radius_factor, str(counts[i]),
                    color=text_color, fontsize=20,
                    ha='center', va='center',
                    transform=ax.get_xaxis_transform())

    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw={'projection': 'polar', 'frame_on': False})

    bars4 = ax.bar(angles, 1, width=1.57, color=colors4,
                   edgecolor='lightgrey', linestyle=':', alpha=1)
    bars3 = ax.bar(angles, 0.8, width=1.57, color=colors3,
                   edgecolor='lightgrey', linestyle=':', alpha=1)
    bars2 = ax.bar(angles, 0.60, width=1.57, color=colors2,
                   edgecolor='lightgrey', linestyle=':', alpha=1)
    bars1 = ax.bar(angles, 0.4, width=1.57, color=colors1,
                   edgecolor='lightgrey', linestyle=':', alpha=1)
    # bars0 = ax.bar(angles, 0.2, width=1.57, color=colors1,
    #                edgecolor='grey', linestyle=':', alpha=1)

    # <1800, 1800-2300, 2300-2800, > 2800
    # Custom radius labels
    radius_labels = [2800, 2300, 1800, 1300]
    for radius, label in zip([0.4, 0.6, 0.8, 1.0], radius_labels):
        ax.text(np.pi/4, radius-0.1, f'{label}m',
                fontsize=16, color='dimgrey')

    ax.set_rticks([])  # Remove the default radial ticks
    ax.grid(False)

    # Add count labels to each sector
    # def add_labels(bars, counts, radius_factor):
    #     for i, bar in enumerate(bars):
    #         angle_rad = angles[i]
    #         ax.text(angle_rad, radius_factor, str(counts[i]),
    #                 transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=10)
    #         # , bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    # add_labels(bars1, counts1, 0.2)
    # add_labels(bars2, counts2, 0.5)
    # add_labels(bars3, counts3, 0.7)
    # add_labels(bars4, counts4, 0.9)

    # Etichette conteggi con contrasto leggibile
    add_labels(bars1, counts1, 0.2, colors1)
    add_labels(bars2, counts2, 0.5, colors2)
    add_labels(bars3, counts3, 0.7, colors3)
    add_labels(bars4, counts4, 0.9, colors4)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['North', 'East', 'South', 'West'], fontsize=16)

    # # n.valanghe classificate come 'diverse altitidutini' e 'diverse esposizioni'
    # av_class_all = 666 - sum(tot_counts)
    # plt.figtext(0.5, -0.5, f'Note:\nAval. days classified on aspect and elevation: {sum(tot_counts)}\nAval. days at various altitudes and various aspects: {av_class_all}',
    #             horizontalalignment='center', fontsize=10, wrap=True)

    plt.title(f'Geographical distribution of avalanche release',
              fontsize=22, fontweight='bold')
    plt.tight_layout()

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'geograpyical_distribution_L3L4.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def avalanche_seasonality_by_type(df, savefig=False):
    global plot_folder
    global codice_nivometeo

    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'], errors='coerce')
    df_filtered = df[(df['L2'] != 0) & (~df['L2'].isna())]

    # Etichette tipo valanga
    index_mapping = codice_nivometeo['L2']
    index_mapping_numeric = {float(k): v for k, v in index_mapping.items()}
    df_filtered['L2_label'] = df_filtered['L2'].map(index_mapping_numeric)

    # Estrai giorno e mese
    df_filtered['day'] = df_filtered['DataRilievo'].dt.day
    df_filtered['month'] = df_filtered['DataRilievo'].dt.month

    # Escludi mesi fuori da intervallo desiderato
    df_filtered = df_filtered[df_filtered['month'].isin([12, 1, 2, 3, 4])]

    # Crea asse temporale con anno fittizio per ordinamento
    def assign_fake_date(row):
        year = 1999 if row['month'] == 12 else 2000
        return datetime(year, row['month'], row['day'])

    df_filtered['fake_date'] = df_filtered.apply(assign_fake_date, axis=1)

    # Aggrega eventi per tipo e giorno
    grouped = df_filtered.groupby(
        ['fake_date', 'L2_label']).size().reset_index(name='count')
    pivot_df = grouped.pivot(
        index='fake_date', columns='L2_label', values='count').fillna(0)

    # Ordina
    pivot_df = pivot_df.sort_index()

    # Colori per tipo
    def get_color(label):
        if 'slab' in label.lower():
            return 'steelblue'
        elif 'loose' in label.lower():
            return 'darkorange'
        elif '+' in label:
            return 'mediumorchid'
        else:
            return 'gray'

    colors = [get_color(col) for col in pivot_df.columns]

    # Plot
    plt.figure(figsize=(15, 6))
    for label, color in zip(pivot_df.columns, colors):
        plt.plot(pivot_df.index, pivot_df[label],
                 label=label, color=color, linewidth=2)

    plt.title('Seasonal Avalanche Frequency by Type (1 Dec – 15 Apr)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Number of Avalanches', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Avalanche Type', fontsize=10,
               title_fontsize=12, loc='upper right')
    plt.tight_layout()

    # Format asse X come "dd - mm"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d - %m'))
    plt.xticks(rotation=45)

    if savefig:
        outpath = plot_folder / 'avalanche_seasonality_by_type_doy_fixed.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def avalanche_seasonality_by_type_weekly_grouped(df, savefig=False, year_range=None):
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    global plot_folder
    global codice_nivometeo

    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'], errors='coerce')
    df_filtered = df[(df['L2'] != 0) & (~df['L2'].isna())].copy()

    # Mappa codici L2 in etichette
    index_mapping = codice_nivometeo['L2']
    index_mapping_numeric = {float(k): v for k, v in index_mapping.items()}
    df_filtered['L2_label'] = df_filtered['L2'].map(index_mapping_numeric)

    # Estrai giorno e mese
    df_filtered['day'] = df_filtered['DataRilievo'].dt.day
    df_filtered['month'] = df_filtered['DataRilievo'].dt.month

    # Filtra mesi dicembre-aprile
    df_filtered = df_filtered[df_filtered['month'].isin(
        [12, 1, 2, 3, 4])].copy()

    # Crea data fittizia per ordinamento
    def assign_fake_date(row):
        year = 1999 if row['month'] == 12 else 2000
        return datetime(year, row['month'], row['day'])

    df_filtered['fake_date'] = df_filtered.apply(assign_fake_date, axis=1)

    # Raggruppa e conta eventi per data e tipo L2
    grouped = df_filtered.groupby(
        ['fake_date', 'L2']).size().reset_index(name='count')

    # Pivot per avere colonne L2 e indice fake_date
    pivot_df = grouped.pivot(
        index='fake_date', columns='L2', values='count').fillna(0)

    # Somma colonne per gruppi di interesse
    slab_sum = pivot_df[[col for col in [1, 2]
                         if col in pivot_df.columns]].sum(axis=1)
    loose_sum = pivot_df[[col for col in [3, 4]
                          if col in pivot_df.columns]].sum(axis=1)
    both_sum = pivot_df[[col for col in [5, 6]
                         if col in pivot_df.columns]].sum(axis=1)

    aggregated_df = pd.DataFrame({
        'Slab': slab_sum,
        'Loose Snow': loose_sum,
        'Both type': both_sum
    })

    aggregated_df.index = pd.to_datetime(aggregated_df.index)
    aggregated_df = aggregated_df.sort_index()

    # Resample settimanale (somma)
    # Settimane che iniziano il lunedì
    weekly_df = aggregated_df.resample('W-MON').mean()

    # Colori e stili linea per maggiore differenziazione
    colors = {
        'Slab': 'steelblue',
        'Loose Snow': 'darkorange',
        'Both type': 'mediumorchid'
    }
    linestyles = {
        'Slab': 'solid',
        'Loose Snow': 'solid',
        'Both type': 'solid'
    }


fig_width = 15 / 2.54  # 15 cm
fig_height = 12 / 2.54  # 12 cm

plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
for label in weekly_df.columns:
    ax.plot(
        weekly_df.index, weekly_df[label],
        label=label,
        color=colors[label],
        linestyle=linestyles[label],
        linewidth=2
    )

# Titolo dinamico
if year_range:
    title = f'Weekly Avalanche Frequency by Grouped Type (Dec – Apr, {year_range})'
else:
    title = 'Weekly Mean Avalanche by Grouped Type (Dec – Apr)'
ax.set_title(title, fontsize=14, fontweight='bold')
ax.set_xlabel('Week', fontsize=12)
ax.set_ylabel('Mean N. Avalanches per week', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Formatta le etichette dell'asse X


def format_week_label(x, pos=None):
    dt = mdates.num2date(x)
    return dt.strftime('%d %b')


ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_week_label))
plt.xticks(rotation=0)

# Legenda sotto il grafico
legend = ax.legend(
    title='Avalanche Type',
    fontsize=9,
    title_fontsize=10,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False
)

# Rende il titolo della legenda in grassetto
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Spazio per la legenda
plt.show()

#
# # Figure setup
# cm_to_inch = 1 / 2.54
# fig_width = 15 * cm_to_inch
# fig_height = 6 * cm_to_inch

# fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# # Plot bars
# bars = ax.bar(grouped['L2'].astype(str), grouped['count'],
#               color=colors, edgecolor='black')

# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, height + max(grouped['count']) * 0.01,
#             f'{int(height)}', ha='center', va='bottom', fontsize=8)

# # Titles and labels
# ax.set_title('Avalanche Type (L2)', fontsize=14, fontweight='bold')
# ax.set_xlabel('Avalanche Type Code')
# ax.set_ylabel('N. Avalanches')
# ax.set_ylim(0, 470)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# ax.set_xticks(range(len(grouped)))
# ax.set_xticklabels(grouped['L2'].astype(str), rotation=0)

# # Legend
# legend_handles = []
# seen_codes = set()
# for code, label in zip(grouped['L2'], grouped['label']):
#     if code not in seen_codes:
#         patch = mpatches.Patch(color=get_color(
#             code), label=f'{code}: {label}')
#         legend_handles.append(patch)
#         seen_codes.add(code)

# ax.legend(
#     handles=legend_handles,
#     title='',
#     fontsize=8,
#     title_fontsize=9,
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.35),
#     ncol=2,
#     frameon=False
# )

# plt.tight_layout()
# plt.subplots_adjust(bottom=0.25)
# plt.show()


def avalanche_seasonality_by_type_2(df, savefig=False):
    global plot_folder
    global codice_nivometeo

    df['DataRilievo'] = pd.to_datetime(df['DataRilievo'], errors='coerce')
    df_filtered = df[(df['L2'] != 0) & (~df['L2'].isna())].copy()

    # Etichette tipo valanga
    index_mapping = codice_nivometeo['L2']
    index_mapping_numeric = {float(k): v for k, v in index_mapping.items()}
    df_filtered['L2_label'] = df_filtered['L2'].map(index_mapping_numeric)

    # Estrai giorno e mese
    df_filtered['day'] = df_filtered['DataRilievo'].dt.day
    df_filtered['month'] = df_filtered['DataRilievo'].dt.month

    # Escludi mesi fuori da intervallo desiderato
    df_filtered = df_filtered[df_filtered['month'].isin(
        [12, 1, 2, 3, 4])].copy()

    # Crea asse temporale con anno fittizio per ordinamento
    def assign_fake_date(row):
        year = 1999 if row['month'] == 12 else 2000
        return datetime(year, row['month'], row['day'])

    df_filtered['fake_date'] = df_filtered.apply(assign_fake_date, axis=1)

    # Aggrega eventi per tipo e giorno
    grouped = df_filtered.groupby(
        ['fake_date', 'L2']).size().reset_index(name='count')
    pivot_df = grouped.pivot(
        index='fake_date', columns='L2', values='count').fillna(0)

    # Somma le colonne secondo i gruppi richiesti
    slab_sum = pivot_df[[col for col in [1, 2]
                         if col in pivot_df.columns]].sum(axis=1)
    loose_sum = pivot_df[[col for col in [3, 4]
                          if col in pivot_df.columns]].sum(axis=1)
    both_sum = pivot_df[[col for col in [5, 6]
                         if col in pivot_df.columns]].sum(axis=1)

    # Costruisci dataframe con i 3 gruppi
    grouped_df = pd.DataFrame({
        'slab': slab_sum,
        'loose snow': loose_sum,
        'both': both_sum
    })

    # Colori fissi
    colors = {
        'slab': 'steelblue',
        'loose snow': 'darkorange',
        'both': 'mediumorchid'
    }

    # Plot
    plt.figure(figsize=(15, 6))
    for label in grouped_df.columns:
        plt.plot(grouped_df.index,
                 grouped_df[label], label=label, color=colors[label], linewidth=2)

    plt.title('Seasonal Avalanche Frequency by Grouped Type (1 Dec – 15 Apr)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Season', fontsize=14)
    plt.ylabel('Number of Avalanches', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Avalanche Type', fontsize=10,
               title_fontsize=12, loc='upper right')
    plt.tight_layout()

    # Format asse X come "dd - mm"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d - %m'))
    plt.xticks(rotation=45)

    if savefig:
        outpath = plot_folder / 'avalanche_seasonality_by_type_grouped.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':

    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\")
    file = 'mod1_tarlenta.csv'

    plot_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\01_Mod1_statistics\\")

    data = data_folder/file

    df = pd.read_csv(data, sep=';', na_values=['NaN', '/', 'nan'])

    df['DataRilievo'] = pd.to_datetime(
        df['DataRilievo'], format="%d/%m/%Y")

    codice_file = 'codice_nivometeorologico.json'
    with open(data_folder/codice_file, 'r') as file:
        codice_nivometeo = json.load(file)

    savefig = False

    L1_counts(df, savefig)
    L1_classification(df, savefig)
    L2_classification(df, savefig)
    L3_classification(df, savefig)
    L4_classification(df, savefig)
    L5_classification(df, savefig)
    L6_classification(df, savefig)

    L1_timeline_season(df, savefig)
    L1_timeline_season_class(df, savefig)
    L1_period(df, savefig)

    L3L4_elevation_aspect_plot(df, savefig)

    L1_classification_bar(df, savefig)
    L2_classification_bar(df, savefig)
    avalanche_seasonality_by_type(df, savefig)
    avalanche_seasonality_by_type_2(df, savefig)
    avalanche_seasonality_by_type_weekly_grouped(df, savefig)

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
