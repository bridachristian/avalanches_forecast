# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:01:43 2024

@author: Christian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json


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

    plt.figure(figsize=(8, 8))
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

    class_distribution = df['L1'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of number and size of observed avalanches from L1')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L1_classification_of_magnitude_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
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

    class_distribution = df['L2'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
    plt.pie(class_distribution, labels=class_distribution.index,
            autopct=make_autopct(class_distribution), startangle=140)
    plt.title('Distribution of types of avalanches from L2')

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'L2_types_of_avalanches_pie.png'
        plt.savefig(outpath, dpi=300)
    else:
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

    class_distribution = df['L3'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
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

    class_distribution = df['L4'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
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

    class_distribution = df['L5'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
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

    class_distribution = df['L6'].value_counts()
    class_distribution = class_distribution.iloc[1:]  # exclude no avalanche

    class_distribution.index = class_distribution.index.map(index_mapping)

    plt.figure(figsize=(8, 8))
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
    plt.figure(figsize=(8, 8))
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
    '''
    Count avalanche days vs non avalanche days and plot a timeline
    grouped by month or year

    Parameters
    ----------
    df : dataframe of mod1 AINEVA
        the dataframe shold contain the columns L1

    savefig : optionla
     default = False

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

    # --- Calculate and plot number of avalanche days ---

    df['AvNoAv'] = df['L1'].apply(transform_value)
    df['NoAval'] = df['L1'].apply(transform_value)
    df['NaNDay'] = df['L1'].apply(transform_value)

    # Group by 'Stagione' and aggregate columns using different functions
    grouped_df = df.groupby('Stagione').agg({
        'AvNoAv': 'sum',             # Sum of avalanche days
        'NoAval': lambda x: x.eq(0).sum(),   # Count of 0 values
        'NaNDay': lambda x: x.isna().sum()   # Count of NaN values
    }).reset_index()

    grouped_df['TotalDays'] = grouped_df.drop(columns=['Stagione']).sum(axis=1)

    # Plotting
    plt.figure(figsize=(8, 8))

    avnoav_bar = plt.bar(grouped_df['Stagione'],
                         grouped_df['AvNoAv'], color="#44a5c2")
    noaval_bar = plt.bar(grouped_df['Stagione'], grouped_df['NoAval'],
                         bottom=grouped_df['AvNoAv'], color="#ffae49")
    nanday_bar = plt.bar(grouped_df['Stagione'], grouped_df['NaNDay'], bottom=np.add(
        grouped_df['AvNoAv'], grouped_df['NoAval']), color="#D3D3D3")

    plt.title('Number of avalanche days for each winter season')
    plt.xlabel('Season')
    plt.ylabel('Days')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping

    # Add color legend
    legend_labels = ['Avalanche Days', 'No Avalanche Days', 'NaN Days']
    legend_colors = ["#44a5c2", "#ffae49", "#D3D3D3"]
    plt.legend([plt.Rectangle((0, 0), 1, 1, color=color)
               for color in legend_colors], legend_labels)

    # Add vertical grid lines every 5 seasons
    for i, stagione in enumerate(grouped_df['Stagione']):
        if i % 5 == 0:  # Check if it's the 5th season and not the first one
            plt.axvline(x=stagione, color='gray', linestyle='--',
                        linewidth=0.5)  # Add vertical line

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'avalanche_number_days_plot.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()

    # --- Calculate and plot percentage of avalanche days ---

    grouped_df_percent = pd.DataFrame(grouped_df)

    # Calculate percentage for each column
    grouped_df_percent['AvNoAv'] = (
        grouped_df_percent['AvNoAv'] / grouped_df_percent['TotalDays']) * 100
    grouped_df_percent['NoAval'] = (
        grouped_df_percent['NoAval'] / grouped_df_percent['TotalDays']) * 100
    grouped_df_percent['NaNDay'] = (
        grouped_df_percent['NaNDay'] / grouped_df_percent['TotalDays']) * 100

    # Plotting
    plt.figure(figsize=(8, 8))

    av_bar = plt.bar(grouped_df_percent['Stagione'],
                     grouped_df_percent['AvNoAv'], color="#44a5c2")
    noaval_bar = plt.bar(grouped_df_percent['Stagione'], grouped_df_percent['NoAval'],
                         bottom=grouped_df_percent['AvNoAv'], color="#ffae49")
    nanday_bar = plt.bar(grouped_df_percent['Stagione'], grouped_df_percent['NaNDay'], bottom=np.add(
        grouped_df_percent['AvNoAv'], grouped_df_percent['NoAval']), color="#D3D3D3")

    plt.title('Percentange of avalanche days for each winter season')
    plt.xlabel('Season')
    plt.ylabel('% of season days')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping

    # Add color legend
    legend_labels = ['Avalanche Days Percent',
                     'No Avalanche Days Percent', 'NaN Days Percent']
    legend_colors = ["#44a5c2", "#ffae49", "#D3D3D3"]
    plt.legend([plt.Rectangle((0, 0), 1, 1, color=color)
               for color in legend_colors], legend_labels)

    # Add vertical grid lines every 5 seasons
    for i, stagione in enumerate(grouped_df['Stagione']):
        if i % 5 == 0:  # Check if it's the 5th season and not the first one
            plt.axvline(x=stagione, color='gray', linestyle='--',
                        linewidth=0.5)  # Add vertical line

    if savefig == True:
        # Save figure with high resolution (300 dpi)
        outpath = plot_folder / 'avalanche_percent_days_plot.png'
        plt.savefig(outpath, dpi=300)
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
    plt.figure(figsize=(8, 8))

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


def L3L4_classification(df, savefig=False):
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

    # DA COMPLETARE!!!!
    # # plot by aspects and elevation

    # # Plotting the data

    # directions = total.iloc[:, 0].values.tolist()
    # elevations = total.iloc[:, 1].values.tolist()
    # counts = total.iloc[:, 2].values.tolist()

    # angles = np.linspace(0, 2 * np.pi, 4,
    #                      endpoint=False).tolist()

    # # Convert wind speeds to colors
    # colors = [plt.cm.Blues(i / max(counts)) for i in counts]

    # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # bars = ax.bar(angles, 1, width=1.57, color=colors,
    #               edgecolor='grey', alpha=0.9)

    # ax.set_rticks([])
    # ax.grid(False)

    # # Add count labels to each sector
    # for i, bar in enumerate(bars):
    #     angle_rad = angles[i]
    #     angle_deg = np.degrees(angle_rad)
    #     if angle_deg >= 0 and angle_deg < 90:
    #         ha = 'center'
    #     elif angle_deg >= 90 and angle_deg < 180:
    #         ha = 'right'
    #     elif angle_deg >= 180 and angle_deg < 270:
    #         ha = 'center'
    #     else:
    #         ha = 'left'
    #     ax.text(angle_rad, 0.7, str(
    #         counts[i]), transform=ax.get_xaxis_transform(), ha=ha, va='center')

    # ax.set_theta_offset(np.pi / 2)
    # ax.set_theta_direction(-1)
    # ax.set_rlabel_position(0)
    # ax.set_xticks(np.radians([0, 90, 180, 270]))
    # ax.set_xticklabels(['N', 'E', 'S', 'W'])
    # plt.title(f'Geographical distribution of avalanche release aspect')

    # plt.show()

    aspects = small_df.groupby(['L3']).size().reset_index(name='count')
    aspects = aspects.iloc[1:]  # exclude no avalanche

    aspects = aspects[aspects['L3'] != 'all']

    # Plotting the data

    directions = aspects.iloc[:, 0].tolist()
    counts = aspects.iloc[:, 1].tolist()

    angles = np.linspace(0, 2 * np.pi, len(aspects), endpoint=False).tolist()

    # Convert wind speeds to colors
    colors = [plt.cm.Blues(i / max(counts)) for i in counts]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    bars = ax.bar(angles, 1, width=1.57, color=colors,
                  edgecolor='grey', alpha=0.9)

    ax.set_rticks([])
    ax.grid(False)

    # Add count labels to each sector
    for i, bar in enumerate(bars):
        angle_rad = angles[i]
        angle_deg = np.degrees(angle_rad)
        if angle_deg >= 0 and angle_deg < 90:
            ha = 'center'
        elif angle_deg >= 90 and angle_deg < 180:
            ha = 'right'
        elif angle_deg >= 180 and angle_deg < 270:
            ha = 'center'
        else:
            ha = 'left'
        ax.text(angle_rad, 0.7, str(
            counts[i]), transform=ax.get_xaxis_transform(), ha=ha, va='center')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(['N', 'E', 'S', 'W'])
    plt.title(f'Geographical distribution of avalanche release aspect')

    plt.show()


def main():
    '''
    Main function

    Returns
    -------
    None.

    '''

    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\")
    file = 'mod1_tarlenta.csv'

    plot_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\01_Mod1_statistics\\")

    data = data_folder/file

    df = pd.read_csv(data, sep=';', na_values=['NaN', '/', 'nan'])

    df['DataRilievo'] = pd.to_datetime(
        df['DataRilievo'], format="%d/%m/%Y")

    savefig = True

    codice_file = 'codice_nivometeorologico.json'
    with open(data_folder/codice_file, 'r') as file:
        codice_nivometeo = json.load(file)

    L1_counts(df, savefig)
    # L1_classification(df, savefig)
    L2_classification(df, savefig)
    L3_classification(df, savefig)
    L4_classification(df, savefig)
    L5_classification(df, savefig)
    L6_classification(df, savefig)

    L1_timeline_season(df, savefig)
    L1_timeline_season_class(df, savefig)
    L1_period(df, savefig)


if __name__ == '__main__':
    main()
