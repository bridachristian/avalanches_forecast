# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:21:54 2024

@author: Christian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns


def timeline(datafile):
    df = pd.read_csv(datafile, sep=';')
    df['Inizio'] = pd.to_datetime(
        df['Inizio'], format="%d/%m/%Y")
    df['Fine'] = pd.to_datetime(
        df['Fine'], format="%d/%m/%Y")

    df['diff'] = df['Fine'] - df['Inizio']

    P = df[(df['Param'] == 'P') | (df['Param'] == 'mod.1')]
    Ta = df[(df.Param == 'Ta') | (df['Param'] == 'mod.1')]
    RH = df[(df.Param == 'RH') | (df['Param'] == 'mod.1')]
    WD = df[(df.Param == 'WD') | (df['Param'] == 'mod.1')]
    WV = df[(df.Param == 'WV') | (df['Param'] == 'mod.1')]
    Patm = df[(df.Param == 'Patm') | (df['Param'] == 'mod.1')]
    RAD = df[(df.Param == 'RAD') | (df['Param'] == 'mod.1')]
    # mod1 = df[(df.Param == 'mod.1') | (df['Param'] == 'mod.1')]

    return P, Ta, RH, WD, WV, Patm, RAD


def plot_timeline(dataframe, savefig=False):

    global plot_folder

    # Define colors for each station
    colors = {
        'T0063': '#FF5733', 'T0064': '#33FF57',
        'T0065': '#3357FF', 'T0066': '#FF33A6',
        'T0068': '#FF8C33', 'T0308': '#33FFF3',
        'T0366': '#8C33FF', 'T0372': '#FFC733',
        'T0380': '#33FF8C', 'T0473': '#FF3333',
        '1PEI': 'red'
    }

    # Sort and reset index
    dataframe = dataframe.sort_values(by='Inizio').reset_index()

    # Define a figure and axis
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each station's data with fixed colors
    for station, station_df in dataframe.groupby('CodiceStazione'):
        if station not in colors:
            print(
                f"Station {station} is not in the colors dictionary. Assigning default color.")
            color = 'grey'  # Assign a default color for stations not in the dictionary
        else:
            color = colors[station]

        for _, row in station_df.sort_values(by='Inizio').reset_index().iterrows():
            start_year = row['Inizio'].year
            duration = row['diff'].days / 365

            ax.broken_barh([(start_year, duration)],
                           (row['Elev'], 70),
                           facecolors=(color),
                           label=station, alpha=0.5)
            # Display elevation
            if row['CodiceStazione'] == '1PEI':
                ax.text(
                    1980 + 0.5, row['Elev'] - 55, str(row['CodiceStazione']), color=color, fontsize=14)
                pei_label_added = True
            elif row['CodiceStazione'] != '1PEI':
                # Display elevation
                ax.text(start_year + 0.5, row['Elev'] - 55,
                        str(row['CodiceStazione']), color=color, fontsize=14)

    # # Remove duplicate labels in the legend
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize=10)

    # Set labels, title, and grid
    variable = dataframe[dataframe['CodiceStazione'] != '1PEI']['Parametro'].unique()[
        0]
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Elevation (m)', fontsize=14)
    plt.ylim([1000, 3300])
    plt.yticks(range(1000, 3300, 250), fontsize=12)
    plt.xticks(range(1980, 2025, 5), fontsize=12)
    plt.title(
        f'Timeline of {variable} and Mod.1 Peio Tarlenta', fontsize=20, weight='bold')
    plt.grid(True, axis='x')
    plt.tight_layout()

    # Save or show figure
    if savefig == True:
        outpath = plot_folder / f'timeline_{variable}.png'
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()


def plot_distance_matrix(file, savefig=False, format='png'):
    global plot_folder

    # Load data
    df = pd.read_csv(file, sep=';')

    # Pivot to square matrix and convert to kilometers
    df_square = df.pivot_table(
        index='InputID', columns='TargetID', values='Distance', fill_value=0) / 1000

    # Sort by distances from '1PEI'
    distances_from_1PEI = df_square.loc['1PEI']
    sorted_ids = distances_from_1PEI.sort_values().index.tolist()
    df_square = df_square.loc[sorted_ids, sorted_ids]

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(df_square, dtype=bool), k=1)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.set(style='white')

    ax = sns.heatmap(
        df_square,
        mask=mask,
        cmap='Blues',
        annot=True,
        fmt='.1f',
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Distance [km]'},
        square=True,
        xticklabels=True,
        yticklabels=True,
        annot_kws={'fontsize': 14}
    )

    # Remove axis labels (titles)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Title and labels
    ax.set_title('Distance Matrix Ordered by Proximity to 1PEI',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    # Save or show
    if savefig:
        outpath = plot_folder / f'distanceMatrix.{format}'
        plt.savefig(outpath, dpi=600, bbox_inches='tight')
    else:
        plt.show()


def main():
    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\Metadati")
    file = 'metadati.csv'
    metadata = data_folder/file

    plot_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\05_Plots\\02_Metadata_AWS\\")

    P, Ta, RH, WD, WV, Patm, RAD = timeline(metadata)

    savefig = True
    plot_timeline(P, savefig)
    plot_timeline(Ta, savefig)
    plot_timeline(RH, savefig)
    plot_timeline(WD, savefig)
    plot_timeline(WV, savefig)
    plot_timeline(Patm, savefig)
    plot_timeline(RAD, savefig)

    distance_file = 'distance_matrix_2.csv'

    distance_matrix = data_folder/distance_file

    savefig = True
    plot_distance_matrix(distance_matrix, savefig)


if __name__ == '__main__':
    main()
