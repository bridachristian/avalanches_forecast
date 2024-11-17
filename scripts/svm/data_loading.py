import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob

def load_data(filepath):
    """
    Load and clean a dataset from the specified CSV file.

    This function reads a CSV file containing data about some time-related measurements,
    cleans the data by handling missing values, and processes the 'DataRilievo' column 
    to ensure it is in the correct datetime format. The resulting DataFrame is indexed 
    by the 'DataRilievo' column for easier time-series analysis.

    Args:
        filepath (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A cleaned DataFrame with 'DataRilievo' as the datetime index.

    Note:
        - The 'DataRilievo' column is converted to datetime format with the format '%Y-%m-%d'.
        - Missing values in the dataset are replaced by NaN for specific placeholders ('NaN', '/', '//', '///').
        - The 'Stagione' column is dropped (this line is commented out).
    """
    # Read the CSV file with custom separators and missing value placeholders
    mod1 = pd.read_csv(filepath, sep=';', na_values=['NaN', '/', '//', '///'])

    # Optional: Drop the 'Stagione' column if needed (currently commented out)
    # mod1 = mod1.drop(columns=['Stagione'])

    # Convert the 'DataRilievo' column to datetime format
    mod1['DataRilievo'] = pd.to_datetime(
        mod1['DataRilievo'], format='%Y-%m-%d')

    # Set 'DataRilievo' as the index of the DataFrame
    mod1.set_index('DataRilievo', inplace=True)

    return mod1