import numpy as np
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import matplotlib.pyplot as plt
from scripts.svm.evaluation import plot_before_nearmiss, plot_after_nearmiss
import seaborn as sns


def undersampling_random(X, y):
    """
    Perform random undersampling on the input data to balance the class distribution.

    This function uses the RandomUnderSampler from the imbalanced-learn library to 
    randomly downsample the majority class in the dataset. It then prints the class 
    distribution before and after undersampling, allowing you to compare the effect.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The RandomUnderSampler uses a fixed random seed (42) to ensure reproducibility.
        - The original and resampled class distributions are printed using the Counter class 
          from the collections module.
    """
    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Check the new class distribution
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def undersampling_random_timelimited(X, y, Ndays=10):
    """
    Perform random undersampling within a time-limited window before avalanche events.

    This function applies random undersampling on the data, but limits the undersampling 
    to only the data within a time window (default of 10 days) before each avalanche event. 
    It ensures that the data used for resampling includes only non-avalanche events within 
    the specified time window, while also preserving the avalanche events.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.
        Ndays (int): The number of days before an avalanche event to consider for undersampling 
                     (default is 10 days).

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The function creates a time window of `Ndays` before each avalanche event and 
          applies undersampling within this window.
        - The function uses RandomUnderSampler from the imbalanced-learn library to balance 
          the class distribution by randomly undersampling the majority class.
        - The random undersampling is applied to non-avalanche events within the selected 
          time window.
        - The original and resampled class distributions are printed using the Counter class 
          from the collections module.

    Example:
        X_res, y_res = undersampling_random_timelimited(X, y, Ndays=10)
    """
    # Convert y to a DataFrame to retain the index
    y_df = pd.DataFrame(y).copy()
    y_df.columns = ['AvalDay']

    # Find indices where avalanche events occur
    avalanche_dates = y_df[y_df['AvalDay'] == 1].index

    # Create a mask to keep data within 10 days before each avalanche event
    mask = pd.Series(False, index=y.index)

    # Mark the 10 days before each avalanche event
    for date in avalanche_dates:
        mask.loc[date - pd.Timedelta(days=Ndays):date] = True

    # Separate the data into non-avalanche events within the 10-day window and other data
    X_window = X[mask]
    y_window = y[mask]
    X_other = X[~mask]
    y_other = y[~mask]

    # Select only non-avalanche events from the window
    non_avalanche_mask = (y_window == 0)
    X_non_avalanche = X_window[non_avalanche_mask]
    y_non_avalanche = y_window[non_avalanche_mask]

    avalanche_mask = (y_window == 1)
    X_avalanche = X_window[avalanche_mask]
    y_avalanche = y_window[avalanche_mask]

    X_new = pd.concat([X_non_avalanche, X_avalanche])
    y_new = pd.concat([y_non_avalanche, y_avalanche])

    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_new, y_new)

    # Check the new class distribution
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_res))

    return X_res, y_res


def undersampling_nearmiss(X, y, version=1, n_neighbors=1):
    """
    Perform undersampling using the NearMiss algorithm.

    This function applies the NearMiss undersampling technique to balance the class distribution
    in the dataset. NearMiss selects samples from the majority class that are closest to the 
    minority class samples. The algorithm can operate in different versions (1, 2, or 3), 
    where each version differs in how the closest majority class samples are selected.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.
        version (int): The version of the NearMiss algorithm to use:
                       - version=1: Selects samples that are closest to the minority class.
                       - version=2: Selects samples farthest from the minority class.
                       - version=3: Selects samples that are closest to the k-th nearest neighbor 
                         from the minority class.
        n_neighbors (int): The number of neighbors to use when selecting samples. 
                           The default is 3.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - NearMiss undersampling aims to balance the class distribution by undersampling the 
          majority class based on its proximity to the minority class samples.
        - The function uses the `NearMiss` class from the imbalanced-learn library for resampling.
        - The class distributions before and after resampling are printed using the `Counter` 
          class from the collections module.

    Example:
        X_res, y_res = undersampling_nearmiss(X, y, version=2, n_neighbors=5)
    """
    # Initialize the NearMiss object with the chosen version
    nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)

    # Apply NearMiss undersampling
    try:
        X_res, y_res = nearmiss.fit_resample(X, y)
    except ValueError as e:
        print(f"Error during resampling: {e}")
        # If an error occurs, return the original dataset
        return X, y

    # Display the class distribution after undersampling
    print("NearMiss: Original class distribution:", Counter(y))
    print("NearMiss: Resampled class distribution:", Counter(y_res))

    counts_orig = Counter(y)
    counts_nm = Counter(y_res)

    plot_before_nearmiss(X, y, palette={0: "blue", 1: "red"})

    # Version 1, n_neighbors = 1
    # version = 1
    # n_neighbors = 1
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    plot_after_nearmiss(X_res, y_res, version, n_neighbors,
                        palette={0: "blue", 1: "red"})

    # # Version 2, n_neighbors = 1
    # version = 2
    # n_neighbors = 1
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    # plot_after_nearmiss(X_res, y_res, version, n_neighbors,
    #                     palette={0: "blue", 1: "red"})

    # # Version 3, n_neighbors = 1
    # version = 3
    # n_neighbors = 1
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    # plot_after_nearmiss(X_res, y_res, version, n_neighbors,
    #                     palette={0: "blue", 1: "red"})

    # # Version 3, n_neighbors = 2
    # version = 3
    # n_neighbors = 2
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    # plot_after_nearmiss(X_res, y_res, version, n_neighbors,
    #                     palette={0: "blue", 1: "red"})

    # # Version 3, n_neighbors = 5
    # version = 3
    # n_neighbors = 5
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    # plot_after_nearmiss(X_res, y_res, version, n_neighbors,
    #                     palette={0: "blue", 1: "red"})

    # # Version 3, n_neighbors = 10
    # version = 3
    # n_neighbors = 10
    # nearmiss = NearMiss(version=version, n_neighbors=n_neighbors)
    # X_res, y_res = nearmiss.fit_resample(X, y)

    # plot_after_nearmiss(X_res, y_res, version, n_neighbors,
    #                     palette={0: "blue", 1: "red"})

    return X_res, y_res
