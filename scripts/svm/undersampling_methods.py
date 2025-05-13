import numpy as np
import pandas as pd
from collections import Counter
from imblearn.under_sampling import (RandomUnderSampler, NearMiss, CondensedNearestNeighbour,
                                     EditedNearestNeighbours, ClusterCentroids, TomekLinks)
import matplotlib.pyplot as plt

from scripts.svm.utils import plot_scatter_original, plot_scatter_under_over_sampling
from scripts.svm.feature_engineering import transform_features, transform_penetration_ratio
from sklearn.base import BaseEstimator, TransformerMixin


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
    sampling_method = 'Random Undersampling'

    # Apply random undersampling
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    # Check the new class distribution
    print("Original class distribution:", Counter(y))
    print("Resampled class distribution:", Counter(y_res))

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

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

    sampling_method = f'Random time-limited {Ndays} days Undersampling'
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

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_nearmiss(X, y, version=1, n_neighbors=3):
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
    sampling_method = 'NearMiss Undersampling'

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

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}: vers.{version}, n.neighbgurs: {n_neighbors}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_cnn(X, y):
    sampling_method = 'Condensed Nearest Neighbour Undersampling'
    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(X, y)

    # # Get the target count (minority class count)
    # target_count = min(Counter(y_res).values())

    # # Find indices for each class
    # indices_0 = np.where(y_res == 0)[0]
    # indices_1 = np.where(y_res == 1)[0]

    # # Randomly sample the majority class (class 0) to match the minority class size
    # indices_0 = np.random.choice(indices_0, size=target_count, replace=False)

    # # Use all samples from the minority class
    # # No sampling needed, just trim if needed
    # indices_1 = indices_1[:target_count]

    # # Combine the balanced indices
    # balanced_indices = np.hstack((indices_0, indices_1))

    # # Shuffle the indices to mix both classes
    # np.random.shuffle(balanced_indices)

    # # Create the balanced dataset
    # X_balanced = X_res.iloc[balanced_indices, :]
    # y_balanced = y_res[balanced_indices]

    # Optionally plot the scatter plot if the dataset has 2 features
    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_enn(X, y, version=1, n_neighbors=3):
    sampling_method = 'Edited Nearest Neighbour Undersampling'
    enn = EditedNearestNeighbours(sampling_strategy='auto')
    X_res, y_res = enn.fit_resample(X, y)

    # Ensure equal number of samples (483) in each class
    # target_count = Counter(y_res)[1]

    # # Find indices for each class
    # indices_0 = np.where(y_res == 0)[0]
    # indices_1 = np.where(y_res == 1)[0]

    # # Randomly sample from the larger class to match the target count
    # indices_0 = np.random.choice(indices_0, size=target_count, replace=False)
    # indices_1 = np.random.choice(indices_1, size=target_count, replace=False)

    # # Combine the balanced indices
    # balanced_indices = np.hstack((indices_0, indices_1))

    # # Create the balanced dataset
    # X_balanced = X_res.iloc[balanced_indices, :]
    # y_balanced = y_res[balanced_indices]

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})

    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_clustercentroids(X, y):
    sampling_method = 'Cluster Centroids Undersampling'
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_resample(X, y)

    # # Ensure equal number of samples (483) in each class
    # target_count = Counter(y_res)[1]

    # # Find indices for each class
    # indices_0 = np.where(y_res == 0)[0]
    # indices_1 = np.where(y_res == 1)[0]

    # # Randomly sample from the larger class to match the target count
    # indices_0 = np.random.choice(indices_0, size=target_count, replace=False)
    # indices_1 = np.random.choice(indices_1, size=target_count, replace=False)

    # # Combine the balanced indices
    # balanced_indices = np.hstack((indices_0, indices_1))

    # # Create the balanced dataset
    # X_balanced = X_res.iloc[balanced_indices, :]
    # y_balanced = y_res[balanced_indices]

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_tomeklinks(X, y, version=1, n_neighbors=3):
    sampling_method = 'Tomek Links Undersampling'
    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X, y)

    # # Ensure equal number of samples (483) in each class
    # target_count = Counter(y_res)[1]

    # # Find indices for each class
    # indices_0 = np.where(y_res == 0)[0]
    # indices_1 = np.where(y_res == 1)[0]

    # # Randomly sample from the larger class to match the target count
    # indices_0 = np.random.choice(indices_0, size=target_count, replace=False)
    # indices_1 = np.random.choice(indices_1, size=target_count, replace=False)

    # # Combine the balanced indices
    # balanced_indices = np.hstack((indices_0, indices_1))

    # # Create the balanced dataset
    # X_balanced = X_res.iloc[balanced_indices, :]
    # y_balanced = y_res[balanced_indices]

    if X.shape[1] == 2:
        plot_scatter_original(X, y,
                              title=f'Original Distribution before {sampling_method}',
                              palette={0: "blue", 1: "red"})

        plot_scatter_under_over_sampling(X_res, y_res,
                                         title=f'{sampling_method}',
                                         palette={0: "blue", 1: "red"})
    else:
        print("Skipping scatter plot: X does not have exactly 2 features.")

    return X_res, y_res


def undersampling_clustercentroids_v2(X, y):
    """
    Applica ClusterCentroids su feature continue, mantenendo binarie, neve fresca e PR normalizzate interpretabili.
    Ignora colonne non trasformabili. Se non ci sono feature continue, bilancia solo con feature binarie.

    Input:
        X: pd.DataFrame - tutte le feature
        y: pd.Series - target binario

    Output:
        X_resampled: DataFrame bilanciato
        y_resampled: Series bilanciata
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import pairwise_distances_argmin_min
    from imblearn.under_sampling import ClusterCentroids
    import numpy as np
    import pandas as pd

    # Feature di interesse
    precip_features = ['HNnum', 'HN_2d', 'HN_3d', 'HN_5d',
                       'Precip_1d', 'Precip_2d', 'Precip_3d', 'Precip_5d']
    bin_features = ['HNnum_bin', 'HN_2d_bin', 'HN_3d_bin', 'HN_5d_bin',
                    'Precip_1d_bin', 'Precip_2d_bin', 'Precip_3d_bin', 'Precip_5d_bin']
    pr_cols = ['PR']
    pr_transformed = {}

    X_copy = X.copy()
    y = y.loc[X_copy.index]  # garantisce indice compatibile

    # Tenta di trasformare le feature (opzionale)
    try:
        X_transformed = transform_features(X_copy)
    except Exception as e:
        print(f"[WARN] Errore durante transform_features(): {e}")
        X_transformed = X_copy.copy()

    # Funzione di trasformazione PR
    def transform_pr_column(series):
        try:
            upper_clip = series.quantile(0.99)
            clipped = series.clip(upper=upper_clip)
            log_vals = np.log1p(clipped)
            std = StandardScaler().fit_transform(log_vals.values.reshape(-1, 1)).flatten()
            norm = MinMaxScaler().fit_transform(series.values.reshape(-1, 1)).flatten()
            return pd.Series(std, index=series.index), pd.Series(norm, index=series.index)
        except Exception as e:
            print(f"[WARN] Colonna PR non trasformabile: {e}")
            return None, None

    # Trasformazioni PR
    for col in pr_cols:
        if col in X_transformed.columns:
            std_col, norm_col = transform_pr_column(X_transformed[col])
            if std_col is not None:
                X_transformed[f'{col}_std'] = std_col
                pr_transformed[f'{col}_norm'] = norm_col

    # Selezione feature binarie e precipitazioni
    bin_in_X = [col for col in bin_features if col in X_transformed.columns]
    fresh_in_X = [
        col for col in precip_features if col in X_transformed.columns]

    # Seleziona solo feature continue valide
    cont_features = X_transformed.drop(
        columns=bin_in_X + fresh_in_X, errors='ignore')
    cont_features = cont_features.select_dtypes(include=[np.number])
    cont_features = cont_features.dropna(axis=1, how='any')
    cont_features = cont_features.reset_index(drop=True)

    if cont_features.shape[1] == 0:
        # Fallback: uso solo binarie se niente di continuo è disponibile
        print("[INFO] Nessuna feature continua utilizzabile: uso solo binarie.")
        X_subset = X_transformed[bin_in_X].copy()
        cc = ClusterCentroids(random_state=42)
        X_resampled_bin, y_resampled = cc.fit_resample(X_subset, y)

        X_bin_for_dist = X_transformed[bin_in_X].reset_index(drop=True)
        closest_idxs, _ = pairwise_distances_argmin_min(
            X_resampled_bin, X_bin_for_dist)

        # Recupera e reintegra colonne di interesse
        X_bin_resampled = X_transformed[bin_in_X].iloc[closest_idxs].reset_index(
            drop=True)
        X_fresh_resampled = X_transformed[fresh_in_X].iloc[closest_idxs].reset_index(
            drop=True)

        X_resampled = pd.DataFrame(
            X_resampled_bin, columns=X_bin_for_dist.columns)
        X_resampled = pd.concat([X_fresh_resampled, X_resampled], axis=1)

        return X_resampled.reset_index(drop=True), pd.Series(y_resampled, name=y.name)

    # Applica ClusterCentroids su continue
    cc = ClusterCentroids(random_state=42)
    X_resampled_cont, y_resampled = cc.fit_resample(cont_features, y)

    # Trova indici dei campioni originali più vicini ai centroidi
    closest_idxs, _ = pairwise_distances_argmin_min(
        X_resampled_cont, cont_features)

    # Recupera e reintegra colonne di interesse
    X_bin_resampled = X_transformed[bin_in_X].iloc[closest_idxs].reset_index(
        drop=True)
    X_fresh_resampled = X_transformed[fresh_in_X].iloc[closest_idxs].reset_index(
        drop=True)
    pr_norm_resampled = {
        k: v.iloc[closest_idxs].reset_index(drop=True) for k, v in pr_transformed.items()
    }

    # Costruzione finale del dataframe
    X_resampled = pd.DataFrame(X_resampled_cont, columns=cont_features.columns)
    X_resampled = pd.concat([X_resampled,
                             X_fresh_resampled,
                             X_bin_resampled], axis=1)

    # Rimuove PR std, aggiunge PR norm interpretabile
    for col in pr_cols:
        if f'{col}_std' in X_resampled.columns:
            X_resampled = X_resampled.drop(columns=f'{col}_std')
        if f'{col}_norm' in pr_norm_resampled:
            X_resampled[col] = pr_norm_resampled[f'{col}_norm']

    return X_resampled.reset_index(drop=True), pd.Series(y_resampled, name=y.name)


class CustomUndersampler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Add any parameters here if needed

    def fit(self, X, y=None):
        # Save y if needed; we’ll pass it to transform
        self.y_ = y
        return self

    def transform(self, X):
        if self.y_ is None:
            raise ValueError(
                "y cannot be None in transform. Ensure fit() was called with y.")

        # Call your custom undersampling function
        X_resampled, y_resampled = undersampling_clustercentroids_v2(
            X, self.y_)

        # Save y_resampled for possible later use
        self.y_resampled_ = y_resampled
        return X_resampled

    def get_resampled_y(self):
        if hasattr(self, 'y_resampled_'):
            return self.y_resampled_
        else:
            raise AttributeError("The transformer has not been fitted yet.")
