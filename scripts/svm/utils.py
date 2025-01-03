import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def save_outputfile(df, output_filepath):
    """
    Saves the given DataFrame to a CSV file at the specified output file path.

    This function saves the DataFrame `df` to a CSV file, where:
    - The DataFrame is saved with its index included.
    - The separator used in the CSV file is a semicolon (`;`).
    - Missing or NaN values are represented as `'NaN'` in the output file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved to a CSV file.
        output_filepath (str): The file path where the CSV file will be saved.

    Returns:
        None: This function performs an I/O operation and does not return a value.

    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        save_outputfile(df, 'output.csv')
        # Saves the df DataFrame to a file named 'output.csv' with semicolon separator and NaN representation.
    """
    df.to_csv(output_filepath, index=True, sep=';', na_rep='NaN')


def get_adjacent_values(arr, best_value):
    """
    Retrieves the previous, current, and next values in a 1D array based on the closest match to a given 'best_value'.

    This function finds the index of the element in the array that is closest to the given `best_value` (using `np.isclose`),
    and then returns the values at the previous, current, and next indices, ensuring that boundary conditions are handled safely.

    Args:
        arr (numpy.ndarray): A 1D numpy array from which to extract values.
        best_value (float): The value to which the closest value in `arr` will be matched.

    Returns:
        tuple: A tuple containing the previous value, the closest (current) value, and the next value from the array.
               If the closest value is at the start or end of the array, it will return the closest value itself
               as the previous or next value, respectively.

    Example:
        arr = np.array([1, 3, 7, 10, 15])
        best_value = 7
        prev_value, current_value, next_value = get_adjacent_values(
            arr, best_value)
        print(prev_value, current_value, next_value)  # Output: 3 7 10
    """

    # Find the index of the closest value to best_value
    idx = np.where(np.isclose(arr, best_value))[0][0]

    # Get previous, current, and next values safely
    prev_value = arr[idx - 1] if idx > 0 else arr[idx]
    next_value = arr[idx + 1] if idx < len(arr) - 1 else arr[idx]

    return prev_value, arr[idx], next_value


def detect_grid_type(values):
    """
    Detects if a list of values is linear, exponential, or neither.

    Parameters:
        values (list or array-like): A sequence of numerical values.

    Returns:
        str: 'linear', 'exponential', or 'neither'
    """
    values = np.array(values)
    if len(values) < 2:
        return "neither"

    # Check for linear progression
    diffs = np.diff(values)  # Consecutive differences
    if np.allclose(diffs, diffs[0]):
        return "linear"

    # Check for exponential progression
    ratios = values[1:] / values[:-1]  # Consecutive ratios
    if np.allclose(ratios, ratios[0]):
        return "exponential"

    return "neither"


def plot_scatter_original(X, y, title, palette={0: "blue", 1: "red"}):
    """
    Creates a scatter plot after applying nearmiss undersampling, showing class 1 points in the foreground
    and class 0 points in the background, with transparency applied.

    Parameters:
    - X: DataFrame, the input data with features.
    - y: Series, the target labels (binary: 0 or 1).
    - version: str, version of the undersampling technique.
    - n_neighbors: int, number of neighbors used in the undersampling technique.
    - palette: dict, custom color palette for class 0 and class 1 (default is blue for 0 and red for 1).
    """

    # Separate the points based on their class
    class_1_points = X[y == 1]  # Points where class == 1
    class_0_points = X[y == 0]  # Points where class == 0

    # Get the counts of each class
    class_counts = y.value_counts()
    class_0_count = class_counts[0]
    class_1_count = class_counts[1]

    colnames = X.columns
    # Create the jointplot
    g = sns.jointplot(data=X, x=colnames[0], y=colnames[1], hue=y, alpha=0,
                      kind='scatter', marginal_kws={'fill': True}, palette=palette, legend=False)

    # Plot class 1 first (foreground) with transparency
    sns.scatterplot(x=class_1_points.iloc[:, 0], y=class_1_points.iloc[:, 1], hue=y[y == 1],
                    palette={1: "red"}, alpha=0.3, ax=g.ax_joint, legend=True)

    # Plot class 0 second (background) with more transparency
    sns.scatterplot(x=class_0_points.iloc[:, 0], y=class_0_points.iloc[:, 1], hue=y[y == 0],
                    palette={0: "blue"}, alpha=0.3, ax=g.ax_joint, legend=True)

    # Add custom legend with counts
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=10, alpha=0.3,
                   label=f"Avalanche (n={class_1_count})"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, alpha=0.3,
                   label=f"No Avalanche (n={class_0_count})")
    ]
    g.ax_joint.legend(handles=handles, loc="best")

    # Add title and other layout adjustments
    g.fig.suptitle(title, fontsize=14)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    # Show the plot
    plt.show()


def plot_scatter_under_over_sampling(X, y, title, palette={0: "blue", 1: "red"}):
    """
    Creates a scatter plot after applying nearmiss undersampling, showing class 1 points in the foreground
    and class 0 points in the background, with transparency applied.

    Parameters:
    - X: DataFrame, the input data with features.
    - y: Series, the target labels (binary: 0 or 1).
    - version: str, version of the undersampling technique.
    - n_neighbors: int, number of neighbors used in the undersampling technique.
    - palette: dict, custom color palette for class 0 and class 1 (default is blue for 0 and red for 1).
    """

    colnames = X.columns

    # Separate the points based on their class
    class_1_points = X[y == 1]  # Points where class == 1
    class_0_points = X[y == 0]  # Points where class == 0

    # Get the counts of each class
    class_counts = y.value_counts()
    class_0_count = class_counts[0]
    class_1_count = class_counts[1]

    # Create the jointplot
    g = sns.jointplot(data=X, x=colnames[0], y=colnames[1], hue=y, alpha=0,
                      kind='scatter', marginal_kws={'fill': True}, palette=palette, legend=False)

    # Plot class 1 first (foreground) with transparency
    sns.scatterplot(x=class_1_points.iloc[:, 0], y=class_1_points.iloc[:, 1], hue=y[y == 1],
                    palette={1: "red"}, alpha=0.3, ax=g.ax_joint, legend=True)

    # Plot class 0 second (background) with more transparency
    sns.scatterplot(x=class_0_points.iloc[:, 0], y=class_0_points.iloc[:, 1], hue=y[y == 0],
                    palette={0: "blue"}, alpha=0.3, ax=g.ax_joint, legend=True)

    # Add custom legend with counts
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="red", markersize=10, alpha=0.3,
                   label=f"Avalanche (n={class_1_count})"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, alpha=0.3,
                   label=f"No Avalanche (n={class_0_count})")
    ]
    g.ax_joint.legend(handles=handles, loc="best")

    # Add title and other layout adjustments
    g.fig.suptitle(title, fontsize=14)
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    # Show the plot
    plt.show()


def plot_decision_boundary(X, y, model, title, palette={0: "blue", 1: "red"}):
    """
    Creates a scatter plot showing the decision boundary of a classifier, with a heatmap of the decision function.

    Parameters:
    - X: DataFrame, the input data with features.
    - y: Series, the target labels (binary: 0 or 1).
    - model: Trained classifier with a decision_function method.
    - title: str, the title of the plot.
    - palette: dict, custom color palette for class 0 and class 1 (default is blue for 0 and red for 1).
    """

    colnames = X.columns

    X_array = X.values.astype(float)
    y_array = y.values.astype(float)

    # Create a grid of points to evaluate the decision function
    xx, yy = np.meshgrid(np.linspace(X_array[:, 0].min(), X_array[:, 0].max(), 100),
                         np.linspace(X_array[:, 1].min(), X_array[:, 1].max(), 100))

    # Evaluate the decision function on the grid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Define a custom colormap: white at 0, blue below, red above
    cmap = LinearSegmentedColormap.from_list(
        'custom_colormap',
        [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
    )

    # norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

    # Create a heatmap of Z values
    plt.figure(figsize=(8, 6))
    # heatmap = plt.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
    heatmap = plt.contourf(xx, yy, Z, levels=50,
                           cmap=cmap, norm=norm, alpha=0.8)

    # Scatter plot of the data points
    plt.scatter(X_array[:, 0], X_array[:, 1],
                c=y_array, cmap='coolwarm', alpha=0.5)

    # Add contour lines for decision boundary at levels [-1, 0, 1]
    ax = plt.gca()
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               linestyles=['--', '-', '--'], colors='k')

    # Add color bar for the heatmap
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Decision Function Value (Z)')

    # Add axis labels
    plt.xlabel(colnames[0])
    plt.ylabel(colnames[1])

    # Add title
    plt.title(f'Decision function - {title}')

    # Add text box with best_C and best_gamma
    textstr = f'Best C: {model.C}\nBest Gamma: {model.gamma}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Show the plot
    plt.show()


def plot_threshold_scoring(X, y, X_test, y_test, model):
    """
    Creates a scatter plot showing the decision boundary of a classifier, with a heatmap of the decision function.

    Parameters:
    - X: DataFrame, the input data with features.
    - y: Series, the target labels (binary: 0 or 1).
    - model: Trained classifier with a decision_function method.
    - title: str, the title of the plot.
    - palette: dict, custom color palette for class 0 and class 1 (default is blue for 0 and red for 1).
    """

    colnames = X.columns

    X_array = X.values.astype(float)
    y_array = y.values.astype(float)

    clf2 = SVC(C=model.C, gamma=model.gamma, probability=True)
    clf2.fit(X, y)

    # Probability of the positive class
    proba = clf2.predict_proba(X_test)[:, 1]
    # Define a range of thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)

    # Store metrics for each threshold
    hk_scores, hss_scores, oa_scores = [], [], []
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    for t in thresholds:
        # Classify using the current threshold
        y_pred = (proba >= t).astype(int)

        # Compute metrics (example for HK, HSS, and OA)
        TP = np.sum((y_test == 1) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))

        # Heidke Skill Score (HSS)
        HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
        hss_scores.append(HSS)

        # Hanssen-Kuipers (HK)
        HK = TP / (TP + FN) - FP / (FP + TN)
        hk_scores.append(HK)

        # Overall Accuracy (OA)
        OA = accuracy_score(y_test, y_pred)
        oa_scores.append(OA)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, hk_scores, label="HK", linestyle='-', color='blue')
    plt.plot(thresholds, hss_scores, label="HSS", linestyle='--', color='red')
    plt.plot(thresholds, oa_scores, label="OA", linestyle=':', color='black')
    plt.xlabel("Threshold")
    plt.ylabel("Performance Measure")
    plt.title("Performance vs. Threshold")
    plt.legend()
    plt.grid()
    plt.show()

    optimal_threshold_hk = thresholds[np.argmax(hk_scores)]
    optimal_threshold_hss = thresholds[np.argmax(hss_scores)]

    print(f"Optimal Threshold (HK): {optimal_threshold_hk}")
    print(f"Optimal Threshold (HSS): {optimal_threshold_hss}")

    return optimal_threshold_hk


class PermutationImportanceWrapper(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, scoring='accuracy', n_repeats=10, random_state=None):
        self.estimator = estimator
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self._feature_importances = None

    def fit(self, X, y):
        # Fit the underlying estimator
        self.estimator.fit(X, y)
        # Compute permutation importance
        perm_importance = permutation_importance(
            self.estimator, X, y, scoring=self.scoring,
            n_repeats=self.n_repeats, random_state=self.random_state
        )
        self._feature_importances = perm_importance.importances_mean
        return self

    @property
    def feature_importances_(self):
        # Return feature importances after fitting
        return self._feature_importances

    def predict(self, X):
        # Delegate the prediction to the underlying estimator
        return self.estimator.predict(X)

    def predict_proba(self, X):
        # Optional: if you want to support probability predictions, add this method
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)
        else:
            raise AttributeError(
                "The estimator does not support probability prediction.")


def remove_correlated_features(X, y):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(
        X), columns=X.columns, index=X.index)

    # Calcola la matrice di correlazione
    corr_matrix = pd.concat([X_scaled, y], axis=1).corr()

    # Correlazione reciproca tra feature
    feature_corr = corr_matrix.loc[X.columns, X.columns]

    # Correlazione con il target
    target_corr = corr_matrix.loc[X.columns, y.name]

    # Trova le feature con alta correlazione reciproca (>0.9)
    high_corr_pairs = np.where((np.abs(feature_corr) > 0.9) & (
        np.triu(np.ones(feature_corr.shape), k=1)))

    # Lista di coppie di feature altamente correlate
    high_corr_feature_pairs = [(X.columns[i], X.columns[j])
                               for i, j in zip(*high_corr_pairs)]

    # Identifica le feature da rimuovere
    features_to_remove = set()
    for feature1, feature2 in high_corr_feature_pairs:
        # Confronta la correlazione di entrambe le feature con il target
        if abs(target_corr[feature1]) > abs(target_corr[feature2]):
            features_to_remove.add(feature2)
        else:
            features_to_remove.add(feature1)

    return list(features_to_remove)


def remove_low_variance(X, threshold=0.01):
    """
    Identify features with variance below a specified threshold.

    Parameters:
    - X: pandas DataFrame, feature matrix.
    - threshold: float, the variance threshold below which features are identified.

    Returns:
    - features_removed: list, names of the features to drop (if X is a DataFrame).
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "Input X must be a pandas DataFrame to return feature names.")

    # Fit the VarianceThreshold selector
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    # Identify low-variance features
    features_removed = X.columns[~selector.get_support()].tolist()

    return features_removed


def select_k_best(X, y, k=5):
    """
    Perform feature selection using SelectKBest with ANOVA F-test to select the top `k` features.

    Parameters:
    - X: pandas DataFrame, feature matrix.
    - y: pandas Series or array, target variable.
    - k: int, number of top features to select using SelectKBest (default=5).

    Returns:
    - selected_features: list, names of the selected features.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "Input X must be a pandas DataFrame to return feature names.")

    # Apply SelectKBest (ANOVA F-test)
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get the selected feature names
    selected_features = X.columns[selector.get_support()].tolist()

    return selected_features
