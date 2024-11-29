import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC


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
        prev_value, current_value, next_value = get_adjacent_values(arr, best_value)
        print(prev_value, current_value, next_value)  # Output: 3 7 10
    """

    # Find the index of the closest value to best_value
    idx = np.where(np.isclose(arr, best_value))[0][0]

    # Get previous, current, and next values safely
    prev_value = arr[idx - 1] if idx > 0 else arr[idx]
    next_value = arr[idx + 1] if idx < len(arr) - 1 else arr[idx]

    return prev_value, arr[idx], next_value


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

    # Create a heatmap of Z values
    plt.figure(figsize=(8, 6))
    heatmap = plt.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)

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
