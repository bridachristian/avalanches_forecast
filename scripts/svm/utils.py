import numpy as np
import pandas as pd
from scripts.svm.data_loading import load_data
from scripts.svm.undersampling_methods import undersampling_random, undersampling_random_timelimited, undersampling_nearmiss
from scripts.svm.oversampling_methods import oversampling_random, oversampling_smote, oversampling_adasyn, oversampling_svmsmote
from scripts.svm.svm_training import cross_validate_svm, tune_train_evaluate_svm, train_evaluate_final_svm
from scripts.svm.evaluation import (plot_learning_curve, plot_confusion_matrix,
                         plot_roc_curve, permutation_ranking, evaluate_svm_with_feature_selection)

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