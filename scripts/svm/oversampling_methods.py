import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE
from scripts.svm.utils import plot_scatter_original, plot_scatter_under_over_sampling


def oversampling_random(X, y):
    """
    Perform oversampling using the RandomOverSampler algorithm.

    This function applies the RandomOverSampler technique to balance the class distribution
    in the dataset by randomly duplicating samples from the minority class. This method 
    increases the number of samples in the minority class by randomly choosing examples 
    and duplicating them.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The RandomOverSampler technique randomly replicates the minority class samples 
          to balance the class distribution.
        - The class distributions before and after oversampling are printed using the `Counter` 
          class from the collections module.
        - The random_state parameter ensures reproducibility by controlling the randomization.

    Example:
        X_res, y_res = oversampling_random(X, y)
    """
    sampling_method = 'Random Oversampling'
    # Initialize the RandomOverSampler object
    ros = RandomOverSampler(random_state=42)

    # Apply Random oversampling
    X_res, y_res = ros.fit_resample(X, y)

    # Display the class distribution before and after oversampling
    print("RandomOverSampler: Original class distribution:", Counter(y))
    print("RandomOverSampler: Resampled class distribution:", Counter(y_res))

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


def oversampling_smote(X, y):
    """
    Perform oversampling using the SMOTE (Synthetic Minority Over-sampling Technique) algorithm.

    This function applies the SMOTE technique to balance the class distribution in the dataset
    by generating synthetic samples for the minority class. SMOTE creates new examples by
    interpolating between existing minority class samples, which helps to improve model 
    generalization and avoid overfitting due to simple duplication.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The SMOTE algorithm synthesizes new data points for the minority class by selecting 
          two or more similar instances and creating synthetic examples that are combinations 
          of the features of these instances.
        - The `sampling_strategy='minority'` argument ensures that only the minority class
          will be oversampled.
        - The `random_state=42` ensures reproducibility of the synthetic sample generation.

    Example:
        X_res, y_res = oversampling_smote(X, y)
    """
    sampling_method = 'SMOTE Oversampling'

    # Initialize the SMOTE object
    smote = SMOTE(sampling_strategy='minority', random_state=42)

    # Apply SMOTE oversampling
    X_res, y_res = smote.fit_resample(X, y)

    # Display the class distribution before and after SMOTE
    print("SMOTE: Original class distribution:", Counter(y))
    print("SMOTE: Resampled class distribution:", Counter(y_res))

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


def oversampling_adasyn(X, y):
    """
    Perform oversampling using the ADASYN (Adaptive Synthetic Sampling) algorithm.

    ADASYN is an oversampling technique that generates synthetic samples for the minority class,
    with an adaptive approach that focuses on generating more synthetic data for minority class
    instances that are harder to classify. ADASYN differs from SMOTE in that it considers the density
    of the minority class instances and generates synthetic samples based on their difficulty.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The `sampling_strategy='minority'` argument ensures that only the minority class
          will be oversampled.
        - The `random_state=42` ensures reproducibility of the synthetic sample generation.
        - ADASYN focuses on generating more synthetic samples in regions where the minority class
          is sparse or hard to classify, unlike SMOTE which generates synthetic samples uniformly.

    Example:
        X_res, y_res = oversampling_adasyn(X, y)
    """

    sampling_method = 'ADASYN Oversampling'

    # Initialize the ADASYN object
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)

    # Apply ADASYN oversampling
    X_res, y_res = adasyn.fit_resample(X, y)

    # Display the class distribution before and after ADASYN
    print("ADASYN: Original class distribution:", Counter(y))
    print("ADASYN: Resampled class distribution:", Counter(y_res))

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


def oversampling_svmsmote(X, y):
    """
    Perform oversampling using the SVMSMOTE (Support Vector Machine Synthetic Minority Oversampling Technique) algorithm.

    SVMSMOTE is an extension of the SMOTE (Synthetic Minority Over-sampling Technique) algorithm that generates synthetic
    samples by considering the support vectors from a Support Vector Machine classifier to create more informative synthetic
    samples for the minority class. It uses the decision boundary of the classifier to guide synthetic sample generation,
    which helps in handling difficult-to-classify instances more effectively.

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix containing the input data.
        y (pd.Series or np.ndarray): The target labels corresponding to the data points.

    Returns:
        tuple: A tuple (X_res, y_res) containing the resampled feature matrix and target labels.
               - X_res: The resampled feature matrix.
               - y_res: The resampled target labels.

    Notes:
        - The `sampling_strategy='minority'` argument ensures that only the minority class will be oversampled.
        - The `random_state=42` ensures reproducibility of the synthetic sample generation.
        - SVMSMOTE considers the decision boundary of a Support Vector Machine (SVM) classifier when generating synthetic samples,
          making it especially useful for imbalanced datasets with difficult-to-classify instances.

    Example:
        X_res, y_res = oversampling_svmsmote(X, y)
    """

    sampling_method = 'SVMSMOTE Oversampling'
    # Initialize the SVMSMOTE object
    svmsmote = SVMSMOTE(sampling_strategy='minority', random_state=42)

    # Apply SVMSMOTE oversampling
    X_res, y_res = svmsmote.fit_resample(X, y)

    # Display the class distribution before and after SVMSMOTE
    print("SVMSMOTE: Original class distribution:", Counter(y))
    print("SVMSMOTE: Resampled class distribution:", Counter(y_res))

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
