"""Utility functions for loading data and computing metrics."""

import numpy as np
import pandas as pd


def load_dataset(dataset_name, include_test=False):
    """
    Load training (and optionally test) data for a given dataset.
    
    Args:
        dataset_name: Name of dataset ('A', 'B', 'C', or 'D')
        include_test: Whether to load test data
    
    Returns:
        Tuple of (X_train, y_train) or (X_train, y_train, X_test, y_test)
    """
    X_train = pd.read_csv(f'data/X_train_{dataset_name}.csv', header=None).values
    
    y_file = f'data/Y_train_{dataset_name}.csv'
    try:
        y_train = pd.read_csv(y_file, header=None).values.ravel()
    except FileNotFoundError:
        y_train = pd.read_csv(f'data/y_train_{dataset_name}.csv', header=None).values.ravel()
    
    if include_test:
        X_test = pd.read_csv(f'data/X_test_{dataset_name}.csv', header=None).values
        try:
            y_test = pd.read_csv(f'data/Y_test_{dataset_name}.csv', header=None).values.ravel()
        except FileNotFoundError:
            y_test = pd.read_csv(f'data/y_test_{dataset_name}.csv', header=None).values.ravel()
        return X_train, y_train, X_test, y_test
    
    return X_train, y_train


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def to_signed_labels(y):
    """Convert binary labels {0, 1} to {-1, +1}."""
    return np.where(y == 0, -1, 1)


def print_section(title, char='='):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(char * 80)


def print_subsection(title, char='-'):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print(char * 80)
