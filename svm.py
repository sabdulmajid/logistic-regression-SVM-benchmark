"""Support Vector Machine implementation using sklearn."""

import numpy as np
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


def train_svm(X_train, y_train, X_test=None, y_test=None, C=1.0, max_iter=10000):
    """
    Train linear SVM and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        C: Regularization parameter (1.0 for soft-margin, float('inf') for hard-margin)
        max_iter: Maximum iterations
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    try:
        model = SVC(kernel='linear', C=C, max_iter=max_iter)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        results = {
            'status': 'SUCCESS',
            'model': model,
            'train_accuracy': train_acc,
            'train_predictions': train_pred,
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0],
            'support_vectors': model.support_,
            'n_support_vectors': len(model.support_),
            'dual_coef': model.dual_coef_[0],
            'coef_norm': np.linalg.norm(model.coef_[0])
        }
        
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            test_acc = np.mean(test_pred == y_test)
            results['test_accuracy'] = test_acc
            results['test_predictions'] = test_pred
        
        return results
        
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e),
            'model': None
        }


def analyze_svm_margins(X, y, w, compute_details=True):
    """
    Analyze SVM margin structure.
    
    Args:
        X: Feature matrix
        y: Binary labels {0, 1}
        w: SVM coefficient vector
        compute_details: Whether to compute detailed statistics
    
    Returns:
        Dictionary with margin analysis results
    """
    y_signed = np.where(y == 0, -1, 1)
    
    inner_products = X @ w
    scaled_products = y_signed * inner_products
    
    count_lte_1 = np.sum(scaled_products <= 1.0 + 1e-9)
    count_lt_1 = np.sum(scaled_products < 1.0 - 1e-9)
    count_on_margin = np.sum(np.abs(scaled_products - 1.0) < 1e-9)
    count_beyond = len(y) - count_lte_1
    
    results = {
        'scaled_products': scaled_products,
        'count_lte_1': count_lte_1,
        'count_lt_1': count_lt_1,
        'count_on_margin': count_on_margin,
        'count_beyond': count_beyond,
        'percentage_lte_1': 100 * count_lte_1 / len(y)
    }
    
    if compute_details:
        results.update({
            'min_scaled_product': np.min(scaled_products),
            'max_scaled_product': np.max(scaled_products),
            'mean_scaled_product': np.mean(scaled_products),
            'median_scaled_product': np.median(scaled_products)
        })
    
    return results


def reconstruct_weight_vector(X_train, support_indices, dual_coef):
    """
    Reconstruct weight vector from support vectors.
    
    Args:
        X_train: Training features
        support_indices: Indices of support vectors
        dual_coef: Dual coefficients from SVM
    
    Returns:
        Tuple of (reconstructed_w, reconstruction_error)
    """
    w_reconstructed = np.zeros(X_train.shape[1])
    
    for i, idx in enumerate(support_indices):
        w_reconstructed += dual_coef[i] * X_train[idx]
    
    return w_reconstructed
