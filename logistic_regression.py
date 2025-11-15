"""Logistic regression implementation using statsmodels."""

import numpy as np
from statsmodels.discrete.discrete_model import Logit
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionModel:
    """Wrapper for statsmodels Logit with sklearn-like interface."""
    
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.model = None
        self.result = None
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit logistic regression model."""
        self.model = Logit(y, X)
        self.result = self.model.fit(disp=0, maxiter=self.max_iter)
        self.coef_ = self.result.params
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.result.predict(X)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        return (self.predict_proba(X) > threshold).astype(int)
    
    def get_params(self):
        """Get model parameters."""
        return {
            'coefficients': self.coef_,
            'norm': np.linalg.norm(self.coef_) if self.coef_ is not None else None
        }


def train_logistic_regression(X_train, y_train, X_test=None, y_test=None, max_iter=1000):
    """
    Train logistic regression and return results.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        max_iter: Maximum iterations
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    try:
        model = LogisticRegressionModel(max_iter=max_iter)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        results = {
            'status': 'SUCCESS',
            'model': model,
            'train_accuracy': train_acc,
            'train_predictions': train_pred,
            'coefficients': model.coef_,
            'coef_norm': np.linalg.norm(model.coef_)
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
