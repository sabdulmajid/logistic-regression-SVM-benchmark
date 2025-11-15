"""Decision tree and ensemble methods implementation."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')


def train_decision_tree(X_train, y_train, X_test, y_test, criterion='entropy', 
                       max_depth=None, random_state=42):
    """Train a single decision tree. Uses CPU (sklearn trees don't support GPU)."""
    if max_depth == 0:
        train_pred = np.full(len(y_train), np.bincount(y_train.astype(int)).argmax())
        test_pred = np.full(len(y_test), np.bincount(y_train.astype(int)).argmax())
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        
        return {
            'model': None,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'max_depth': 0
        }
    
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    
    return {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'max_depth': max_depth
    }


def evaluate_tree_depths(X_train, y_train, X_test, y_test, 
                        criteria=['entropy', 'gini', 'log_loss'],
                        max_depth_range=range(0, 21)):
    """
    Evaluate decision trees across multiple depths and criteria.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        criteria: List of splitting criteria to evaluate
        max_depth_range: Range of max_depth values to try
    
    Returns:
        Dictionary with results for each criterion
    """
    results = {criterion: {'train': [], 'test': []} for criterion in criteria}
    
    for criterion in criteria:
        for max_depth in max_depth_range:
            result = train_decision_tree(
                X_train, y_train, X_test, y_test,
                criterion=criterion, max_depth=max_depth
            )
            results[criterion]['train'].append(result['train_accuracy'])
            results[criterion]['test'].append(result['test_accuracy'])
    
    return results


def train_bagging_ensemble(X_train, y_train, X_test, y_test,
                          n_estimators=101, max_depth=3, criterion='entropy',
                          random_state=42):
    """Train bagging ensemble. n_jobs=-1 enables parallel training on all CPU cores."""
    base_tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )
    
    bagging = BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    bagging.fit(X_train, y_train)
    test_pred = bagging.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    
    return {
        'model': bagging,
        'test_accuracy': test_acc
    }


def train_random_forest(X_train, y_train, X_test, y_test,
                       n_estimators=101, max_depth=3, max_features=4,
                       criterion='entropy', random_state=42):
    """Train random forest. n_jobs=-1 enables parallel training on all CPU cores."""
    base_tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state
    )
    
    rf = BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    test_pred = rf.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    
    return {
        'model': rf,
        'test_accuracy': test_acc
    }


def run_ensemble_comparison(X_train, y_train, X_test, y_test,
                           n_runs=11, n_estimators=101, max_depth=3):
    """
    Compare bagging and random forest over multiple runs.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_runs: Number of times to run each method
        n_estimators: Number of trees in each ensemble
        max_depth: Maximum depth of trees
    
    Returns:
        Dictionary with statistics for both methods
    """
    n_features = X_train.shape[1]
    max_features_rf = int(np.sqrt(n_features))
    
    bagging_accuracies = []
    rf_accuracies = []
    
    for run in range(n_runs):
        bagging_result = train_bagging_ensemble(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=run
        )
        bagging_accuracies.append(bagging_result['test_accuracy'])
        
        rf_result = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features_rf, random_state=run
        )
        rf_accuracies.append(rf_result['test_accuracy'])
    
    bagging_accuracies = np.array(bagging_accuracies)
    rf_accuracies = np.array(rf_accuracies)
    
    return {
        'bagging': {
            'accuracies': bagging_accuracies,
            'median': np.median(bagging_accuracies),
            'min': np.min(bagging_accuracies),
            'max': np.max(bagging_accuracies),
            'mean': np.mean(bagging_accuracies),
            'std': np.std(bagging_accuracies)
        },
        'random_forest': {
            'accuracies': rf_accuracies,
            'median': np.median(rf_accuracies),
            'min': np.min(rf_accuracies),
            'max': np.max(rf_accuracies),
            'mean': np.mean(rf_accuracies),
            'std': np.std(rf_accuracies)
        },
        'max_features_rf': max_features_rf
    }
