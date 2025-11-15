"""
Part 4b: Ensemble Methods (Bagging and Random Forest) on Dataset D
"""

from utils import load_dataset, print_section, print_subsection
from decision_trees import run_ensemble_comparison
import numpy as np


def main():
    X_train, y_train, X_test, y_test = load_dataset('D', include_test=True)
    
    print_section("DATASET D: ENSEMBLE METHODS")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    n_estimators = 101
    max_depth = 3
    n_runs = 11
    
    print(f"\nConfiguration:")
    print(f"  Number of trees per ensemble: {n_estimators}")
    print(f"  Maximum depth per tree: {max_depth}")
    print(f"  Number of independent runs: {n_runs}")
    print(f"  Random Forest max_features: √d ≈ {int(np.sqrt(X_train.shape[1]))} features per split")
    
    # Run comparison
    print_section("RUNNING ENSEMBLE EXPERIMENTS")
    print("Training ensembles (this may take a moment)...")
    
    results = run_ensemble_comparison(X_train, y_train, X_test, y_test,
                                     n_runs=n_runs, n_estimators=n_estimators,
                                     max_depth=max_depth)
    
    # Display results
    print_subsection("Bagging Results")
    for i, acc in enumerate(results['bagging']['accuracies'], 1):
        print(f"  Run {i:2d}: {acc:.6f}")
    
    print(f"\nStatistics:")
    print(f"  Median:  {results['bagging']['median']:.6f}")
    print(f"  Minimum: {results['bagging']['min']:.6f}")
    print(f"  Maximum: {results['bagging']['max']:.6f}")
    print(f"  Mean:    {results['bagging']['mean']:.6f}")
    print(f"  Std Dev: {results['bagging']['std']:.6f}")
    
    print_subsection("Random Forest Results")
    for i, acc in enumerate(results['random_forest']['accuracies'], 1):
        print(f"  Run {i:2d}: {acc:.6f}")
    
    print(f"\nStatistics:")
    print(f"  Median:  {results['random_forest']['median']:.6f}")
    print(f"  Minimum: {results['random_forest']['min']:.6f}")
    print(f"  Maximum: {results['random_forest']['max']:.6f}")
    print(f"  Mean:    {results['random_forest']['mean']:.6f}")
    print(f"  Std Dev: {results['random_forest']['std']:.6f}")
    
    # Comparison
    print_section("COMPARISON")
    print(f"\n{'Method':<20} {'Median':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 70)
    print(f"{'Bagging':<20} {results['bagging']['median']:.6f}   "
          f"{results['bagging']['min']:.6f}   {results['bagging']['max']:.6f}   "
          f"{results['bagging']['std']:.6f}")
    print(f"{'Random Forest':<20} {results['random_forest']['median']:.6f}   "
          f"{results['random_forest']['min']:.6f}   {results['random_forest']['max']:.6f}   "
          f"{results['random_forest']['std']:.6f}")
    
    diff = abs(results['bagging']['median'] - results['random_forest']['median'])
    bagging_better = results['bagging']['median'] > results['random_forest']['median']
    
    print_section("ANALYSIS")
    print("Bagging: All features. Random Forest: √d features per split.")
    print("Both reduce variance via bootstrap aggregation.\n")
    
    if diff < 0.01:
        print(f"Similar performance (diff={diff:.4f}). Dataset doesn't benefit from feature randomization.")
    elif bagging_better:
        print(f"Bagging better (diff={diff:.4f}). All features provide valuable information.")
    else:
        print(f"Random Forest better (diff={diff:.4f}). Feature randomization decorrelates trees.")
    
    print("\nEnsembles vs single trees: Higher accuracy, lower variance, better generalization.")
    print("Implicit regularization through averaging reduces overfitting.")


if __name__ == "__main__":
    main()
