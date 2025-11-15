"""
Part 4a: Decision Tree Analysis on Dataset D
"""

from utils import load_dataset, print_section, print_subsection
from decision_trees import evaluate_tree_depths
import matplotlib.pyplot as plt
import numpy as np


def main():
    X_train, y_train, X_test, y_test = load_dataset('D', include_test=True)
    
    print_section("DATASET D: DECISION TREE ANALYSIS")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes: {np.unique(y_train)}")
    
    # Evaluate across depths and criteria
    print_section("EVALUATING DECISION TREES")
    print("Training trees with varying max_depth for each criterion...")
    
    criteria = ['entropy', 'gini', 'log_loss']
    max_depths = list(range(0, 21))
    
    results = evaluate_tree_depths(X_train, y_train, X_test, y_test,
                                   criteria=criteria, max_depth_range=max_depths)
    
    # Print results summary
    print_subsection("Results Summary")
    for criterion in criteria:
        best_idx = np.argmax(results[criterion]['test'])
        best_depth = max_depths[best_idx]
        best_acc = results[criterion]['test'][best_idx]
        print(f"{criterion:12s} - Best depth: {best_depth:2d}, Test accuracy: {best_acc:.6f}")
    
    # Create visualization
    print_section("CREATING VISUALIZATIONS")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    criterion_names = {
        'log_loss': 'Misclassification Error',
        'gini': 'Gini Index',
        'entropy': 'Entropy'
    }
    
    for idx, criterion in enumerate(criteria):
        ax = axes[idx]
        
        ax.plot(max_depths, results[criterion]['train'],
                marker='o', label='Training', linewidth=2, markersize=4)
        ax.plot(max_depths, results[criterion]['test'],
                marker='s', label='Test', linewidth=2, markersize=4)
        
        ax.set_xlabel('Maximum Depth', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(criterion_names[criterion], fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    plt.savefig('decision_trees_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: decision_trees_analysis.png")
    
    print_section("OBSERVATIONS")
    print("Loss functions: Entropy and Gini perform similarly (smooth, differentiable).")
    print("\nShallow (0-3): Underfitting, high bias.")
    print("Optimal (4-8): Test accuracy peaks, good generalization.")
    print("Deep (9-20): Training → 100%, test plateaus (overfitting signature).")
    print("\nKey insight: Widening train/test gap indicates overfitting.")


if __name__ == "__main__":
    main()
