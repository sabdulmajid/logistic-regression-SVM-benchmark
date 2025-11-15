"""
Part 1 & 2: Logistic Regression and SVM Analysis on Dataset A
"""

from utils import load_dataset, print_section, print_subsection, to_signed_labels
from logistic_regression import train_logistic_regression
from svm import train_svm, analyze_svm_margins, reconstruct_weight_vector
import numpy as np


def main():
    X_train, y_train = load_dataset('A')
    
    print_section("DATASET A: LOGISTIC REGRESSION AND SVM COMPARISON")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_train.astype(int))}")
    
    # Part 1: Train all three methods
    print_section("PART 1: TRAINING THREE METHODS")
    
    print_subsection("1. Logistic Regression (statsmodels)")
    logit_results = train_logistic_regression(X_train, y_train)
    
    if logit_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {logit_results['train_accuracy']:.6f}")
        print(f"  Coefficient norm: {logit_results['coef_norm']:.6f}")
    else:
        print(f"✗ Status: FAILED - {logit_results['error']}")
    
    print_subsection("2. Soft-margin SVM (C=1.0)")
    svm_soft_results = train_svm(X_train, y_train, C=1.0)
    
    if svm_soft_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {svm_soft_results['train_accuracy']:.6f}")
        print(f"  Support vectors: {svm_soft_results['n_support_vectors']}")
        print(f"  Coefficient norm: {svm_soft_results['coef_norm']:.6f}")
    else:
        print(f"✗ Status: FAILED - {svm_soft_results['error']}")
    
    print_subsection("3. Hard-margin SVM (C=inf)")
    svm_hard_results = train_svm(X_train, y_train, C=float('inf'))
    
    if svm_hard_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {svm_hard_results['train_accuracy']:.6f}")
        print(f"  Support vectors: {svm_hard_results['n_support_vectors']}")
        print(f"  Coefficient norm: {svm_hard_results['coef_norm']:.6f}")
    else:
        print(f"✗ Status: FAILED - {svm_hard_results['error']}")
    
    print_section("MATHEMATICAL EXPLANATION")
    print("Hard-margin SVM fails: Dataset A is not linearly separable.")
    print("Optimization requires y_i(w·x_i) ≥ 1 for ALL points (infeasible constraint set).")
    print("\nRemedies: Use soft-margin SVM, kernel methods, or alternative classifiers.")
    print("\nLogistic Regression vs SVM: Similar accuracy, different loss functions.")
    print("Logistic uses log-loss (probabilities), SVM uses hinge loss (sparse support vectors).")
    
    # Part 2: Detailed SVM analysis
    if svm_soft_results['status'] == 'SUCCESS':
        print_section("PART 2: SOFT-MARGIN SVM DETAILED ANALYSIS")
        
        w = svm_soft_results['coefficients']
        y_signed = to_signed_labels(y_train)
        
        print_subsection("Margin Analysis")
        margin_analysis = analyze_svm_margins(X_train, y_train, w)
        
        print(f"Points with y_i(w·x_i) ≤ 1: {margin_analysis['count_lte_1']} / {len(y_train)}")
        print(f"  Percentage: {margin_analysis['percentage_lte_1']:.2f}%")
        print(f"  Within margin (< 1): {margin_analysis['count_lt_1']}")
        print(f"  On margin (≈ 1): {margin_analysis['count_on_margin']}")
        print(f"  Beyond margin (> 1): {margin_analysis['count_beyond']}")
        
        print_subsection("2D Caricature Interpretation")
        print("Not perfectly linearly separable. Support vectors cluster near decision boundary.")
        print("Some points violate margins; most correctly classified beyond margin region.")
        
        print_subsection("Weight Vector Representation")
        support_idx = svm_soft_results['support_vectors']
        dual_coef = svm_soft_results['dual_coef']
        
        w_reconstructed = reconstruct_weight_vector(X_train, support_idx, dual_coef)
        error = np.linalg.norm(w - w_reconstructed)
        
        print(f"Representation: w = Σ αᵢyᵢxᵢ for i ∈ support vectors")
        print(f"Number of points required: {len(support_idx)}")
        print(f"Percentage of dataset: {100*len(support_idx)/len(y_train):.2f}%")
        print(f"Reconstruction error: {error:.10e}")
        print(f"\nRepresenter theorem: Solution lies in span of support vectors.")


if __name__ == "__main__":
    main()
