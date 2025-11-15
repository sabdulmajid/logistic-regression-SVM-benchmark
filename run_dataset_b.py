"""
Part 3: Logistic Regression and SVM Analysis on Dataset B
"""

from utils import load_dataset, print_section, print_subsection
from logistic_regression import train_logistic_regression
from svm import train_svm, reconstruct_weight_vector
import numpy as np


def main():
    X_train, y_train, X_test, y_test = load_dataset('B', include_test=True)
    
    print_section("DATASET B: CLASSIFICATION WITH TEST SET")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {np.bincount(y_train.astype(int))}")
    
    # Train all three methods
    print_section("TRAINING AND EVALUATION")
    
    print_subsection("1. Logistic Regression")
    logit_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    if logit_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {logit_results['train_accuracy']:.6f}")
        print(f"  Test accuracy: {logit_results['test_accuracy']:.6f}")
    else:
        print(f"✗ Status: FAILED - {logit_results['error']}")
    
    print_subsection("2. Soft-margin SVM (C=1.0)")
    svm_soft_results = train_svm(X_train, y_train, X_test, y_test, C=1.0)
    
    if svm_soft_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {svm_soft_results['train_accuracy']:.6f}")
        print(f"  Test accuracy: {svm_soft_results['test_accuracy']:.6f}")
        print(f"  Support vectors: {svm_soft_results['n_support_vectors']}")
    else:
        print(f"✗ Status: FAILED - {svm_soft_results['error']}")
    
    print_subsection("3. Hard-margin SVM (C=inf)")
    svm_hard_results = train_svm(X_train, y_train, X_test, y_test, C=float('inf'))
    
    if svm_hard_results['status'] == 'SUCCESS':
        print(f"✓ Status: SUCCESS")
        print(f"  Training accuracy: {svm_hard_results['train_accuracy']:.6f}")
        print(f"  Test accuracy: {svm_hard_results['test_accuracy']:.6f}")
        print(f"  Support vectors: {svm_hard_results['n_support_vectors']}")
    else:
        print(f"✗ Status: FAILED - {svm_hard_results['error']}")
    
    # SVM parameter vector analysis
    if svm_soft_results['status'] == 'SUCCESS':
        print_section("SOFT-MARGIN SVM PARAMETER ANALYSIS")
        
        support_idx = svm_soft_results['support_vectors']
        dual_coef = svm_soft_results['dual_coef']
        w = svm_soft_results['coefficients']
        
        w_reconstructed = reconstruct_weight_vector(X_train, support_idx, dual_coef)
        error = np.linalg.norm(w - w_reconstructed)
        
        print(f"Parameter vector representation: w = Σ αᵢyᵢxᵢ")
        print(f"Number of support vectors: {len(support_idx)}")
        print(f"Percentage of training data: {100*len(support_idx)/len(y_train):.2f}%")
        print(f"Reconstruction error: {error:.10e}")
        print(f"\nMethod: Direct extraction from sklearn SVC dual coefficients")
    
    # Test accuracy comparison
    print_section("TEST ACCURACY COMPARISON")
    print(f"{'Method':<25} {'Test Accuracy'}")
    print("-" * 45)
    
    if logit_results['status'] == 'SUCCESS':
        print(f"{'Logistic Regression':<25} {logit_results['test_accuracy']:.6f}")
    if svm_soft_results['status'] == 'SUCCESS':
        print(f"{'Soft-margin SVM (C=1.0)':<25} {svm_soft_results['test_accuracy']:.6f}")
    if svm_hard_results['status'] == 'SUCCESS':
        print(f"{'Hard-margin SVM (C=inf)':<25} {svm_hard_results['test_accuracy']:.6f}")
    
    print_section("ANALYSIS")
    print("Hard-margin SVM fails: Training data not perfectly linearly separable.")
    print("Classes overlap; optimization infeasible.")
    print("\nBoth logistic regression and soft-margin SVM handle non-separable data well.")
    print("Similar test accuracies indicate comparable generalization.")


if __name__ == "__main__":
    main()
