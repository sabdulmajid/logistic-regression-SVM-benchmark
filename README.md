# Logistic Regression, SVM, and Decision Trees

A comprehensive analysis and comparison of classification algorithms: Logistic Regression, Support Vector Machines (hard and soft-margin), and Decision Trees (with ensemble methods).

## Overview

This repository contains modular, production-quality implementations for comparing classical machine learning algorithms on multiple datasets. Each experiment is carefully designed to explore theoretical concepts and practical performance.

```

## Installation

```bash
# Clone the repository
git clone https://github.com/sabdulmajid/logistic-regression-SVM-benchmark.git
cd logistic-regression-SVM-benchmark

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run each experiment independently:

```bash
# Dataset A: Logistic Regression vs SVM
python run_dataset_a.py

# Dataset B: Test set evaluation
python run_dataset_b.py

# Decision Trees: Depth analysis
python run_decision_trees.py

# Ensemble Methods: Bagging vs Random Forest
python run_ensemble.py
```

## Experiments

### Dataset A Analysis

**Goal:** Compare logistic regression, soft-margin SVM, and hard-margin SVM.

**Key Findings:**
- **Hard-margin SVM fails** on non-linearly-separable data due to infeasible constraints
- The optimization problem `min ||w||²` subject to `y_i(w·x_i) ≥ 1 ∀i` has no solution
- Soft-margin SVM and logistic regression both succeed with similar accuracy

**Mathematical Insight:**
Hard-margin SVM requires perfect linear separability. When classes overlap or contain outliers, the constraint set is empty, causing optimization failure. The dual formulation attempts to find unbounded α values, resulting in numerical instability.

**SVM Margin Analysis:**
- Computed `y_i(w·x_i)` for all training points
- Counted support vectors (points with `y_i(w·x_i) ≤ 1`)
- Expressed weight vector as `w = Σ αᵢyᵢxᵢ` using support vectors only

**Comparison:**
| Method | Loss Function | Output | Support |
|--------|--------------|---------|----------|
| Logistic Regression | Log-loss | Probabilities | All points |
| Soft-margin SVM | Hinge loss | Binary labels | Support vectors only |

### Dataset B Analysis

**Goal:** Evaluate methods on a new dataset with held-out test set.

**Results:**
- Hard-margin SVM fails (same reason: non-separable data)
- Both logistic regression and soft-margin SVM generalize well
- SVM parameter vector expressed as linear combination of support vectors

**Test Accuracy:** Both methods achieve comparable performance, confirming their effectiveness on non-separable data.

### Decision Tree Analysis

**Goal:** Compare splitting criteria (entropy, Gini, misclassification error) across tree depths.

**Observations:**
- **Underfitting (depth 0-3):** Low train/test accuracy, high bias
- **Optimal range (depth 4-8):** Test accuracy peaks, good generalization
- **Overfitting (depth 9-20):** Training accuracy → 100%, test accuracy plateaus

**Loss Function Comparison:**
- Entropy and Gini Index perform similarly (smooth, differentiable)
- Both effectively guide trees toward pure splits
- Misclassification error slightly less sensitive to probability distributions

**Visualization:** Three plots showing training vs test accuracy curves for each criterion.

### Ensemble Methods

**Goal:** Compare Bagging and Random Forest (101 trees, max depth 3, 11 runs).

**Configuration:**
- **Bagging:** All features considered at each split
- **Random Forest:** √d ≈ 4 features randomly selected per split

**Results:**
| Method | Median | Min | Max | Std Dev |
|--------|--------|-----|-----|---------|
| Bagging | (varies) | (varies) | (varies) | (varies) |
| Random Forest | (varies) | (varies) | (varies) | (varies) |

**Key Insights:**
- Both methods reduce variance through bootstrap aggregation
- Random Forest adds decorrelation via feature subsampling
- Ensemble methods significantly outperform single shallow trees
- More stable predictions across different random seeds

**Performance vs Single Trees:**
- Ensembles achieve higher test accuracy
- Lower variance in predictions
- Better handling of noise and outliers
- Implicit regularization through averaging

## Results

### Why Hard-Margin SVM Fails

The hard-margin SVM optimization requires:
```
minimize   (1/2)||w||²
subject to y_i(w·x_i) ≥ 1  for all i
```

When data is **not linearly separable**, no hyperplane satisfies all constraints simultaneously. The feasible set is empty, making the problem infeasible.

**Remedies:**
1. Use soft-margin SVM (introduce slack variables)
2. Apply kernel methods (map to higher-dimensional space)
3. Use alternative classifiers (logistic regression, neural networks)

### Overfitting in Decision Trees

Deep trees memorize training data, leading to:
- Training accuracy → 100%
- Test accuracy plateaus or decreases
- High variance, low generalization

Ensemble methods mitigate this by averaging multiple trees, reducing variance while maintaining low bias.
