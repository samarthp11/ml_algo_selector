# ML Algorithm Selector — Meta-Learning System

A meta-learning system that automatically recommends the best-performing machine learning
algorithm for a given dataset, based on the dataset's own characteristics — removing the
need to manually try every algorithm through trial and error.

## Problem

Given a new dataset, which ML algorithm (Logistic Regression, KNN, SVM, or Random Forest)
is likely to perform best? Normally this requires manually training and evaluating every
candidate — slow and repetitive. This project builds a "meta-classifier" that predicts the
best algorithm directly from dataset characteristics, without running all candidates first.

## Approach

1. **Base datasets**: Iris, Wine, Breast Cancer, and Digits (from scikit-learn) are used as
   training examples for the meta-learning system.
2. **Meta-feature extraction**: For each dataset, five characteristics are computed —
   number of samples, number of features, mean, variance, and class imbalance ratio.
3. **Candidate evaluation**: Four algorithms (Logistic Regression, KNN, SVM, Random Forest)
   are trained and evaluated on each dataset; the best-performing one is recorded.
4. **Meta-dataset construction**: Each dataset's meta-features are paired with its best
   algorithm to form a new "meta-dataset."
5. **Meta-classifier**: A Decision Tree is trained on the meta-dataset to learn the mapping
   from dataset characteristics → best algorithm.
6. **Validation on unseen data**: The trained meta-classifier is tested on the Olivetti
   Faces dataset (never seen during training) to check that it generalizes.

## Result

The meta-classifier correctly predicted **SVM** as the best algorithm for the unseen
Olivetti Faces dataset, matching the empirically-measured best result (~90% accuracy),
confirming the system generalizes beyond its training data.

| Dataset | Logistic Regression | SVM | Random Forest | KNN | Best Algorithm |
|---|---|---|---|---|---|
| Iris | 67% | 75% | 91% | 78% | Random Forest |
| Olivetti Faces | 76% | 90% | 88% | 75% | SVM |
| Wine | 89% | 87% | 92% | 90% | Random Forest |

## Tech Stack

- Python
- scikit-learn
- Pandas
- NumPy

## Setup & Usage

```bash
git clone https://github.com/samarthp11/ml_algo_selector.git
cd ml-algo-selector
pip install -r requirements.txt
jupyter notebook ml_algo_selector.ipynb
```

Run all cells in order. The final cell prints the meta-classifier's predicted best
algorithm for the held-out Olivetti Faces dataset.

## Future Extensions

- Add more meta-features (skewness, feature correlation)
- Test additional base algorithms (Gradient Boosting, Neural Networks)
- Expand meta-dataset with more benchmark datasets for a more robust meta-classifier
