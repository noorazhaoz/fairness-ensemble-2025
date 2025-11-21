"""
Quickstart demo for the Adult dataset with a simple global mixture
between a performance-oriented model and a simple one-feature model.

This script:
    1. Loads and preprocesses the Adult dataset via load_adult()
    2. Trains a Random Forest classifier (performance model) on all features
    3. Trains a simple Logistic Regression model using only a single feature
       (for demonstration)
    4. Searches over a scalar weight alpha in [0, 1] to combine the two
       prediction scores
    5. Evaluates accuracy and a group fairness metric (demographic parity
       difference) on the test set

The goal is to provide a compact, self-contained example of how a simple
mixture between a strong model and a simple one-feature model can be
optimized to balance performance and fairness.

Usage:
    python demo/quickstart_adult.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.data_loader import load_adult
from metrics.fairness import demographic_parity_diff


def main():
    # ------------------------------------------------------------------
    # 1. Load Adult dataset
    # ------------------------------------------------------------------
    # We assume load_adult returns:
    #   X_train, X_test, y_train, y_test, s_train, s_test, preproc_pipeline
    X_train, X_test, y_train, y_test, s_train, s_test, preproc = load_adult()

    print("\n=== Adult Quickstart Demo (global mixture with 1-feature model) ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # ------------------------------------------------------------------
    # 2. Performance model: Random Forest on all features
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=0,
    )
    rf.fit(X_train, y_train)
    print("Random Forest (performance model) trained.")

    # Predicted probabilities from the performance model
    y_rf_train = rf.predict_proba(X_train)[:, 1]
    y_rf_test = rf.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 3. Simple model: Logistic Regression on a single feature
    # ------------------------------------------------------------------
    # For demonstration, we use the FIRST feature column only.
    # In your experiments, this can be replaced with a specific feature
    # (e.g., 'hours-per-week') by selecting the corresponding column.
    X_train_simple = X_train[:, [0]]  # single feature
    X_test_simple = X_test[:, [0]]    # same feature on test

    lr_simple = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
    )
    lr_simple.fit(X_train_simple, y_train)
    print("Logistic Regression (1-feature simple model) trained.")

    y_simple_train = lr_simple.predict_proba(X_train_simple)[:, 1]
    y_simple_test = lr_simple.predict_proba(X_test_simple)[:, 1]

    # ------------------------------------------------------------------
    # 4. Search over alpha in [0, 1] for a simple trade-off
    # ------------------------------------------------------------------
    lambda_fair = 1.0  # trade-off parameter (demo value)
    alpha_grid = np.linspace(0.0, 1.0, 21)  # 0.00, 0.05, ..., 1.00

    best_obj = np.inf
    best_alpha = None
    best_acc = None
    best_dp = None

    print("\nSearching over alpha to balance accuracy and demographic parity difference ...")

    for alpha in alpha_grid:
        # Combine the two probability scores
        y_mix_test = alpha * y_rf_test + (1.0 - alpha) * y_simple_test
        y_mix_pred = (y_mix_test >= 0.5).astype(int)

        # Performance: accuracy
        acc = accuracy_score(y_test, y_mix_pred)

        # Fairness: demographic parity difference (your metric)
        dp_diff = demographic_parity_diff(
            y_true=y_test,
            y_pred=y_mix_pred,
            sensitive=s_test,
        )

        # Objective: (1 - accuracy) + lambda * |dp_diff|
        obj = (1.0 - acc) + lambda_fair * abs(dp_diff)

        if obj < best_obj:
            best_obj = obj
            best_alpha = alpha
            best_acc = acc
            best_dp = dp_diff

    # ------------------------------------------------------------------
    # 5. Baseline: RF only (no mixture) for comparison
    # ------------------------------------------------------------------
    y_rf_pred = (y_rf_test >= 0.5).astype(int)
    rf_acc = accuracy_score(y_test, y_rf_pred)
    rf_dp = demographic_parity_diff(
        y_true=y_test,
        y_pred=y_rf_pred,
        sensitive=s_test,
    )

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print("\n=== Results on Adult Test Set ===")
    print("Random Forest only:")
    print(f"  Accuracy:                   {rf_acc:.4f}")
    print(f"  Demographic parity diff:    {rf_dp:.4f}")

    print("\nBest global mixture (RF + simple 1-feature model):")
    print(f"  Best alpha:                 {best_alpha:.2f}")
    print(f"  Accuracy at best alpha:     {best_acc:.4f}")
    print(f"  Demographic parity diff:    {best_dp:.4f}")
    print(f"  Objective value:            {best_obj:.4f}")
    print(f"  (lambda_fair = {lambda_fair})")

    print("\nAdult quickstart demo complete.\n")


if __name__ == "__main__":
    main()
